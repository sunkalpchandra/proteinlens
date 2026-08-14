"""Train the attention pooling head on frozen ESM-2 residue embeddings.

The language model stays frozen. A small additive-attention pooler and a linear
classifier are trained jointly to predict UniProt family labels on a subset of
the *train split only*. The classifier is discarded; the pooler is kept and
used to produce the corpus's attention-pooled embeddings and per-residue
attention weights (an interpretability signal).

Early stopping uses a within-train holdout (family-stratified). The proper
leakage-audited evaluation of pooling strategies happens later in the probe
benchmark; this script only fits pooler parameters.

Usage:
    python scripts/train_attention_pooler.py [--max-classes 150] [--per-family 25]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.splitting import load_splits  # noqa: E402
from ml.tracking import log_experiment  # noqa: E402
from models.encoder import ESM2Encoder, resolve_device  # noqa: E402
from models.pooling import AttentionPooling  # noqa: E402


def pad_batch(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = [t.shape[0] for t in tensors]
    padded = torch.zeros(len(tensors), max(lengths), tensors[0].shape[1])
    mask = torch.zeros(len(tensors), max(lengths), dtype=torch.bool)
    for i, t in enumerate(tensors):
        padded[i, : t.shape[0]] = t
        mask[i, : t.shape[0]] = True
    return padded, mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/proteins.parquet"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument("--out", type=Path, default=Path("data/embeddings/attention_pooler.pt"))
    parser.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--max-classes", type=int, default=120)
    parser.add_argument("--min-members", type=int, default=8)
    parser.add_argument("--per-family", type=int, default=20)
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.corpus)
    splits, _ = load_splits(args.splits)
    df["split"] = df["accession"].map(splits)
    train = df[(df["split"] == "train") & df["family"].notna()]

    counts = train["family"].value_counts()
    classes = counts[counts >= args.min_members].head(args.max_classes).index.tolist()
    pool = train[train["family"].isin(classes)]
    sampled = (
        pool.groupby("family", group_keys=False)[pool.columns]
        .apply(lambda g: g.sample(min(len(g), args.per_family), random_state=args.seed))
        .reset_index(drop=True)
    )
    class_to_idx = {c: i for i, c in enumerate(classes)}
    labels_all = sampled["family"].map(class_to_idx).to_numpy()
    print(f"Pooler training set: {len(sampled)} proteins, {len(classes)} family classes")

    # --- Encode once with the frozen LM, keep residue embeddings in RAM -----
    encoder = ESM2Encoder(args.model)
    t0 = time.time()
    encoded = encoder.encode_batch(sampled["sequence"].tolist())
    residue_sets = [e.residue_embeddings for e in encoded]
    print(f"Encoded in {time.time() - t0:.1f}s on {encoder.device}")

    # --- Stratified within-train holdout for early stopping ------------------
    holdout_idx: list[int] = []
    for cls in range(len(classes)):
        members = np.flatnonzero(labels_all == cls)
        k = max(1, int(round(len(members) * args.holdout_frac)))
        holdout_idx.extend(rng.choice(members, size=k, replace=False).tolist())
    holdout_mask = np.zeros(len(sampled), dtype=bool)
    holdout_mask[holdout_idx] = True
    train_idx = np.flatnonzero(~holdout_mask)
    val_idx = np.flatnonzero(holdout_mask)

    device = resolve_device()
    pooler = AttentionPooling(encoder.hidden_size).to(device)
    head = nn.Linear(encoder.hidden_size, len(classes)).to(device)
    params = list(pooler.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    def run_epoch(indices: np.ndarray, train_mode: bool) -> tuple[float, float]:
        pooler.train(train_mode)
        head.train(train_mode)
        order = rng.permutation(indices) if train_mode else indices
        total_loss, correct = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            padded, mask = pad_batch([residue_sets[i] for i in batch_idx])
            padded, mask = padded.to(device), mask.to(device)
            target = torch.as_tensor(labels_all[batch_idx], device=device)
            with torch.set_grad_enabled(train_mode):
                pooled, _ = pooler(padded, mask)
                logits = head(pooled)
                loss = loss_fn(logits, target)
                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total_loss += float(loss) * len(batch_idx)
            correct += int((logits.argmax(-1) == target).sum())
        return total_loss / len(order), correct / len(order)

    best_val, best_state, patience_left = -1.0, None, args.patience
    history = []
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(train_idx, True)
        _, val_acc = run_epoch(val_idx, False)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4)})
        print(f"epoch {epoch:02d}  loss {train_loss:.4f}  train_acc {train_acc:.3f}  val_acc {val_acc:.3f}")
        if val_acc > best_val:
            best_val, patience_left = val_acc, args.patience
            best_state = {k: v.detach().cpu().clone() for k, v in pooler.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping.")
                break

    assert best_state is not None
    pooler.load_state_dict(best_state)
    pooler.cpu().save(args.out)
    print(f"Saved attention pooler → {args.out} (best holdout acc {best_val:.3f})")

    log_experiment(
        "attention_pooler",
        config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        metrics={"best_holdout_acc": best_val, "n_train": int(len(train_idx)),
                 "n_holdout": int(len(val_idx)), "n_classes": len(classes),
                 "history": history},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
