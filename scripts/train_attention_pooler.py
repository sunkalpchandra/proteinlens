"""Train the attention pooling head on frozen ESM-2 residue embeddings.

The language model stays frozen; only the small additive-attention pooler
learns. Two objectives, selected with ``--objective``:

  ce      (default) pooler + linear classifier trained jointly with
          cross-entropy on family labels; the classifier is discarded.
          Early stopping: holdout classification accuracy.
  supcon  supervised contrastive loss on the pooled embeddings with
          class-balanced batches — optimizes the cosine geometry retrieval
          actually uses. Early stopping: holdout 1-NN family accuracy.

Both train on the *train split only* with a within-train stratified holdout;
the leakage-audited comparison happens in the benchmark suite. Artifacts:
``attention_pooler.pt`` (ce) / ``attention_pooler_supcon.pt`` (supcon).

Usage:
    python scripts/train_attention_pooler.py [--objective supcon] [--max-classes 150]
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

from ml.losses import supcon_loss  # noqa: E402
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


def balanced_batches(
    labels: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
    classes_per_batch: int = 8,
    per_class: int = 4,
    n_batches: int | None = None,
) -> list[np.ndarray]:
    """Class-balanced batches so every SupCon anchor has in-batch positives."""
    by_class: dict[int, np.ndarray] = {}
    for cls in np.unique(labels[indices]):
        members = indices[labels[indices] == cls]
        if len(members) >= 2:
            by_class[int(cls)] = members
    classes = np.array(list(by_class))
    n_batches = n_batches or max(1, len(indices) // (classes_per_batch * per_class))
    batches = []
    for _ in range(n_batches):
        picked = rng.choice(classes, size=min(classes_per_batch, len(classes)), replace=False)
        rows = [rng.choice(by_class[int(c)], size=min(per_class, len(by_class[int(c)])),
                           replace=False) for c in picked]
        batches.append(np.concatenate(rows))
    return batches


@torch.inference_mode()
def knn_accuracy(
    train_z: torch.Tensor, train_y: np.ndarray,
    val_z: torch.Tensor, val_y: np.ndarray,
) -> float:
    """1-NN cosine classification of holdout against train embeddings."""
    a = torch.nn.functional.normalize(val_z, dim=1)
    b = torch.nn.functional.normalize(train_z, dim=1)
    nearest = (a @ b.T).argmax(dim=1).cpu().numpy()
    return float((train_y[nearest] == val_y).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/proteins.parquet"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument("--out", type=Path, default=None,
                        help="default: attention_pooler.pt (ce) / attention_pooler_supcon.pt (supcon)")
    parser.add_argument("--objective", choices=["ce", "supcon"], default="ce")
    parser.add_argument("--temperature", type=float, default=0.1)
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
    if args.out is None:
        name = "attention_pooler.pt" if args.objective == "ce" else "attention_pooler_supcon.pt"
        args.out = Path("data/embeddings") / name

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
    # Memory discipline for 8GB unified hosts: small token budget, chunked
    # encoding with MPS cache flushes between chunks, and fp16 residue storage
    # (the RAM-side cache halves; training upcasts per batch).
    encoder = ESM2Encoder(args.model, token_budget=4096)
    t0 = time.time()
    sequences = sampled["sequence"].tolist()
    residue_sets: list[torch.Tensor] = []
    for start in range(0, len(sequences), 128):
        for enc in encoder.encode_batch(sequences[start : start + 128]):
            residue_sets.append(enc.residue_embeddings.half())
        if encoder.device.type == "mps":
            torch.mps.empty_cache()
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
    params = list(pooler.parameters())
    if args.objective == "ce":
        params += list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss()

    def pool_batch(batch_idx: np.ndarray, grad: bool) -> torch.Tensor:
        padded, mask = pad_batch([residue_sets[i].float() for i in batch_idx])
        padded, mask = padded.to(device), mask.to(device)
        with torch.set_grad_enabled(grad):
            pooled, _ = pooler(padded, mask)
        return pooled

    @torch.inference_mode()
    def pool_all(indices: np.ndarray) -> torch.Tensor:
        pooler.eval()
        chunks = [pool_batch(indices[s : s + 64], grad=False)
                  for s in range(0, len(indices), 64)]
        return torch.cat(chunks)

    def run_epoch_ce(indices: np.ndarray, train_mode: bool) -> tuple[float, float]:
        pooler.train(train_mode)
        head.train(train_mode)
        order = rng.permutation(indices) if train_mode else indices
        total_loss, correct = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            target = torch.as_tensor(labels_all[batch_idx], device=device)
            with torch.set_grad_enabled(train_mode):
                logits = head(pool_batch(batch_idx, grad=train_mode))
                loss = ce_loss(logits, target)
                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total_loss += float(loss) * len(batch_idx)
            correct += int((logits.argmax(-1) == target).sum())
        return total_loss / len(order), correct / len(order)

    def run_epoch_supcon(indices: np.ndarray) -> float:
        pooler.train(True)
        total, seen = 0.0, 0
        for batch_idx in balanced_batches(labels_all, indices, rng):
            target = torch.as_tensor(labels_all[batch_idx], device=device)
            pooled = pool_batch(batch_idx, grad=True)
            loss = supcon_loss(pooled, target, temperature=args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch_idx)
            seen += len(batch_idx)
        return total / max(seen, 1)

    best_val, best_state, patience_left = -1.0, None, args.patience
    history = []
    for epoch in range(args.epochs):
        if args.objective == "ce":
            train_loss, train_acc = run_epoch_ce(train_idx, True)
            _, val_acc = run_epoch_ce(val_idx, False)
            metric_name = "val_acc"
        else:
            train_loss = run_epoch_supcon(train_idx)
            val_acc = knn_accuracy(pool_all(train_idx), labels_all[train_idx],
                                   pool_all(val_idx), labels_all[val_idx])
            metric_name = "val_1nn_acc"
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        metric_name: round(val_acc, 4)})
        print(f"epoch {epoch:02d}  loss {train_loss:.4f}  {metric_name} {val_acc:.3f}")
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
    print(f"Saved {args.objective} attention pooler → {args.out} "
          f"(best holdout metric {best_val:.3f})")

    log_experiment(
        f"attention_pooler_{args.objective}",
        config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        metrics={"best_holdout_metric": best_val, "n_train": int(len(train_idx)),
                 "n_holdout": int(len(val_idx)), "n_classes": len(classes),
                 "history": history},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
