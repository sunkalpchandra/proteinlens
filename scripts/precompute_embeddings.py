"""Precompute protein-level embeddings for the whole corpus.

One pass through the frozen ESM-2 encoder; every pooling strategy is computed
from the same residue representations, so the four matrices are directly
comparable. Output layout (see ``ml.embeddings.EmbeddingStore``):

    data/embeddings/corpus_{mean,max,bos,attention}.npy   float32 [N, D]
    data/embeddings/accessions.json
    data/embeddings/store_meta.json

Attention pooling requires the trained pooler from
``scripts/train_attention_pooler.py``; without it the script warns and writes
the three parameter-free poolings only.

Usage:
    python scripts/precompute_embeddings.py [--chunk 384]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.embeddings import EMBEDDING_VERSION  # noqa: E402
from ml.tracking import log_experiment  # noqa: E402
from models.encoder import ESM2Encoder  # noqa: E402
from models.pooling import AttentionPooling, Pooler  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/proteins.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--pooler", type=Path, default=Path("data/embeddings/attention_pooler.pt"))
    parser.add_argument("--chunk", type=int, default=384, help="proteins per progress chunk")
    args = parser.parse_args()

    df = pd.read_parquet(args.corpus)
    sequences = df["sequence"].tolist()
    accessions = df["accession"].tolist()
    n = len(df)

    encoder = ESM2Encoder(args.model)
    if args.pooler.exists():
        pooler = Pooler(AttentionPooling.load(args.pooler))
    else:
        print(f"WARNING: {args.pooler} not found — skipping attention pooling.")
        pooler = Pooler(None)
    poolings = pooler.available()
    print(f"Embedding {n} proteins on {encoder.device} | poolings: {poolings}")

    matrices = {p: np.zeros((n, encoder.hidden_size), dtype=np.float32) for p in poolings}
    t0 = time.time()
    done = 0
    for start in range(0, n, args.chunk):
        batch = sequences[start : start + args.chunk]
        encoded = encoder.encode_batch(batch)
        with torch.inference_mode():
            for offset, enc in enumerate(encoded):
                for pooling in poolings:
                    vec, _ = pooler.pool(enc.residue_embeddings, enc.bos_embedding, pooling)
                    matrices[pooling][start + offset] = vec.numpy()
        done += len(batch)
        rate = done / (time.time() - t0)
        eta = (n - done) / max(rate, 1e-9)
        print(f"  {done:>6}/{n}  {rate:6.1f} proteins/s  ETA {eta/60:5.1f} min", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    for pooling, matrix in matrices.items():
        np.save(args.out / f"corpus_{pooling}.npy", matrix)
    (args.out / "accessions.json").write_text(json.dumps(accessions))

    corpus_hash = hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest()[:16]
    meta = {
        "model": args.model,
        "embedding_version": EMBEDDING_VERSION,
        "dim": encoder.hidden_size,
        "poolings": poolings,
        "n_proteins": n,
        "corpus_file": str(args.corpus),
        "corpus_sha256_16": corpus_hash,
        "device": str(encoder.device),
        "created_at": datetime.now(UTC).isoformat(),
        "wall_time_s": round(time.time() - t0, 1),
    }
    (args.out / "store_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done in {meta['wall_time_s']}s → {args.out}")

    log_experiment(
        "precompute_embeddings",
        config={"model": args.model, "corpus_sha256_16": corpus_hash, "n": n},
        metrics={"wall_time_s": meta["wall_time_s"], "poolings": poolings},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
