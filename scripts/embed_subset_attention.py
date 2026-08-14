"""Embed the evaluation subset with a specific trained attention pooler.

Used to compare pooling *objectives* (CE vs SupCon) on identical residue
representations: encodes the subset once with the serving ESM-2 checkpoint and
applies the given pooler head.

Usage:
    python scripts/embed_subset_attention.py \
        --pooler data/embeddings/attention_pooler_supcon.pt --name attention_supcon
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.encoder import ESM2Encoder  # noqa: E402
from models.pooling import AttentionPooling, Pooler  # noqa: E402
from models.registry import DEFAULT_MODEL, study_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooler", type=Path, required=True)
    parser.add_argument("--name", required=True,
                        help="artifact suffix, e.g. attention_supcon")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--subset", type=Path, default=Path("data/processed/eval_subset.json"))
    parser.add_argument("--chunk", type=int, default=256)
    args = parser.parse_args()

    subset = json.loads(args.subset.read_text())["accessions"]
    df = pd.read_parquet("data/processed/proteins.parquet").set_index("accession")
    sequences = df.loc[subset, "sequence"].tolist()

    encoder = ESM2Encoder(args.model)
    pooler = Pooler(AttentionPooling.load(args.pooler))
    matrix = np.zeros((len(subset), encoder.hidden_size), dtype=np.float32)

    print(f"Pooling {len(subset)} proteins with {args.pooler.name} on {encoder.device}")
    t0 = time.time()
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    for start in range(0, len(order), args.chunk):
        idx = order[start : start + args.chunk]
        encoded = encoder.encode_batch([sequences[i] for i in idx])
        for row, enc in zip(idx, encoded, strict=True):
            vec, _ = pooler.pool(enc.residue_embeddings, enc.bos_embedding, "attention")
            matrix[row] = vec.numpy()
        print(f"  {min(start + args.chunk, len(order))}/{len(order)}", flush=True)

    out = study_dir(args.model)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"subset_{args.name}.npy", matrix)
    print(f"Done in {(time.time()-t0)/60:.1f} min → {out / f'subset_{args.name}.npy'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
