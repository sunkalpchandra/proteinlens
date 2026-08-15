"""Embed the evaluation subset with the ProstT5 encoder (structure-aware baseline).

Runs fp16 on GPU/MPS. On an 8GB host this is the largest model ProteinLens
will run — keep it alone on the accelerator (don't run concurrently with ESM
jobs) and expect ~30–60 min for 3k proteins.

Usage:
    python scripts/embed_subset_prost.py [--chunk 8] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.tracking import log_experiment  # noqa: E402
from models.registry import study_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=Path("data/processed/eval_subset.json"))
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/proteins.parquet"))
    parser.add_argument("--chunk", type=int, default=8,
                        help="sequences per forward pass (memory-bound)")
    parser.add_argument("--limit", type=int, default=None,
                        help="embed only the first N subset proteins (reduced-scale fallback)")
    args = parser.parse_args()

    subset = json.loads(args.subset.read_text())["accessions"]
    if args.limit:
        subset = subset[: args.limit]
    df = pd.read_parquet(args.corpus).set_index("accession")
    sequences = df.loc[subset, "sequence"].tolist()

    from models.prost_encoder import PROST_MODEL, ProstT5Encoder

    encoder = ProstT5Encoder()
    print(f"Embedding {len(subset)} proteins with {PROST_MODEL} "
          f"({encoder.hidden_size}-d, fp16={encoder.half}) on {encoder.device}")

    matrix = np.zeros((len(subset), encoder.hidden_size), dtype=np.float32)
    t0 = time.time()
    # Length-sorted chunks reduce padding waste; results unsorted afterwards.
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    for start in range(0, len(order), args.chunk):
        idx = order[start : start + args.chunk]
        matrix[idx] = encoder.embed_mean([sequences[i] for i in idx])
        done = start + len(idx)
        rate = done / (time.time() - t0)
        print(f"  {done:>5}/{len(order)}  {rate:5.2f}/s  "
              f"ETA {(len(order)-done)/max(rate,1e-9)/60:5.1f} min", flush=True)

    out = study_dir(PROST_MODEL)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "subset_mean.npy", matrix)
    (out / "meta.json").write_text(json.dumps({
        "model": PROST_MODEL, "slug": "prostt5", "dim": encoder.hidden_size,
        "n": len(subset), "poolings": ["mean"], "fp16": encoder.half,
        "tokenizer": encoder.tokenizer_provenance,
        "subset_file": str(args.subset), "limit": args.limit,
        "wall_time_s": round(time.time() - t0, 1),
        "device": str(encoder.device),
        "created_at": datetime.now(UTC).isoformat(),
    }, indent=2))
    print(f"Done in {(time.time()-t0)/60:.1f} min → {out}")

    log_experiment("embed_subset_prost",
                   config={"n": len(subset), "chunk": args.chunk, "limit": args.limit},
                   metrics={"wall_time_s": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
