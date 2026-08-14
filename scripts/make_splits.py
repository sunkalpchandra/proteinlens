"""Create leakage-aware train/val/test splits for the corpus.

Usage:
    python scripts/make_splits.py [--seed 42] [--ratios 0.70 0.15 0.15]

Writes ``data/processed/splits.json`` with per-accession assignments, the
grouping statistics, and a k-mer leakage audit comparing cross-split pair
similarity against a within-train reference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.splitting import audit_leakage, make_splits, save_splits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/proteins.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=["annotation", "mmseqs"], default="annotation",
                        help="grouping: annotation union-find (default) or MMseqs2 "
                             "30%%-identity clusters (requires the mmseqs binary)")
    args = parser.parse_args()

    df = pd.read_parquet(args.corpus)
    splits, summary = make_splits(df, tuple(args.ratios), args.seed, method=args.method)
    print("Split sizes:", summary["split_sizes"])
    print("Grouping:", summary["group_stats"])

    audit = audit_leakage(df, splits, seed=args.seed)
    summary["leakage_audit"] = audit
    print("Leakage audit (4-mer cosine): cross-split p99 ="
          f" {audit['cross_split']['p99']:.3f} vs train-reference p99 ="
          f" {audit['train_reference']['p99']:.3f}")

    save_splits(splits, df, summary, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
