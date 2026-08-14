"""Define the shared evaluation subset for checkpoint/baseline studies.

Embedding the full corpus with every candidate representation (larger ESM-2
checkpoints, ProstT5, contrastive poolers) is not affordable on a laptop, so
comparative studies run on one fixed, stratified subset that every candidate
embeds identically. Selection is deterministic:

  * stratified across organisms proportionally,
  * preferring proteins that carry probe labels (enzyme/EC/localization) so
    subset probes keep statistical power,
  * keeping existing train/val/test assignments (probes reuse them),
  * always including the showcase proteins.

Writes ``data/processed/eval_subset.json`` (accessions + composition stats).

Usage:
    python scripts/make_eval_subset.py [--size 3000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.corpus import SHOWCASE_ACCESSIONS  # noqa: E402
from ml.splitting import load_splits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/proteins.parquet"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/eval_subset.json"))
    parser.add_argument("--size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(args.corpus)
    split_map, _ = load_splits(args.splits)
    df["split"] = df["accession"].map(split_map)

    # Label-bearing proteins keep probe power; weight them 3× in sampling.
    labeled = df["is_enzyme"] | df["ec_class"].notna() | df["localization"].notna()
    weight = np.where(labeled, 3.0, 1.0)

    chosen: set[str] = {a for a in SHOWCASE_ACCESSIONS if a in set(df["accession"])}
    frac = (args.size - len(chosen)) / len(df)
    for _, group in df[~df["accession"].isin(chosen)].groupby("organism_short"):
        k = max(1, int(round(len(group) * frac)))
        w = weight[group.index]
        picks = rng.choice(
            group["accession"].to_numpy(), size=min(k, len(group)),
            replace=False, p=w / w.sum(),
        )
        chosen.update(picks.tolist())

    subset = df[df["accession"].isin(chosen)].sort_values("accession")
    payload = {
        "seed": args.seed,
        "size": len(subset),
        "accessions": subset["accession"].tolist(),
        "splits": subset["split"].value_counts(dropna=False).to_dict(),
        "organisms": subset["organism_short"].value_counts().to_dict(),
        "n_enzyme_labeled": int(subset["is_enzyme"].sum()),
        "n_localization_labeled": int(subset["localization"].notna().sum()),
        "n_showcase": int(subset["accession"].isin(SHOWCASE_ACCESSIONS).sum()),
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Eval subset: {len(subset)} proteins → {args.out}")
    print(f"  splits: {payload['splits']}")
    print(f"  labeled: enzyme {payload['n_enzyme_labeled']}, "
          f"localization {payload['n_localization_labeled']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
