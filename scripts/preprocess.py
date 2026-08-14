"""Preprocess raw UniProt TSVs into the ProteinLens corpus.

Steps (all deterministic given ``--seed``):
  1. Load every ``swissprot_*.tsv.gz`` in ``data/raw/``.
  2. Validate sequences against the canonical 20-letter alphabet; drop the rest.
  3. Drop exact duplicate sequences (cross-organism duplicates keep first accession).
  4. Drop "(putative) uncharacterized" entries (no functional signal).
  5. Derive analysis fields (family, Pfam, EC class, localization — see ml/corpus.py).
  6. Cap proteins per family, then sample organisms proportionally down to
     ``--target``. Curated showcase accessions (ml.corpus.SHOWCASE_ACCESSIONS)
     are exempt from both reductions so demos stay legible.
  7. Write ``data/processed/proteins.parquet`` + ``corpus_manifest.json``.

Usage:
    python scripts/preprocess.py [--target 12000] [--family-cap 80] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.corpus import OUTPUT_COLUMNS, SHOWCASE_ACCESSIONS, derive_fields, load_raw  # noqa: E402
from ml.sequence import is_valid_sequence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--target", type=int, default=12000)
    parser.add_argument("--family-cap", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_raw(args.raw)
    n_raw = len(df)
    print(f"Loaded {n_raw} raw entries")

    # --- Validation -------------------------------------------------------
    df["sequence"] = df["sequence"].str.strip().str.upper()
    valid_mask = df["sequence"].map(lambda s: isinstance(s, str) and is_valid_sequence(s, 50, 512))
    n_invalid = int((~valid_mask).sum())
    df = df[valid_mask].copy()

    # --- Deduplication ----------------------------------------------------
    df = df.sort_values("accession", kind="stable")
    n_before = len(df)
    df = df.drop_duplicates(subset="sequence", keep="first")
    n_dupes = n_before - len(df)

    # --- Drop entries with no functional signal -----------------------------
    unchar_mask = df["protein_name"].str.match(r"(?i)^(putative\s+)?uncharacterized", na=False)
    n_uncharacterized = int(unchar_mask.sum())
    df = df[~unchar_mask].copy()

    # --- Derived fields ----------------------------------------------------
    df = derive_fields(df)
    keep_always = df["accession"].isin(SHOWCASE_ACCESSIONS)

    # --- Family cap (showcase-exempt) ----------------------------------------
    def cap_family(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) <= args.family_cap:
            return group
        protected = group[group["accession"].isin(SHOWCASE_ACCESSIONS)]
        pool = group[~group["accession"].isin(SHOWCASE_ACCESSIONS)]
        n_pick = max(0, args.family_cap - len(protected))
        idx = rng.choice(len(pool), size=min(n_pick, len(pool)), replace=False)
        return pd.concat([protected, pool.iloc[np.sort(idx)]])

    with_family = df[df["family"].notna()]
    without_family = df[df["family"].isna()]
    capped = (
        with_family.groupby("family", group_keys=False, sort=False)[df.columns]
        .apply(cap_family)
    )
    df = pd.concat([capped, without_family], ignore_index=True)
    keep_always = df["accession"].isin(SHOWCASE_ACCESSIONS)
    print(f"After family cap ({args.family_cap}): {len(df)}")

    # --- Proportional organism subsample (showcase-exempt) --------------------
    if len(df) > args.target:
        counts = df["organism"].value_counts()
        frac = args.target / len(df)
        keep_indices = [df.index[keep_always].to_numpy()]
        sampleable = df[~keep_always]
        for organism, n in counts.items():
            pool = sampleable.index[sampleable["organism"] == organism].to_numpy()
            k = max(1, int(round(n * frac)))
            keep_indices.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
        keep = np.concatenate(keep_indices)
        df = df.loc[np.unique(keep)]
    df = df.sort_values("accession", kind="stable").reset_index(drop=True)

    # --- Write ---------------------------------------------------------------
    corpus = df[OUTPUT_COLUMNS]
    corpus_path = args.out / "proteins.parquet"
    corpus.to_parquet(corpus_path, index=False)

    fasta_path = args.out / "corpus.fasta"
    with open(fasta_path, "w") as fh:
        for row in corpus.itertuples():
            fh.write(f">{row.accession} {row.protein_name}\n{row.sequence}\n")

    raw_manifest = json.loads((args.raw / "manifest.json").read_text())
    n_showcase = int(corpus["accession"].isin(SHOWCASE_ACCESSIONS).sum())
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "uniprot_release": raw_manifest["files"][0]["uniprot_release"],
        "seed": args.seed,
        "target": args.target,
        "family_cap": args.family_cap,
        "n_raw": n_raw,
        "n_invalid_alphabet": n_invalid,
        "n_exact_duplicates": n_dupes,
        "n_uncharacterized_dropped": n_uncharacterized,
        "n_final": len(corpus),
        "n_showcase_retained": n_showcase,
        "organisms": corpus["organism"].value_counts().to_dict(),
        "n_families": int(corpus["family"].nunique()),
        "n_with_pfam": int(corpus["pfam_primary"].notna().sum()),
        "n_enzymes": int(corpus["is_enzyme"].sum()),
        "localization_counts": corpus["localization"].value_counts().to_dict(),
        "top_families": corpus["family"].value_counts().head(25).to_dict(),
    }
    (args.out / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Final corpus: {len(corpus)} proteins ({n_showcase} showcase) → {corpus_path}")
    print(f"Families: {manifest['n_families']}, enzymes: {manifest['n_enzymes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
