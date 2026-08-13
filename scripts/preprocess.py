"""Preprocess raw UniProt TSVs into the ProteinLens corpus.

Steps (all deterministic given ``--seed``):
  1. Load every ``swissprot_*.tsv.gz`` in ``data/raw/``.
  2. Validate sequences against the canonical 20-letter alphabet; drop the rest.
  3. Drop exact duplicate sequences (cross-organism duplicates keep first accession).
  4. Derive analysis fields: primary Pfam domain, coarse family label, enzyme
     class from EC number, coarse subcellular localization.
  5. Cap the number of proteins per family (prevents huge families dominating
     the map), then sample organisms proportionally down to ``--target`` proteins.
  6. Write ``data/processed/proteins.parquet`` + ``corpus_manifest.json``.

Usage:
    python scripts/preprocess.py [--target 12000] [--family-cap 80] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.sequence import EC_CLASS_NAMES, is_valid_sequence  # noqa: E402

RAW_COLUMNS = {
    "Entry": "accession",
    "Entry Name": "entry_name",
    "Protein names": "protein_name",
    "Gene Names (primary)": "gene",
    "Organism": "organism",
    "Organism (ID)": "taxon_id",
    "Length": "length",
    "Sequence": "sequence",
    "Pfam": "pfam",
    "EC number": "ec",
    "Keywords": "keywords",
    "Subcellular location [CC]": "subcellular_location",
    "Protein families": "protein_families",
}

# Priority-ordered controlled vocabulary for coarse localization labels.
LOCALIZATION_VOCAB = [
    ("Secreted", r"\bsecreted\b"),
    ("Nucleus", r"\bnucleus|nucleolus|nucleoplasm\b"),
    ("Mitochondrion", r"\bmitochondri"),
    ("Chloroplast", r"\bchloroplast|plastid"),
    ("Endoplasmic reticulum", r"\bendoplasmic reticulum\b"),
    ("Golgi", r"\bgolgi\b"),
    ("Lysosome/Vacuole", r"\blysosome|vacuole\b"),
    ("Peroxisome", r"\bperoxisome\b"),
    ("Cell membrane", r"\bcell membrane|plasma membrane\b"),
    ("Membrane", r"\bmembrane\b"),
    ("Periplasm", r"\bperiplasm"),
    ("Cell wall", r"\bcell wall\b"),
    ("Cytoplasm", r"\bcytoplasm|cytosol\b"),
]


def parse_family(text: str | float) -> str | None:
    """Coarse family label: first comma-separated segment of UniProt's
    'Protein families' annotation (e.g. 'Globin family')."""
    if not isinstance(text, str) or not text.strip():
        return None
    first = text.split(";")[0].split(",")[0].strip()
    return first or None


def parse_pfam(text: str | float) -> list[str]:
    if not isinstance(text, str):
        return []
    return [p for p in text.strip().strip(";").split(";") if p]


def parse_ec_class(text: str | float) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    first_digit = text.strip().split(";")[0].strip().split(".")[0]
    return EC_CLASS_NAMES.get(first_digit)


def parse_localization(text: str | float) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    lowered = text.lower()
    for label, pattern in LOCALIZATION_VOCAB:
        if re.search(pattern, lowered):
            return label
    return None


def clean_protein_name(text: str | float) -> str:
    """UniProt protein names carry EC refs and synonyms in parens; keep the head."""
    if not isinstance(text, str):
        return ""
    name = re.split(r" \(", text, maxsplit=1)[0].strip()
    return name


def load_raw(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("swissprot_*.tsv.gz"))
    if not files:
        raise SystemExit(f"No raw files in {raw_dir}. Run scripts/download_data.py first.")
    frames = []
    for path in files:
        df = pd.read_csv(path, sep="\t", compression="gzip", dtype=str)
        df["source_file"] = path.name
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    missing = set(RAW_COLUMNS) - set(merged.columns)
    if missing:
        raise SystemExit(f"Raw TSVs missing expected columns: {missing}")
    return merged.rename(columns=RAW_COLUMNS)


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
    # "(Putative) uncharacterized protein" entries (mostly dubious ORFs) carry
    # no family/function metadata and add unlabeled noise to every evaluation.
    unchar_mask = df["protein_name"].str.match(r"(?i)^(putative\s+)?uncharacterized", na=False)
    n_uncharacterized = int(unchar_mask.sum())
    df = df[~unchar_mask].copy()

    # --- Derived fields ----------------------------------------------------
    df["length"] = df["length"].astype(int)
    df["protein_name_full"] = df["protein_name"]
    df["protein_name"] = df["protein_name"].map(clean_protein_name)
    df["family"] = df["protein_families"].map(parse_family)
    df["pfam_all"] = df["pfam"].map(parse_pfam)
    df["pfam_primary"] = df["pfam_all"].map(lambda xs: xs[0] if xs else None)
    df["ec_class"] = df["ec"].map(parse_ec_class)
    df["is_enzyme"] = df["ec"].notna() & df["ec"].str.strip().astype(bool)
    df["localization"] = df["subcellular_location"].map(parse_localization)

    def short_organism(name: str) -> str:
        words = re.sub(r"\(.*", "", name).strip().split()
        if len(words) >= 2:
            return f"{words[0][0]}. {words[1]}"
        return name

    df["organism_short"] = df["organism"].map(short_organism)

    # --- Family cap ---------------------------------------------------------
    def cap_family(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) <= args.family_cap:
            return group
        idx = rng.choice(len(group), size=args.family_cap, replace=False)
        return group.iloc[np.sort(idx)]

    with_family = df[df["family"].notna()]
    without_family = df[df["family"].isna()]
    capped = (
        with_family.groupby("family", group_keys=False, sort=False)[df.columns]
        .apply(cap_family)
    )
    df = pd.concat([capped, without_family], ignore_index=True)
    print(f"After family cap ({args.family_cap}): {len(df)}")

    # --- Proportional organism subsample ------------------------------------
    if len(df) > args.target:
        counts = df["organism"].value_counts()
        frac = args.target / len(df)
        keep_indices = []
        for organism, n in counts.items():
            pool = df.index[df["organism"] == organism].to_numpy()
            k = max(1, int(round(n * frac)))
            keep_indices.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
        keep = np.concatenate(keep_indices)
        df = df.loc[keep]
    df = df.sort_values("accession", kind="stable").reset_index(drop=True)

    # --- Write ---------------------------------------------------------------
    out_cols = [
        "accession", "entry_name", "protein_name", "protein_name_full", "gene",
        "organism", "organism_short", "taxon_id", "length", "sequence", "family",
        "pfam_all", "pfam_primary", "ec", "ec_class", "is_enzyme",
        "keywords", "localization", "subcellular_location",
    ]
    corpus = df[out_cols]
    corpus_path = args.out / "proteins.parquet"
    corpus.to_parquet(corpus_path, index=False)

    fasta_path = args.out / "corpus.fasta"
    with open(fasta_path, "w") as fh:
        for row in corpus.itertuples():
            fh.write(f">{row.accession} {row.protein_name}\n{row.sequence}\n")

    raw_manifest = json.loads((args.raw / "manifest.json").read_text())
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "uniprot_release": raw_manifest["files"][0]["uniprot_release"],
        "seed": args.seed,
        "target": args.target,
        "family_cap": args.family_cap,
        "n_raw": n_raw,
        "n_invalid_alphabet": n_invalid,
        "n_exact_duplicates": n_dupes,
        "n_uncharacterized_dropped": n_uncharacterized,
        "n_final": len(corpus),
        "organisms": corpus["organism"].value_counts().to_dict(),
        "n_families": int(corpus["family"].nunique()),
        "n_with_pfam": int(corpus["pfam_primary"].notna().sum()),
        "n_enzymes": int(corpus["is_enzyme"].sum()),
        "localization_counts": corpus["localization"].value_counts().to_dict(),
        "top_families": corpus["family"].value_counts().head(25).to_dict(),
    }
    (args.out / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Final corpus: {len(corpus)} proteins → {corpus_path}")
    print(f"Families: {manifest['n_families']}, enzymes: {manifest['n_enzymes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
