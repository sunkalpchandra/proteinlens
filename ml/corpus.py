"""Shared corpus parsing: raw UniProt TSV → derived analysis fields.

Used by ``scripts/preprocess.py`` (full corpus build) and
``scripts/add_proteins.py`` (incremental append), so both derive identical
fields from identical rules.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ml.sequence import EC_CLASS_NAMES

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

# Curated, widely recognizable proteins (all within the corpus filters) that
# make demos legible. Preprocessing always retains them; the demo showcase and
# figure scripts reference them.
SHOWCASE_ACCESSIONS = [
    "P69905",  # Hemoglobin subunit alpha (human)
    "P68871",  # Hemoglobin subunit beta (human) — sickle site β6 (seq pos 7)
    "P02144",  # Myoglobin (human)
    "P01308",  # Insulin (human)
    "P61626",  # Lysozyme C (human)
    "P0DP23",  # Calmodulin-1 (human)
    "P04637",  # Cellular tumor antigen p53 (human)
    "P01112",  # GTPase HRas (human)
    "P00441",  # Superoxide dismutase [Cu-Zn] (human)
    "P68431",  # Histone H3.1 (human)
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
    return re.split(r" \(", text, maxsplit=1)[0].strip()


def short_organism(name: str) -> str:
    words = re.sub(r"\(.*", "", name).strip().split()
    if len(words) >= 2:
        return f"{words[0][0]}. {words[1]}"
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


def derive_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add every derived analysis column, in place, and return the frame."""
    df["length"] = df["length"].astype(int)
    df["protein_name_full"] = df["protein_name"]
    df["protein_name"] = df["protein_name"].map(clean_protein_name)
    df["family"] = df["protein_families"].map(parse_family)
    df["pfam_all"] = df["pfam"].map(parse_pfam)
    df["pfam_primary"] = df["pfam_all"].map(lambda xs: xs[0] if xs else None)
    df["ec_class"] = df["ec"].map(parse_ec_class)
    df["is_enzyme"] = df["ec"].notna() & df["ec"].str.strip().astype(bool)
    df["localization"] = df["subcellular_location"].map(parse_localization)
    df["organism_short"] = df["organism"].map(short_organism)
    return df


OUTPUT_COLUMNS = [
    "accession", "entry_name", "protein_name", "protein_name_full", "gene",
    "organism", "organism_short", "taxon_id", "length", "sequence", "family",
    "pfam_all", "pfam_primary", "ec", "ec_class", "is_enzyme",
    "keywords", "localization", "subcellular_location",
]
