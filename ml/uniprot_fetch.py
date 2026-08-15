"""Fetch a single UniProt entry on demand.

Lets the live API answer for accessions outside the corpus: the entry is
downloaded from UniProt's REST API, parsed with the exact same field
derivation as the corpus pipeline, and embedded on the fly. Nothing is
persisted to the corpus — the embedding lands in the ad-hoc cache only.
"""

from __future__ import annotations

import io
import re

import pandas as pd

ACCESSION_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{5,9}$")

ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}.tsv"

FIELDS = (
    "accession,id,protein_name,gene_primary,organism_name,organism_id,"
    "length,sequence,xref_pfam,ec,keyword,cc_subcellular_location,protein_families"
)


class UniProtFetchError(Exception):
    """Fetch failed in a way the caller should surface (bad id, not found, down)."""


def validate_accession(accession: str) -> str:
    cleaned = accession.strip().upper()
    if not ACCESSION_PATTERN.match(cleaned):
        raise UniProtFetchError(
            f"'{accession}' does not look like a UniProt accession "
            "(expected e.g. P69905 or A0A023PXB0)."
        )
    return cleaned


def parse_entry_tsv(text: str) -> dict:
    """One derived-field record from a single-entry UniProt TSV."""
    from ml.corpus import RAW_COLUMNS, derive_fields

    frame = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    if frame.empty:
        raise UniProtFetchError("UniProt returned an empty record.")
    missing = set(RAW_COLUMNS) - set(frame.columns)
    if missing:
        raise UniProtFetchError(f"UniProt response missing columns: {sorted(missing)}")
    frame = frame.rename(columns=RAW_COLUMNS)
    frame["sequence"] = frame["sequence"].str.strip().str.upper()
    row = derive_fields(frame).iloc[0]
    return {
        "accession": row["accession"],
        "name": row["protein_name"],
        "gene": row["gene"] if isinstance(row["gene"], str) else None,
        "organism": row["organism_short"],
        "length": int(row["length"]),
        "family": row["family"] if isinstance(row["family"], str) else None,
        "pfam": row["pfam_primary"] if isinstance(row["pfam_primary"], str) else None,
        "ec_class": row["ec_class"] if isinstance(row["ec_class"], str) else None,
        "localization": row["localization"] if isinstance(row["localization"], str) else None,
        "sequence": row["sequence"],
    }


def fetch_entry(accession: str, timeout: int = 20) -> dict:
    """Download and parse one entry; raises UniProtFetchError on any failure."""
    import requests

    cleaned = validate_accession(accession)
    try:
        response = requests.get(
            ENTRY_URL.format(accession=cleaned),
            params={"fields": FIELDS},
            timeout=timeout,
            headers={"User-Agent": "ProteinLens/0.2 (live fetch)"},
        )
    except requests.RequestException as exc:
        raise UniProtFetchError(f"UniProt is unreachable: {exc}") from exc
    if response.status_code == 404 or not response.text.strip():
        raise UniProtFetchError(f"UniProt has no entry '{cleaned}'.")
    if response.status_code != 200:
        raise UniProtFetchError(
            f"UniProt returned HTTP {response.status_code} for '{cleaned}'."
        )
    return parse_entry_tsv(response.text)
