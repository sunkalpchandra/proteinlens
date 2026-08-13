"""Download the ProteinLens corpus from UniProtKB (Swiss-Prot, reviewed entries).

Pulls real, reviewed protein records with rich metadata (Pfam domains, EC numbers,
subcellular localization, protein family text) for a panel of model organisms via
the UniProt REST streaming API. One gzipped TSV per organism lands in ``data/raw/``,
together with a manifest recording the exact queries, UniProt release, and file
hashes so the download is reproducible.

Usage:
    python scripts/download_data.py [--min-length 50] [--max-length 512] [--out data/raw]

Swiss-Prot is distributed under CC BY 4.0 (https://www.uniprot.org/help/license).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter, Retry

STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"

# Diverse panel across the tree of life: mammals, fish, insect, nematode, plant,
# fungus, gram+/- bacteria. Taxon IDs are UniProt proteome-level organism IDs.
ORGANISMS: dict[str, int] = {
    "human": 9606,
    "mouse": 10090,
    "zebrafish": 7955,
    "drosophila": 7227,
    "c_elegans": 6239,
    "arabidopsis": 3702,
    "yeast": 559292,        # S. cerevisiae S288C
    "e_coli": 83333,        # E. coli K-12
    "b_subtilis": 224308,   # B. subtilis 168
}

# Return fields (UniProtKB REST field names).
FIELDS = [
    "accession",
    "id",
    "protein_name",
    "gene_primary",
    "organism_name",
    "organism_id",
    "length",
    "sequence",
    "xref_pfam",
    "ec",
    "keyword",
    "cc_subcellular_location",
    "protein_families",
]


def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers["User-Agent"] = "ProteinLens/0.1 (data pipeline)"
    return session


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_organism(
    session: requests.Session,
    name: str,
    taxon_id: int,
    out_dir: Path,
    min_length: int,
    max_length: int,
) -> dict:
    query = f"(reviewed:true) AND (organism_id:{taxon_id}) AND (length:[{min_length} TO {max_length}])"
    params = {
        "query": query,
        "format": "tsv",
        "fields": ",".join(FIELDS),
        "compressed": "true",
    }
    out_path = out_dir / f"swissprot_{name}.tsv.gz"

    started = time.time()
    with session.get(STREAM_URL, params=params, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        release = resp.headers.get("x-uniprot-release", "unknown")
        release_date = resp.headers.get("x-uniprot-release-date", "unknown")
        # iter_content undoes HTTP transport encoding but leaves the payload
        # (the actual .gz file requested via compressed=true) intact.
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)

    # Row count (minus header) without holding the file in memory.
    import gzip

    with gzip.open(out_path, "rt") as fh:
        n_rows = sum(1 for _ in fh) - 1

    elapsed = time.time() - started
    print(f"  {name:<12} taxon={taxon_id:<7} rows={n_rows:>6}  {out_path.stat().st_size / 1e6:6.1f} MB  {elapsed:5.1f}s")
    return {
        "organism": name,
        "taxon_id": taxon_id,
        "query": query,
        "file": out_path.name,
        "rows": n_rows,
        "sha256": sha256_file(out_path),
        "uniprot_release": release,
        "uniprot_release_date": release_date,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    session = make_session()

    print(f"Downloading Swiss-Prot subsets ({args.min_length}-{args.max_length} aa) for {len(ORGANISMS)} organisms")
    entries = []
    for name, taxon_id in ORGANISMS.items():
        entries.append(
            download_organism(session, name, taxon_id, args.out, args.min_length, args.max_length)
        )

    manifest = {
        "source": "UniProtKB Swiss-Prot (reviewed:true) via REST stream API",
        "license": "CC BY 4.0 — https://www.uniprot.org/help/license",
        "endpoint": STREAM_URL,
        "fields": FIELDS,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": entries,
        "total_rows": sum(e["rows"] for e in entries),
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nTotal rows: {manifest['total_rows']}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
