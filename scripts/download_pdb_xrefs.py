"""Download PDB cross-references for the corpus organisms.

Mirrors the domain download: one stream request per organism for
``accession,xref_pdb``, joined against the corpus and written to
``data/processed/pdb_xrefs.parquet`` (one row per accession with its PDB id
list). Profiles link these to RCSB; no structural data is downloaded.

Usage:
    python scripts/download_pdb_xrefs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.download_data import ORGANISMS, STREAM_URL, make_session  # noqa: E402


def parse_pdb_ids(text: str | float) -> list[str]:
    if not isinstance(text, str):
        return []
    return [p.strip() for p in text.strip().strip(";").split(";") if p.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/proteins.parquet"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/pdb_xrefs.parquet"))
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    session = make_session()
    frames = []
    print(f"Downloading PDB cross-references for {len(ORGANISMS)} organisms")
    for name, taxon_id in ORGANISMS.items():
        params = {
            "query": f"(reviewed:true) AND (organism_id:{taxon_id}) "
                     f"AND (length:[{args.min_length} TO {args.max_length}])",
            "format": "tsv",
            "fields": "accession,xref_pdb",
            "compressed": "true",
        }
        out_path = args.raw / f"pdb_{name}.tsv.gz"
        with session.get(STREAM_URL, params=params, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
        frame = pd.read_csv(out_path, sep="\t", compression="gzip", dtype=str)
        frames.append(frame)
        n_with = frame["PDB"].notna().sum()
        print(f"  {name:<12} {len(frame):>6} rows, {n_with:>5} with PDB entries")

    merged = pd.concat(frames, ignore_index=True).rename(
        columns={"Entry": "accession", "PDB": "pdb"}
    )
    corpus_accessions = set(pd.read_parquet(args.corpus, columns=["accession"])["accession"])
    merged = merged[merged["accession"].isin(corpus_accessions)]
    merged["pdb_ids"] = merged["pdb"].map(parse_pdb_ids)
    merged = merged[merged["pdb_ids"].map(len) > 0][["accession", "pdb_ids"]]
    merged = merged.sort_values("accession").reset_index(drop=True)
    merged.to_parquet(args.out, index=False)

    stats = {
        "created_at": datetime.now(UTC).isoformat(),
        "n_proteins_with_pdb": len(merged),
        "corpus_coverage": round(len(merged) / len(corpus_accessions), 4),
        "total_structures": int(merged["pdb_ids"].map(len).sum()),
    }
    Path(str(args.out).replace(".parquet", "_manifest.json")).write_text(
        json.dumps(stats, indent=2)
    )
    print(f"{len(merged)} proteins with PDB entries "
          f"({stats['corpus_coverage']:.1%} of corpus, "
          f"{stats['total_structures']} structures) → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
