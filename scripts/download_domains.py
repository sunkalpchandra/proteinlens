"""Download UniProt DOMAIN features (coordinates) for the corpus organisms.

Fetches the ``ft_domain`` field for the same nine Swiss-Prot organism subsets
as the main download, parses feature strings like

    DOMAIN 37..64; /note="EF-hand 1"; /evidence=…; DOMAIN 73..108; /note="EF-hand 2"

into per-protein (name, start, end) records, joins against the corpus, and
writes ``data/processed/domains.parquet``. Coverage is partial by nature —
UniProt curates DOMAIN features for a minority of entries; Pfam-only domain
locations would need InterPro and are out of scope.

Usage:
    python scripts/download_domains.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.download_data import ORGANISMS, STREAM_URL, make_session  # noqa: E402

FEATURE_PATTERN = re.compile(
    r'DOMAIN\s+(?P<start>\d+)\.\.(?P<end>\d+);(?:\s*/note="(?P<note>[^"]*)")?'
)


def parse_domains(text: str | float) -> list[tuple[str, int, int]]:
    if not isinstance(text, str):
        return []
    out = []
    for match in FEATURE_PATTERN.finditer(text):
        name = (match.group("note") or "domain").strip()
        out.append((name, int(match.group("start")), int(match.group("end"))))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/proteins.parquet"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/domains.parquet"))
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    session = make_session()
    frames = []
    print(f"Downloading DOMAIN features for {len(ORGANISMS)} organisms")
    for name, taxon_id in ORGANISMS.items():
        params = {
            "query": f"(reviewed:true) AND (organism_id:{taxon_id}) "
                     f"AND (length:[{args.min_length} TO {args.max_length}])",
            "format": "tsv",
            "fields": "accession,ft_domain",
            "compressed": "true",
        }
        out_path = args.raw / f"domains_{name}.tsv.gz"
        with session.get(STREAM_URL, params=params, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
        frame = pd.read_csv(out_path, sep="\t", compression="gzip", dtype=str)
        frames.append(frame)
        n_with = frame["Domain [FT]"].notna().sum()
        print(f"  {name:<12} {len(frame):>6} rows, {n_with:>5} with DOMAIN features")

    merged = pd.concat(frames, ignore_index=True).rename(
        columns={"Entry": "accession", "Domain [FT]": "ft_domain"}
    )
    corpus_accessions = set(pd.read_parquet(args.corpus, columns=["accession"])["accession"])
    merged = merged[merged["accession"].isin(corpus_accessions)]

    records = []
    for row in merged.itertuples():
        for name, start, end in parse_domains(row.ft_domain):
            records.append({"accession": row.accession, "name": name,
                            "start": start, "end": end})
    domains = pd.DataFrame(records, columns=["accession", "name", "start", "end"])
    domains = domains.sort_values(["accession", "start"]).reset_index(drop=True)
    domains.to_parquet(args.out, index=False)

    stats = {
        "created_at": datetime.now(UTC).isoformat(),
        "n_domains": len(domains),
        "n_proteins_with_domains": int(domains["accession"].nunique()),
        "corpus_coverage": round(domains["accession"].nunique() / len(corpus_accessions), 4),
        "top_domain_names": domains["name"].value_counts().head(15).to_dict(),
    }
    Path(str(args.out).replace(".parquet", "_manifest.json")).write_text(
        json.dumps(stats, indent=2)
    )
    print(f"{len(domains)} domain records on {stats['n_proteins_with_domains']} proteins "
          f"({stats['corpus_coverage']:.1%} of corpus) → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
