"""Download a deep-mutational-scanning score set from MaveDB.

Defaults to the measured (not imputed) human Calmodulin DMS-TileSeq set
(urn:mavedb:00000001-c-1, Weile et al. 2017 complementation assay) — CALM1 /
P0DP23 is a corpus showcase protein. Single-residue substitutions are parsed
from three-letter HGVS (``p.Gly34Arg``), validated against the corpus
sequence (with automatic Met-offset detection), and written to
``data/processed/dms_{accession}.parquet`` plus a provenance manifest.

MaveDB data is CC BY-NC-SA 4.0 unless the score set states otherwise; this
project uses it for a non-commercial validation analysis and records the URN.

Usage:
    python scripts/download_dms.py [--urn urn:mavedb:00000001-c-1] [--accession P0DP23]
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

API = "https://api.mavedb.org/api/v1"

THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
HGVS_SUB = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$")


def parse_substitution(hgvs: str) -> tuple[str, int, str] | None:
    """('G', 34, 'R') from 'p.Gly34Arg'; None for anything but a clean
    single-residue substitution (synonymous, Ter, indels, multis)."""
    match = HGVS_SUB.match(hgvs.strip())
    if not match:
        return None
    wt3, pos, mut3 = match.groups()
    wt, mut = THREE_TO_ONE.get(wt3), THREE_TO_ONE.get(mut3)
    if wt is None or mut is None or wt == mut:
        return None
    return wt, int(pos), mut


def detect_offset(frame: pd.DataFrame, sequence: str) -> int | None:
    """DMS position numbering may or may not count the initiator Met; pick the
    offset (0 or +1) that makes wild-type letters agree with the sequence."""
    for offset in (0, 1):
        checked = matches = 0
        for row in frame.itertuples():
            pos = row.position + offset
            if 1 <= pos <= len(sequence):
                checked += 1
                matches += sequence[pos - 1] == row.wt
        if checked and matches / checked > 0.98:
            return offset
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urn", default="urn:mavedb:00000001-c-1")
    parser.add_argument("--accession", default="P0DP23")
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/proteins.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    import requests

    corpus = pd.read_parquet(args.corpus).set_index("accession")
    if args.accession not in corpus.index:
        raise SystemExit(f"{args.accession} not in corpus.")
    sequence = corpus.loc[args.accession, "sequence"]

    meta = requests.get(f"{API}/score-sets/{args.urn}", timeout=60).json()
    print(f"Score set: {meta.get('title')} ({args.urn})")

    response = requests.get(f"{API}/score-sets/{args.urn}/scores", timeout=120)
    response.raise_for_status()
    raw = pd.read_csv(io.StringIO(response.text))
    n_raw = len(raw)

    records = []
    for row in raw.itertuples():
        parsed = parse_substitution(str(row.hgvs_pro))
        if parsed is None or pd.isna(row.score):
            continue
        wt, pos, mut = parsed
        records.append({"wt": wt, "position": pos, "mut": mut,
                        "score": float(row.score)})
    frame = pd.DataFrame(records)
    print(f"Parsed {len(frame)} single-substitution scores from {n_raw} rows")

    offset = detect_offset(frame, sequence)
    if offset is None:
        raise SystemExit(
            "Wild-type letters disagree with the corpus sequence under offsets "
            "0/+1 — wrong target or numbering scheme; refusing to write."
        )
    frame["position"] = frame["position"] + offset
    in_range = frame["position"].between(1, len(sequence))
    frame = frame[in_range]
    agree = (frame.apply(lambda r: sequence[r.position - 1] == r.wt, axis=1)).mean()
    frame = frame[frame.apply(lambda r: sequence[r.position - 1] == r.wt, axis=1)]
    # Duplicate variants (assay replicates/tiles): keep the mean score.
    frame = frame.groupby(["wt", "position", "mut"], as_index=False)["score"].mean()

    out = args.out_dir / f"dms_{args.accession}.parquet"
    frame.to_parquet(out, index=False)
    manifest = {
        "urn": args.urn,
        "title": meta.get("title"),
        "accession": args.accession,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "n_raw_rows": n_raw,
        "n_variants": len(frame),
        "position_offset_applied": offset,
        "wt_agreement": round(float(agree), 4),
        "positions_covered": int(frame["position"].nunique()),
        "license_note": "MaveDB score sets: CC BY-NC-SA 4.0 unless stated otherwise",
    }
    (args.out_dir / f"dms_{args.accession}_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"{len(frame)} variants across {manifest['positions_covered']} positions "
          f"(offset {offset:+d}, wt agreement {agree:.1%}) → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
