"""Append proteins to an existing corpus + embedding store without a full rebuild.

Pulls the requested accessions from the raw Swiss-Prot TSVs, derives the same
fields as preprocessing, embeds them with the frozen encoder (all available
poolings, including the trained attention pooler), and appends rows to:

    data/processed/proteins.parquet     (+ corpus.fasta)
    data/embeddings/corpus_{pooling}.npy
    data/embeddings/accessions.json / store_meta.json

Rows append at the end, keeping parquet order aligned with the store. Appended
proteins have no split assignment, so probes simply exclude them; retrieval,
maps, and profiles include them. Re-run ``scripts/build_index.py`` afterwards —
the embedding-bytes fingerprint invalidates cached projections automatically.

Usage:
    python scripts/add_proteins.py                 # missing showcase proteins
    python scripts/add_proteins.py --accessions P12345 Q67890
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.corpus import OUTPUT_COLUMNS, SHOWCASE_ACCESSIONS, derive_fields, load_raw  # noqa: E402
from ml.sequence import is_valid_sequence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accessions", nargs="+", default=None,
                        help="default: showcase accessions missing from the corpus")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/proteins.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings"))
    args = parser.parse_args()

    corpus = pd.read_parquet(args.corpus)
    have = set(corpus["accession"])
    wanted = args.accessions or SHOWCASE_ACCESSIONS
    missing = [acc for acc in wanted if acc not in have]
    if not missing:
        print("Nothing to add — all requested accessions already present.")
        return 0
    print(f"Adding {len(missing)}: {missing}")

    raw = load_raw(args.raw)
    rows = raw[raw["accession"].isin(missing)].copy()
    not_found = sorted(set(missing) - set(rows["accession"]))
    if not_found:
        print(f"  WARNING: not in raw data (skipped): {not_found}")
    rows["sequence"] = rows["sequence"].str.strip().str.upper()
    valid = rows["sequence"].map(lambda s: isinstance(s, str) and is_valid_sequence(s, 50, 512))
    if (~valid).any():
        print(f"  WARNING: failed validation (skipped): {rows.loc[~valid, 'accession'].tolist()}")
        rows = rows[valid]
    if rows.empty:
        print("No valid rows to add.")
        return 1
    rows = derive_fields(rows).sort_values("accession")[OUTPUT_COLUMNS]

    # --- embed with every available pooling -----------------------------------
    from ml.embeddings import EmbeddingPipeline

    store_meta = json.loads((args.embeddings / "store_meta.json").read_text())
    pipeline = EmbeddingPipeline(
        model_name=store_meta["model"],
        cache_path=None,
        attention_pooler_path=args.embeddings / "attention_pooler.pt",
    )
    poolings = [p for p in store_meta["poolings"] if p in pipeline.pooler.available()]
    if set(poolings) != set(store_meta["poolings"]):
        raise SystemExit(f"Store has poolings {store_meta['poolings']} but only "
                         f"{poolings} are computable — refusing a partial append.")

    encoded = pipeline.encoder.encode_batch(rows["sequence"].tolist())
    new_vectors: dict[str, np.ndarray] = {}
    for pooling in poolings:
        vecs = []
        for enc in encoded:
            pooled, _ = pipeline.pooler.pool(enc.residue_embeddings, enc.bos_embedding, pooling)
            vecs.append(pooled.numpy().astype(np.float32))
        new_vectors[pooling] = np.stack(vecs)

    # --- append everything (parquet last: matrices first keeps store readable) --
    for pooling in poolings:
        path = args.embeddings / f"corpus_{pooling}.npy"
        matrix = np.load(path)
        np.save(path, np.concatenate([matrix, new_vectors[pooling]]))

    accessions = json.loads((args.embeddings / "accessions.json").read_text())
    accessions.extend(rows["accession"].tolist())
    (args.embeddings / "accessions.json").write_text(json.dumps(accessions))

    combined = pd.concat([corpus, rows], ignore_index=True)
    combined.to_parquet(args.corpus, index=False)
    with open(args.corpus.parent / "corpus.fasta", "a") as fh:
        for row in rows.itertuples():
            fh.write(f">{row.accession} {row.protein_name}\n{row.sequence}\n")

    store_meta["n_proteins"] = len(combined)
    store_meta["corpus_sha256_16"] = hashlib.sha256(args.corpus.read_bytes()).hexdigest()[:16]
    store_meta["appended"] = store_meta.get("appended", []) + rows["accession"].tolist()
    (args.embeddings / "store_meta.json").write_text(json.dumps(store_meta, indent=2))

    print(f"Appended {len(rows)} proteins → corpus now {len(combined)}. "
          f"Re-run scripts/build_index.py (and optionally run_benchmarks.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
