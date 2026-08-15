"""Build the static demo bundle served by the deployed frontend.

Selects a ~1,200-protein subset of the corpus (all showcase proteins, the
largest families for visible structure, and a stratified sample of the rest),
then writes everything the frontend's demo mode needs into
``frontend/public/demo/``:

    map_mean.json            subset of the full map payload (same coordinates)
    proteins.json            summaries for the text finder
    profiles/{acc}.json      profile + neighbors (restricted to the subset so
                             every link resolves) (+ attention for showcase)
    landscapes/{acc}.json    precomputed 19-way mutation landscapes for
                             showcase proteins at selected positions
    benchmark.json           benchmark rows + seq-vs-emb sample + markdown
    health.json              static corpus metadata

Requires embeddings + index artifacts; benchmark files are optional (skipped
with a warning if absent). Landscape computation runs the real model.

Usage:
    python scripts/build_demo_bundle.py [--size 1200] [--skip-inference]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.embeddings import EmbeddingStore  # noqa: E402
from ml.projection import MAP_PRESETS, map_filename  # noqa: E402
from ml.retrieval import ProteinIndex  # noqa: E402

# Well-known proteins that make the demo immediately legible (kept in the
# corpus by preprocessing — see ml.corpus.SHOWCASE_ACCESSIONS). Positions are
# 1-based sequence positions (initiator Met included) chosen for
# recognizability; e.g. HBB position 7 = sickle-cell site β6.
SHOWCASE: dict[str, list[int]] = {
    "P69905": [59, 63, 88],     # Hemoglobin α: distal His region / proximal His
    "P68871": [7, 64, 93],      # Hemoglobin β: sickle site, distal/proximal His
    "P02144": [65, 94],         # Myoglobin: distal/proximal His
    "P01308": [48, 90],         # Insulin
    "P61626": [53, 63],         # Lysozyme C: catalytic-region residues
    "P0DP23": [21, 77],         # Calmodulin-1: EF-hand region
    "P04637": [175, 248, 273],  # p53: hotspot codons R175/R248/R273
    "P01112": [12, 13, 61],     # HRAS: G12/G13/Q61
    "P00441": [5, 94],          # SOD1: A4V site (pos 5 with Met)
    "P68431": [5, 10, 28],      # Histone H3.1: K4/K9/K27 sites (+1 for Met)
}


def summary_of(row: pd.Series) -> dict:
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
    }


def select_subset(df: pd.DataFrame, size: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    chosen: set[str] = {acc for acc in SHOWCASE if acc in set(df["accession"])}

    top_families = df["family"].value_counts().head(12).index
    for family in top_families:
        chosen.update(df[df["family"] == family]["accession"])

    remainder = df[~df["accession"].isin(chosen)]
    n_left = max(0, size - len(chosen))
    if n_left and len(remainder):
        frac = n_left / len(remainder)
        for _, group in remainder.groupby("organism_short"):
            k = max(1, int(round(len(group) * frac)))
            picks = rng.choice(group["accession"].to_numpy(), size=min(k, len(group)), replace=False)
            chosen.update(picks.tolist())
    return sorted(chosen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/proteins.parquet"))
    parser.add_argument("--out", type=Path, default=Path("frontend/public/demo"))
    parser.add_argument("--size", type=int, default=1200)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-inference", action="store_true",
                        help="skip attention weights and mutation landscapes")
    args = parser.parse_args()

    df = pd.read_parquet(args.corpus)
    by_acc = df.set_index("accession", drop=False)
    store = EmbeddingStore("data/embeddings")
    index = ProteinIndex.load("data/index", "mean")
    map_payload = json.loads(Path("data/index/map_mean.json").read_text())
    point_of = {p["id"]: p for p in map_payload["points"]}

    pdb_path = Path("data/processed/pdb_xrefs.parquet")
    pdb_by_acc: dict[str, list[str]] = {}
    if pdb_path.exists():
        for row in pd.read_parquet(pdb_path).itertuples():
            pdb_by_acc[row.accession] = list(row.pdb_ids)[:12]

    domains_path = Path("data/processed/domains.parquet")
    domains_by_acc: dict[str, list[dict]] = {}
    if domains_path.exists():
        for row in pd.read_parquet(domains_path).itertuples():
            domains_by_acc.setdefault(row.accession, []).append(
                {"name": row.name, "start": int(row.start), "end": int(row.end)}
            )

    subset = select_subset(df, args.size, args.seed)
    subset_set = set(subset)
    print(f"Demo subset: {len(subset)} proteins "
          f"({sum(1 for a in SHOWCASE if a in subset_set)} showcase)")

    out = args.out
    (out / "profiles").mkdir(parents=True, exist_ok=True)
    (out / "landscapes").mkdir(parents=True, exist_ok=True)

    # --- map + finder ---------------------------------------------------------
    for preset in MAP_PRESETS:
        name = map_filename("mean", preset)
        src = Path("data/index") / name
        if not src.exists():
            print(f"  WARNING: {name} not built — demo will lack the '{preset}' preset")
            continue
        full = json.loads(src.read_text()) if preset != "default" else map_payload
        by_id = {p["id"]: p for p in full["points"]}
        demo_map = {
            **{k: v for k, v in full.items() if k != "points"},
            "points": [by_id[acc] for acc in subset if acc in by_id],
        }
        (out / name).write_text(json.dumps(demo_map))
    # Cluster summaries travel as-is: they describe full-corpus clusters, and
    # the map subset carries the same cluster ids.
    for cluster_file in ("clusters_mean.json", "clusters_mean_hdbscan.json"):
        src = Path("data/index") / cluster_file
        if src.exists():
            (out / cluster_file).write_text(src.read_text())
    (out / "proteins.json").write_text(
        json.dumps([summary_of(by_acc.loc[acc]) for acc in subset])
    )

    # --- optional model inference ------------------------------------------------
    attention: dict[str, list[float]] = {}
    if not args.skip_inference:
        from ml.embeddings import EmbeddingPipeline
        from models.mutation import MutationAnalyzer

        pipeline = EmbeddingPipeline(cache_path=None)
        analyzer = MutationAnalyzer(pipeline)
        pooler = pipeline.pooler.attention_pooler

        for acc, positions in SHOWCASE.items():
            if acc not in subset_set:
                continue
            seq = by_acc.loc[acc, "sequence"]
            if pooler is not None:
                import torch

                with torch.inference_mode():
                    encoded = pipeline.encode_residues(seq)
                    _, weights = pooler(encoded.residue_embeddings)
                attention[acc] = [round(float(w), 6) for w in weights.numpy()]

            landscapes = {}
            for pos in positions:
                if 1 <= pos <= len(seq):
                    landscape = analyzer.landscape(seq, pos, "mean")
                    landscape["note"] = (
                        "Representation-space perturbation of a frozen protein "
                        "language model; not a fitness, stability, or "
                        "pathogenicity prediction."
                    )
                    landscapes[str(pos)] = landscape
            (out / "landscapes" / f"{acc}.json").write_text(json.dumps(landscapes))
            print(f"  landscapes: {acc} at {list(landscapes)}")

    # --- profiles ------------------------------------------------------------------
    for acc in subset:
        row = by_acc.loc[acc]
        vector = store.vector(acc, "mean")
        point = point_of.get(acc)

        hits = index.search(vector, k=120, exclude=acc)
        neighbors = [
            {"rank": n + 1, "similarity": round(h.score, 4),
             "protein": summary_of(by_acc.loc[h.accession])}
            for n, h in enumerate(h for h in hits if h.accession in subset_set)
        ][: args.neighbors]

        keywords = row["keywords"] if isinstance(row["keywords"], str) else ""
        profile = {
            "protein": summary_of(row),
            "protein_name_full": row["protein_name_full"],
            "pdb": pdb_by_acc.get(acc, []),
            "keywords": [k.strip() for k in keywords.split(";") if k.strip()][:20],
            "sequence": row["sequence"],
            "model": store.meta["model"],
            "stats": {
                "embedding_norm": round(float(np.linalg.norm(vector)), 4),
                "dim": int(vector.shape[0]),
                "nn_distance": round(1.0 - hits[0].score, 5) if hits else None,
                "knn_mean_distance": point["knn_dist"] if point else None,
                "cluster": point["cluster"] if point else None,
                "outlier_score": point["outlier"] if point else None,
                "x": point["x"] if point else None,
                "y": point["y"] if point else None,
            },
            "neighbors": neighbors,
        }
        if acc in attention:
            profile["attention_weights"] = attention[acc]
        if acc in domains_by_acc:
            profile["domains"] = domains_by_acc[acc]
        (out / "profiles" / f"{acc}.json").write_text(json.dumps(profile))

    # --- benchmark + health -----------------------------------------------------------
    bench_csv = Path("reports/benchmark.csv")
    if bench_csv.exists():
        payload: dict = {"rows": json.loads(pd.read_csv(bench_csv).to_json(orient="records"))}
        sve = Path("reports/seq_vs_emb.csv")
        if sve.exists():
            frame = pd.read_csv(sve)
            if len(frame) > 2000:
                frame = frame.sample(2000, random_state=args.seed)
            payload["seq_vs_emb"] = json.loads(frame.to_json(orient="records"))
        md = Path("reports/benchmark.md")
        if md.exists():
            payload["markdown"] = md.read_text()
        extended = Path("reports/extended_benchmark.csv")
        if extended.exists():
            payload["extended"] = json.loads(
                pd.read_csv(extended).to_json(orient="records")
            )
        (out / "benchmark.json").write_text(json.dumps(payload))
    else:
        print("  WARNING: no reports/benchmark.csv — demo benchmark page will 404")

    (out / "health.json").write_text(json.dumps({
        "status": "ok",
        "model": store.meta["model"],
        "corpus_size": len(subset),
        "poolings": ["mean"],
        "device": "static demo",
        "encoder_loaded": False,
    }))

    total_kb = sum(f.stat().st_size for f in out.rglob("*.json")) / 1024
    print(f"Demo bundle: {total_kb/1024:.1f} MB across {sum(1 for _ in out.rglob('*.json'))} files → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
