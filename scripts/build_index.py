"""Build all serving-time artifacts from precomputed embeddings.

Produces, under ``data/index/``:

    index_{pooling}.faiss + accession lists     exact cosine FAISS indexes
    projections/projection_{hash}.npz           cached PCA→UMAP coordinates
    map_{pooling}.json                          full map payload (coords +
                                                metadata + cluster + outlier)
    clusters_{pooling}.json                     cluster summaries

The API serves these files directly; nothing here is recomputed at request
time. Default map payloads are built for mean and attention pooling.

Usage:
    python scripts/build_index.py [--map-poolings mean attention] [--n-clusters 25]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.clustering import cluster_summaries, kmeans_clusters, outlier_scores  # noqa: E402
from ml.embeddings import EmbeddingStore     # noqa: E402
from ml.projection import ProjectionCache, ProjectionParams  # noqa: E402
from ml.retrieval import ProteinIndex        # noqa: E402
from ml.tracking import log_experiment       # noqa: E402

MAP_COLUMNS = [
    "accession", "protein_name", "gene", "organism_short", "length",
    "family", "pfam_primary", "ec_class", "is_enzyme", "localization",
]


def build_map_payload(
    df: pd.DataFrame,
    store: EmbeddingStore,
    pooling: str,
    out_dir: Path,
    n_clusters: int,
    seed: int,
) -> dict:
    embeddings = np.asarray(store.matrix(pooling))
    fingerprint = f"{store.meta['corpus_sha256_16']}:{pooling}"

    params = ProjectionParams(pooling=pooling, seed=seed)
    cache = ProjectionCache(out_dir / "projections")
    t0 = time.time()
    coords, info, cached = cache.load_or_compute(embeddings, params, fingerprint)
    print(f"  projection ({pooling}): {'cache' if cached else f'{time.time()-t0:.0f}s'}"
          f" | PCA var {info.get('pca_explained_variance', 0):.2f}")

    labels, cluster_info = kmeans_clusters(embeddings, n_clusters=n_clusters, seed=seed)
    print(f"  kmeans ({pooling}): {n_clusters} clusters, silhouette {cluster_info['silhouette_cosine']:.3f}")

    index = ProteinIndex.load(out_dir, pooling)
    knn_dist = index.knn_distances(k=10)
    outliers = outlier_scores(knn_dist)

    meta = df.set_index("accession").loc[store.accessions].reset_index()
    points = []
    for i, row in enumerate(meta.itertuples()):
        points.append({
            "id": row.accession,
            "name": row.protein_name,
            "gene": row.gene if isinstance(row.gene, str) else None,
            "org": row.organism_short,
            "len": int(row.length),
            "family": row.family if isinstance(row.family, str) else None,
            "pfam": row.pfam_primary if isinstance(row.pfam_primary, str) else None,
            "ec": row.ec_class if isinstance(row.ec_class, str) else None,
            "enzyme": bool(row.is_enzyme),
            "loc": row.localization if isinstance(row.localization, str) else None,
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "cluster": int(labels[i]),
            "knn_dist": round(float(knn_dist[i]), 5),
            "outlier": round(float(outliers[i]), 4),
        })

    payload = {
        "pooling": pooling,
        "model": store.meta["model"],
        "projection": info,
        "clustering": cluster_info,
        "points": points,
    }
    (out_dir / f"map_{pooling}.json").write_text(json.dumps(payload))

    summaries = cluster_summaries(meta, labels)
    (out_dir / f"clusters_{pooling}.json").write_text(
        json.dumps({"pooling": pooling, "clustering": cluster_info, "clusters": summaries})
    )
    return {"projection_cached": cached, **cluster_info}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/proteins.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--out", type=Path, default=Path("data/index"))
    parser.add_argument("--map-poolings", nargs="+", default=["mean", "attention"])
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.corpus)
    store = EmbeddingStore(args.embeddings)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Building FAISS indexes for: {store.poolings}")
    for pooling in store.poolings:
        index = ProteinIndex.build(np.asarray(store.matrix(pooling)), store.accessions, pooling)
        index.save(args.out)

    metrics = {}
    for pooling in args.map_poolings:
        if pooling not in store.poolings:
            print(f"  skipping map for '{pooling}' (no embeddings)")
            continue
        print(f"Map payload: {pooling}")
        metrics[pooling] = build_map_payload(df, store, pooling, args.out, args.n_clusters, args.seed)

    log_experiment(
        "build_index",
        config={"poolings": store.poolings, "map_poolings": args.map_poolings,
                "n_clusters": args.n_clusters, "seed": args.seed,
                "corpus_sha256_16": store.meta["corpus_sha256_16"]},
        metrics=metrics,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
