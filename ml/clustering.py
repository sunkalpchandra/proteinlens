"""Clustering and representation-space outlier detection.

K-means (fixed k, always available) and HDBSCAN (density-based, labels sparse
regions as noise) run on L2-normalized embeddings. Outlier scores are the mean
cosine distance to the k nearest neighbors — proteins that are unusually far
from everything else in the *selected representation space*. That is a
geometric statement about the embedding, not a biological anomaly claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.embeddings import l2_normalize


def kmeans_clusters(
    embeddings: np.ndarray, n_clusters: int = 25, seed: int = 42
) -> tuple[np.ndarray, dict]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    x = l2_normalize(embeddings)
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=4)
    labels = model.fit_predict(x)

    # Silhouette on a subsample keeps this O(sample²), not O(N²).
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(x), size=min(3000, len(x)), replace=False)
    silhouette = float(silhouette_score(x[sample], labels[sample], metric="cosine"))
    return labels, {"algorithm": "kmeans", "n_clusters": n_clusters,
                    "inertia": float(model.inertia_), "silhouette_cosine": silhouette}


def hdbscan_clusters(
    embeddings: np.ndarray, min_cluster_size: int = 15, min_samples: int = 5
) -> tuple[np.ndarray, dict]:
    from sklearn.cluster import HDBSCAN

    x = l2_normalize(embeddings)
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                    metric="euclidean")  # euclidean on unit sphere ≈ monotone in cosine
    labels = model.fit_predict(x)
    n_clusters = int(labels.max()) + 1
    return labels, {"algorithm": "hdbscan", "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples, "n_clusters": n_clusters,
                    "noise_fraction": float((labels == -1).mean())}


def cluster_summaries(df: pd.DataFrame, labels: np.ndarray, top: int = 3) -> list[dict]:
    """Per-cluster size, length stats, dominant families/organisms."""
    frame = df.assign(cluster=labels)
    summaries = []
    for cluster_id, group in frame.groupby("cluster"):
        summaries.append(
            {
                "cluster": int(cluster_id),
                "size": int(len(group)),
                "mean_length": float(group["length"].mean()),
                "top_families": group["family"].value_counts().head(top).to_dict(),
                "top_organisms": group["organism_short"].value_counts().head(top).to_dict(),
                "enzyme_fraction": float(group["is_enzyme"].mean()),
            }
        )
    return sorted(summaries, key=lambda s: -s["size"])


def outlier_scores(knn_mean_distance: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1] of each protein's mean k-NN cosine distance.

    1.0 = the most isolated protein in this representation space.
    """
    order = knn_mean_distance.argsort().argsort()
    return (order / max(len(knn_mean_distance) - 1, 1)).astype(np.float32)


def cluster_purity(labels: np.ndarray, categories: pd.Series) -> float:
    """Fraction of proteins whose cluster's majority category matches theirs.

    Only rows with a non-null category participate. Standard purity: for each
    cluster take its most common category, sum those counts, divide by total.
    """
    frame = pd.DataFrame({"cluster": labels, "category": categories.values})
    frame = frame[frame["category"].notna()]
    if frame.empty:
        return float("nan")
    total = 0
    for _, group in frame.groupby("cluster"):
        total += int(group["category"].value_counts().iloc[0])
    return total / len(frame)
