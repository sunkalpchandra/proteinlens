"""Unsupervised representation quality metrics.

  * retrieval precision@k — do a protein's nearest neighbors share its label?
  * cluster purity / NMI    — does unsupervised structure recover annotations?
  * perturbation stability  — how far does one random substitution move z?

All metrics operate on L2-normalized embeddings (cosine geometry), matching
what the search engine and map actually use.
"""

from __future__ import annotations

import faiss
import numpy as np
import pandas as pd

from ml.embeddings import l2_normalize
from ml.sequence import CANONICAL_AA


def retrieval_precision_at_k(
    embeddings: np.ndarray,
    labels: pd.Series,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Mean fraction of top-k cosine neighbors sharing the query's label.

    Only proteins whose label occurs ≥ 2 times can have a correct neighbor;
    others are excluded from the average (reported as ``n_evaluated``).
    """
    labeled = labels.notna().to_numpy()
    counts = labels.value_counts()
    eligible = labeled & (labels.map(counts).fillna(0).to_numpy() >= 2)

    x = l2_normalize(np.ascontiguousarray(embeddings[eligible], dtype=np.float32))
    y = labels[eligible].to_numpy()
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    k_max = max(ks)
    _, rows = index.search(x, k_max + 1)

    hits = {k: [] for k in ks}
    for i in range(len(x)):
        neighbors = [r for r in rows[i] if r != i][:k_max]
        matches = np.array([y[r] == y[i] for r in neighbors], dtype=np.float32)
        for k in ks:
            hits[k].append(matches[:k].mean())
    return {
        "n_evaluated": int(len(x)),
        **{f"precision@{k}": float(np.mean(hits[k])) for k in ks},
    }


def clustering_agreement(
    embeddings: np.ndarray,
    categories: pd.Series,
    n_clusters: int = 25,
    seed: int = 42,
) -> dict:
    """K-means purity and NMI against an annotation column."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score

    from ml.clustering import cluster_purity

    x = l2_normalize(embeddings)
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=4).fit_predict(x)
    mask = categories.notna().to_numpy()
    return {
        "n_clusters": n_clusters,
        "purity": float(cluster_purity(labels, categories)),
        "nmi": float(
            normalized_mutual_info_score(categories[mask], labels[mask])
        ),
    }


def perturbation_pairs(
    sequences: list[str], n: int = 150, seed: int = 42
) -> list[tuple[int, str, str]]:
    """(index, wild-type sequence, sequence with one random substitution)."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(sequences), size=min(n, len(sequences)), replace=False)
    pairs = []
    for idx in chosen:
        seq = sequences[int(idx)]
        pos = int(rng.integers(len(seq)))
        alternatives = [aa for aa in CANONICAL_AA if aa != seq[pos]]
        mutant = seq[:pos] + str(rng.choice(alternatives)) + seq[pos + 1 :]
        pairs.append((int(idx), seq, mutant))
    return pairs


def stability_from_vectors(
    wild: np.ndarray, mutated: np.ndarray
) -> dict:
    """Cosine similarity statistics between wild-type and 1-substitution vectors.

    High mean = representation barely moves for single substitutions (smooth);
    the spread shows how position/residue dependent the response is.
    """
    a = l2_normalize(wild)
    b = l2_normalize(mutated)
    sims = np.sum(a * b, axis=1)
    return {
        "n_pairs": int(len(sims)),
        "cosine_mean": float(sims.mean()),
        "cosine_std": float(sims.std()),
        "cosine_p05": float(np.quantile(sims, 0.05)),
        "cosine_min": float(sims.min()),
    }
