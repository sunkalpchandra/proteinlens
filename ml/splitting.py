"""Leakage-aware train/validation/test splitting.

Random splits of protein datasets leak: homologous sequences land on both sides
and probe metrics become homology detection, not representation quality. We
split by *group*, where a group is:

  1. the UniProt family annotation when present,
  2. else the primary Pfam domain,
  3. else a greedy 5-mer Jaccard cluster (threshold 0.5) over the remainder.

All members of a group share a split. This is a pragmatic middle ground — it
does not replace a profile-HMM or MMseqs2 identity clustering, and remote
cross-family homology can still cross splits; ``audit_leakage`` quantifies the
residual risk by sampling cross-split pairs and measuring k-mer similarity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.sequence import kmer_features


def _jaccard_clusters(sequences: list[str], k: int = 5, threshold: float = 0.5) -> list[int]:
    """Greedy single-linkage clustering on k-mer set Jaccard similarity."""
    kmer_sets = [
        {seq[i : i + k] for i in range(len(seq) - k + 1)} for seq in sequences
    ]
    labels = [-1] * len(sequences)
    next_label = 0
    for i in range(len(sequences)):
        if labels[i] != -1:
            continue
        labels[i] = next_label
        for j in range(i + 1, len(sequences)):
            if labels[j] != -1:
                continue
            inter = len(kmer_sets[i] & kmer_sets[j])
            union = len(kmer_sets[i] | kmer_sets[j])
            if union and inter / union >= threshold:
                labels[j] = next_label
        next_label += 1
    return labels


def assign_groups(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Group id per protein using family → Pfam → k-mer cluster fallback."""
    groups = pd.Series(index=df.index, dtype="object")

    has_family = df["family"].notna()
    groups[has_family] = "fam:" + df.loc[has_family, "family"]

    has_pfam = ~has_family & df["pfam_primary"].notna()
    groups[has_pfam] = "pfam:" + df.loc[has_pfam, "pfam_primary"]

    rest = df.index[~has_family & ~has_pfam]
    if len(rest) > 0:
        labels = _jaccard_clusters(df.loc[rest, "sequence"].tolist())
        groups[rest] = [f"kmer:{label}" for label in labels]

    stats = {
        "by_family": int(has_family.sum()),
        "by_pfam": int(has_pfam.sum()),
        "by_kmer_cluster": int(len(rest)),
        "n_groups": int(groups.nunique()),
    }
    return groups, stats


def make_splits(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[pd.Series, dict]:
    """Assign each protein to train/val/test such that groups never straddle splits.

    Groups are shuffled and greedily assigned to the split furthest below its
    target fraction (largest groups first, so big families cannot overshoot a
    small split at the end).
    """
    assert abs(sum(ratios) - 1.0) < 1e-9
    groups, group_stats = assign_groups(df)
    rng = np.random.default_rng(seed)

    sizes = groups.value_counts()
    order = sizes.sample(frac=1.0, random_state=seed).sort_values(ascending=False, kind="stable")

    names = ("train", "val", "test")
    targets = {name: ratio * len(df) for name, ratio in zip(names, ratios, strict=True)}
    filled = dict.fromkeys(names, 0)
    assignment: dict[str, str] = {}
    for group, size in order.items():
        deficits = {name: (targets[name] - filled[name]) / max(targets[name], 1) for name in names}
        best = max(names, key=lambda n: (deficits[n], rng.random()))
        assignment[group] = best
        filled[best] += size

    splits = groups.map(assignment)
    summary = {
        "seed": seed,
        "ratios": list(ratios),
        "group_stats": group_stats,
        "split_sizes": splits.value_counts().to_dict(),
        "groups_per_split": {
            name: int(sum(1 for g, s in assignment.items() if s == name)) for name in names
        },
    }
    return splits, summary


def audit_leakage(
    df: pd.DataFrame,
    splits: pd.Series,
    n_pairs: int = 2000,
    k: int = 4,
    seed: int = 42,
) -> dict:
    """Estimate residual train↔test similarity with k-mer cosine similarity.

    Samples random cross-split pairs plus, as a reference, random train↔train
    pairs. High cross-split similarity relative to the reference would indicate
    leakage the grouping failed to prevent.
    """
    rng = np.random.default_rng(seed)
    train_idx = df.index[splits == "train"].to_numpy()
    test_idx = df.index[splits == "test"].to_numpy()

    def pair_sims(a_idx: np.ndarray, b_idx: np.ndarray) -> np.ndarray:
        rows_a = rng.choice(a_idx, n_pairs)
        rows_b = rng.choice(b_idx, n_pairs)
        sims = np.empty(n_pairs, dtype=np.float32)
        feats: dict[int, np.ndarray] = {}

        def feat(i: int) -> np.ndarray:
            if i not in feats:
                v = kmer_features(df.at[i, "sequence"], k=k)
                feats[i] = v / max(np.linalg.norm(v), 1e-12)
            return feats[i]

        for n, (i, j) in enumerate(zip(rows_a, rows_b, strict=True)):
            sims[n] = float(np.dot(feat(int(i)), feat(int(j))))
        return sims

    cross = pair_sims(train_idx, test_idx)
    within = pair_sims(train_idx, train_idx)
    return {
        "n_pairs": n_pairs,
        "kmer_k": k,
        "cross_split": {
            "mean": float(cross.mean()),
            "p95": float(np.quantile(cross, 0.95)),
            "p99": float(np.quantile(cross, 0.99)),
            "max": float(cross.max()),
        },
        "train_reference": {
            "mean": float(within.mean()),
            "p95": float(np.quantile(within, 0.95)),
            "p99": float(np.quantile(within, 0.99)),
            "max": float(within.max()),
        },
    }


def save_splits(splits: pd.Series, df: pd.DataFrame, summary: dict, path: str | Path) -> None:
    payload = {
        "summary": summary,
        "splits": dict(zip(df["accession"], splits, strict=True)),
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_splits(path: str | Path) -> tuple[dict[str, str], dict]:
    payload = json.loads(Path(path).read_text())
    return payload["splits"], payload["summary"]
