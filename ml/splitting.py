"""Leakage-aware train/validation/test splitting.

Random splits of protein datasets leak: homologous sequences land on both sides
and probe metrics become homology detection, not representation quality. We
split by *group*, where groups are connected components of a union-find over
annotation tokens: every protein links its UniProt family label and **all** of
its Pfam domains, so two proteins sharing a domain end up in one group even
when only one of them carries a family annotation (tiered fallbacks would leak
exactly those pairs). Unannotated proteins join the group of their most
similar annotated protein when 5-mer Jaccard ≥ 0.5, otherwise they form greedy
k-mer clusters among themselves.

All members of a group share a split. This is a pragmatic middle ground — it
does not replace a profile-HMM or MMseqs2 identity clustering, and remote
homology with no shared Pfam annotation can still cross splits;
``audit_leakage`` quantifies the residual risk by sampling cross-split pairs
and measuring k-mer similarity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.sequence import kmer_features


def _kmer_set(sequence: str, k: int = 5) -> set[str]:
    return {sequence[i : i + k] for i in range(len(sequence) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        root = x
        while self.parent.setdefault(root, root) != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def assign_groups(
    df: pd.DataFrame, k: int = 5, link_threshold: float = 0.5
) -> tuple[pd.Series, dict]:
    """Group id per protein.

    Annotated proteins: connected components over ``fam:*`` and ``pfam:*``
    tokens — a protein carrying family F and domains P1, P2 unions all three,
    so any shared domain (directly or via a family bridge) merges groups.
    Unannotated proteins: joined to the most k-mer-similar annotated protein's
    group when Jaccard ≥ ``link_threshold``, else clustered among themselves.
    """
    uf = _UnionFind()
    tokens_per_row: list[list[str]] = []
    for row in df.itertuples():
        tokens = []
        if isinstance(row.family, str):
            tokens.append(f"fam:{row.family}")
        pfams = row.pfam_all if isinstance(row.pfam_all, list | np.ndarray) else []
        tokens.extend(f"pfam:{p}" for p in pfams)
        tokens_per_row.append(tokens)
        for token in tokens[1:]:
            uf.union(tokens[0], token)

    groups = pd.Series(index=df.index, dtype="object")
    annotated_rows = []
    for idx, tokens in zip(df.index, tokens_per_row, strict=True):
        if tokens:
            groups[idx] = uf.find(tokens[0])
            annotated_rows.append(idx)

    rest = [idx for idx, tokens in zip(df.index, tokens_per_row, strict=True) if not tokens]
    n_linked = 0
    if rest:
        annotated_sets = [
            (idx, _kmer_set(df.at[idx, "sequence"], k)) for idx in annotated_rows
        ]
        orphan_sets: list[tuple[int, set[str]]] = []
        for idx in rest:
            kmers = _kmer_set(df.at[idx, "sequence"], k)
            best_idx, best_sim = None, 0.0
            for a_idx, a_set in annotated_sets:
                sim = _jaccard(kmers, a_set)
                if sim > best_sim:
                    best_idx, best_sim = a_idx, sim
            if best_idx is not None and best_sim >= link_threshold:
                groups[idx] = groups[best_idx]
                n_linked += 1
            else:
                # Greedy single-linkage among the remaining orphans.
                placed = False
                for o_idx, o_set in orphan_sets:
                    if _jaccard(kmers, o_set) >= link_threshold:
                        groups[idx] = groups[o_idx]
                        placed = True
                        break
                if not placed:
                    groups[idx] = f"kmer:{idx}"
                orphan_sets.append((idx, kmers))

    sizes = groups.value_counts()
    stats = {
        "annotated": int(len(annotated_rows)),
        "unannotated_linked_to_annotated": n_linked,
        "unannotated_own_cluster": int(len(rest) - n_linked),
        "n_groups": int(groups.nunique()),
        "largest_group": int(sizes.iloc[0]) if len(sizes) else 0,
        "largest_group_fraction": float(sizes.iloc[0] / len(df)) if len(df) else 0.0,
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
        # Ties break toward the largest split so that mega-components (multi-
        # domain proteins can chain many families into one group) land in
        # train instead of monopolizing a small evaluation split.
        best = max(names, key=lambda n: (deficits[n], targets[n], rng.random()))
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
