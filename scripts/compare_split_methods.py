"""Compare annotation-based and MMseqs2 identity-based split groupings.

Answers three questions and writes ``reports/split_methods.md``:

  1. How do the groupings differ structurally (group counts, sizes)?
  2. Do they *disagree* — pairs one method separates that the other joins?
     (Pairs MMseqs joins but annotation separates are potential leaks under
     annotation grouping; the reverse are cases where annotations link what
     sequence identity cannot see.)
  3. Do probe metrics move when the split method changes? (ESM-2 mean
     embeddings, same probe suite, both split files.)

Usage:
    python scripts/compare_split_methods.py [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.embeddings import EmbeddingStore  # noqa: E402
from ml.probes import build_tasks, run_probe  # noqa: E402
from ml.splitting import assign_groups, audit_leakage, make_splits, mmseqs_groups  # noqa: E402
from ml.tracking import log_experiment  # noqa: E402


def disagreement_stats(groups_a: pd.Series, groups_b: pd.Series, n_pairs: int, seed: int) -> dict:
    """Sample random protein pairs; count joins/separations per method."""
    rng = np.random.default_rng(seed)
    idx = groups_a.index.to_numpy()
    a_joins_b_separates = b_joins_a_separates = both_join = 0
    for _ in range(n_pairs):
        i, j = rng.choice(idx, 2, replace=False)
        same_a = groups_a[i] == groups_a[j]
        same_b = groups_b[i] == groups_b[j]
        if same_a and same_b:
            both_join += 1
        elif same_a:
            a_joins_b_separates += 1
        elif same_b:
            b_joins_a_separates += 1
    return {
        "sampled_pairs": n_pairs,
        "both_join": both_join,
        "annotation_joins_mmseqs_separates": a_joins_b_separates,
        "mmseqs_joins_annotation_separates": b_joins_a_separates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/proteins.parquet"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--pairs", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.corpus)
    store = EmbeddingStore("data/embeddings")
    embeddings = np.asarray(store.matrix("mean"))
    assert store.accessions == df["accession"].tolist()

    print("Grouping (annotation union-find)…")
    ann_groups, ann_stats = assign_groups(df)
    print("Grouping (MMseqs2 30% identity)…")
    mm_groups, mm_stats = mmseqs_groups(df)

    disagreement = disagreement_stats(ann_groups, mm_groups, args.pairs, args.seed)
    print(f"Disagreement on {args.pairs:,} random pairs: {disagreement}")

    results: dict[str, dict] = {}
    for method in ("annotation", "mmseqs"):
        print(f"Splitting + probing under {method}…")
        splits, summary = make_splits(df, seed=args.seed, method=method)
        audit = audit_leakage(df, splits, seed=args.seed)
        probe_rows = []
        for task in build_tasks(df):
            metrics = run_probe(embeddings, task, splits, seed=args.seed)
            probe_rows.append(metrics)
            print(f"  {task.name:<26} acc {metrics['accuracy']:.3f} "
                  f"macroF1 {metrics['macro_f1']:.3f}")
        results[method] = {"summary": summary, "audit": audit, "probes": probe_rows}

    lines = [
        "# Split methods: annotation union-find vs MMseqs2 identity clustering",
        "",
        "| | annotation | mmseqs (30% id, 80% cov) |",
        "|---|---|---|",
        f"| groups | {ann_stats['n_groups']:,} | {mm_stats['n_groups']:,} |",
        f"| largest group | {ann_stats['largest_group']:,} "
        f"({ann_stats['largest_group_fraction']:.1%}) | {mm_stats['largest_group']:,} "
        f"({mm_stats['largest_group_fraction']:.1%}) |",
        f"| singleton fraction | — | {mm_stats['singleton_fraction']:.1%} |",
        "",
        "## Where the methods disagree",
        "",
        f"On {disagreement['sampled_pairs']:,} random pairs: "
        f"{disagreement['annotation_joins_mmseqs_separates']} joined only by annotation "
        "(annotations link what <30% identity cannot see — safe, conservative), "
        f"{disagreement['mmseqs_joins_annotation_separates']} joined only by MMseqs "
        "(**potential leaks under annotation grouping** — homologous by identity yet "
        "annotation-disjoint).",
        "",
        "## Probe metrics under each split (ESM-2 mean pooling)",
        "",
        "| task | accuracy (annotation) | accuracy (mmseqs) | macro-F1 (annotation) | macro-F1 (mmseqs) |",
        "|---|---|---|---|---|",
    ]
    for row_a, row_m in zip(results["annotation"]["probes"], results["mmseqs"]["probes"], strict=True):
        lines.append(
            f"| {row_a['task']} | {row_a['accuracy']:.3f} | {row_m['accuracy']:.3f} "
            f"| {row_a['macro_f1']:.3f} | {row_m['macro_f1']:.3f} |"
        )
    for method in ("annotation", "mmseqs"):
        audit = results[method]["audit"]
        lines.append("")
        lines.append(
            f"Leakage audit ({method}): cross-split 4-mer cosine p99 "
            f"{audit['cross_split']['p99']:.3f} vs within-train "
            f"{audit['train_reference']['p99']:.3f}."
        )
    lines += ["", "Close agreement between the two probe columns indicates the "
              "annotation grouping was already controlling the leakage that "
              "identity clustering formalizes.", ""]
    (args.reports / "split_methods.md").write_text("\n".join(lines))
    print(f"Wrote {args.reports / 'split_methods.md'}")

    log_experiment("split_methods",
                   config={"pairs": args.pairs, "seed": args.seed},
                   metrics={"disagreement": disagreement,
                            "annotation": ann_stats, "mmseqs": mm_stats})
    return 0


if __name__ == "__main__":
    sys.exit(main())
