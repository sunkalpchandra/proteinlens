"""Representation benchmark suite.

Compares four ESM-2 pooling strategies against two sequence-only baselines:

    kmer3       3-mer frequency vector (8000 d)
    onehot      position-averaged one-hot composition (20 d)
    esm_mean / esm_max / esm_bos / esm_attention

across four axes:

    probes      linear probes for enzyme / EC class / localization
                (leakage-aware family-grouped splits)
    retrieval   precision@k for Pfam domain and family labels
    clustering  k-means purity + NMI against family annotations
    stability   cosine(z_wt, z_mut) for one random substitution

Writes reports/benchmark.csv, reports/benchmark.md, reports/seq_vs_emb.csv,
and an experiments/ entry. Nothing is hand-entered; re-running regenerates
every number from the artifacts on disk.

Usage:
    python scripts/run_benchmarks.py [--skip-stability] [--pairs 4000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.embeddings import EmbeddingPipeline, EmbeddingStore, l2_normalize  # noqa: E402
from ml.evaluation import (  # noqa: E402
    clustering_agreement,
    perturbation_pairs,
    retrieval_precision_at_k,
    stability_from_vectors,
)
from ml.probes import build_tasks, run_probe  # noqa: E402
from ml.sequence import kmer_features, onehot_mean_features, sequence_identity  # noqa: E402
from ml.splitting import load_splits  # noqa: E402
from ml.tracking import log_experiment  # noqa: E402

ESM_POOLINGS = ["mean", "max", "bos", "attention"]


def build_representations(df: pd.DataFrame, store: EmbeddingStore) -> dict[str, np.ndarray]:
    print("Building baseline features…")
    t0 = time.time()
    kmer = np.stack([kmer_features(s, k=3) for s in df["sequence"]])
    onehot = np.stack([onehot_mean_features(s) for s in df["sequence"]])
    print(f"  kmer3 {kmer.shape}, onehot {onehot.shape} in {time.time()-t0:.0f}s")

    reps: dict[str, np.ndarray] = {"kmer3": kmer, "onehot": onehot}
    for pooling in ESM_POOLINGS:
        if pooling in store.poolings:
            reps[f"esm_{pooling}"] = np.asarray(store.matrix(pooling))
    return reps


def sequence_vs_embedding(
    df: pd.DataFrame,
    store: EmbeddingStore,
    n_pairs: int,
    seed: int,
    out_path: Path,
) -> dict:
    """Sample protein pairs, compute alignment identity vs embedding cosine.

    Random pairs alone are nearly all <15% identity, so half the sample comes
    from embedding nearest neighbors to populate the interesting region.
    """
    import faiss

    rng = np.random.default_rng(seed)
    emb = l2_normalize(np.asarray(store.matrix("mean")))
    n = len(df)

    pairs: set[tuple[int, int]] = set()
    while len(pairs) < n_pairs // 2:
        i, j = rng.integers(n), rng.integers(n)
        if i != j:
            pairs.add((min(i, j), max(i, j)))

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    seeds = rng.choice(n, size=n_pairs // 8, replace=False)
    _, rows = index.search(emb[seeds], 5)
    for s, row in zip(seeds, rows, strict=True):
        for r in row[1:]:
            pairs.add((min(int(s), int(r)), max(int(s), int(r))))

    records = []
    sequences = df["sequence"].tolist()
    t0 = time.time()
    for i, j in pairs:
        identity = sequence_identity(sequences[i], sequences[j])
        records.append({
            "a": df["accession"].iat[i], "b": df["accession"].iat[j],
            "identity": round(identity, 4),
            "cosine": round(float(np.dot(emb[i], emb[j])), 4),
            "same_family": bool(
                isinstance(df["family"].iat[i], str)
                and df["family"].iat[i] == df["family"].iat[j]
            ),
        })
    frame = pd.DataFrame(records)
    frame.to_csv(out_path, index=False)
    print(f"  {len(frame)} pairs aligned in {time.time()-t0:.0f}s → {out_path}")

    discordant_high = frame[(frame.identity < 0.20) & (frame.cosine > 0.90)]
    discordant_low = frame[(frame.identity > 0.50) & (frame.cosine < 0.70)]
    return {
        "n_pairs": len(frame),
        "pearson_r": float(frame["identity"].corr(frame["cosine"])),
        "low_identity_high_cosine": int(len(discordant_high)),
        "high_identity_low_cosine": int(len(discordant_low)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/proteins.parquet"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--pairs", type=int, default=4000)
    parser.add_argument("--stability-n", type=int, default=150)
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.reports.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.corpus)
    split_map, split_summary = load_splits(args.splits)
    splits = df["accession"].map(split_map)
    store = EmbeddingStore(args.embeddings)
    assert store.accessions == df["accession"].tolist(), "store/corpus order mismatch"

    reps = build_representations(df, store)
    rows = []

    # --- Probes -------------------------------------------------------------
    tasks = build_tasks(df)
    for rep_name, matrix in reps.items():
        for task in tasks:
            t0 = time.time()
            metrics = run_probe(matrix, task, splits, seed=args.seed)
            rows.append({"representation": rep_name, "axis": "probe", **metrics})
            print(f"probe {task.name:<26} {rep_name:<14} acc {metrics['accuracy']:.3f} "
                  f"macroF1 {metrics['macro_f1']:.3f} ({time.time()-t0:.0f}s)")

    # --- Retrieval ------------------------------------------------------------
    for rep_name, matrix in reps.items():
        for label_name in ("pfam_primary", "family"):
            metrics = retrieval_precision_at_k(matrix, df[label_name])
            rows.append({"representation": rep_name, "axis": "retrieval",
                         "task": f"same_{label_name}", **metrics})
            print(f"retrieval {label_name:<14} {rep_name:<14} "
                  f"P@1 {metrics['precision@1']:.3f} P@10 {metrics['precision@10']:.3f}")

    # --- Clustering agreement ---------------------------------------------------
    for rep_name, matrix in reps.items():
        metrics = clustering_agreement(matrix, df["family"], seed=args.seed)
        rows.append({"representation": rep_name, "axis": "clustering",
                     "task": "kmeans_vs_family", **metrics})
        print(f"clustering {rep_name:<14} purity {metrics['purity']:.3f} NMI {metrics['nmi']:.3f}")

    # --- Stability -----------------------------------------------------------
    if not args.skip_stability:
        pairs = perturbation_pairs(df["sequence"].tolist(), n=args.stability_n, seed=args.seed)
        pipeline = EmbeddingPipeline(model_name=store.meta["model"], cache_path=None)
        mut_seqs = [m for _, _, m in pairs]
        wt_rows = [i for i, _, _ in pairs]
        for rep_name in reps:
            if rep_name == "kmer3":
                wt = np.stack([kmer_features(s) for _, s, _ in pairs])
                mut = np.stack([kmer_features(m) for m in mut_seqs])
            elif rep_name == "onehot":
                wt = np.stack([onehot_mean_features(s) for _, s, _ in pairs])
                mut = np.stack([onehot_mean_features(m) for m in mut_seqs])
            else:
                pooling = rep_name.removeprefix("esm_")
                if pooling not in store.poolings:
                    continue
                wt = np.asarray(store.matrix(pooling))[wt_rows]
                mut = np.stack(pipeline.embed_batch(mut_seqs, pooling))
            metrics = stability_from_vectors(wt, mut)
            rows.append({"representation": rep_name, "axis": "stability",
                         "task": "one_substitution", **metrics})
            print(f"stability {rep_name:<14} cos mean {metrics['cosine_mean']:.4f}")

    # --- Sequence vs embedding similarity ------------------------------------
    sve = sequence_vs_embedding(df, store, args.pairs, args.seed,
                                args.reports / "seq_vs_emb.csv")
    print(f"seq-vs-emb: r = {sve['pearson_r']:.3f}")

    # --- Write ------------------------------------------------------------------
    table = pd.DataFrame(rows)
    table.to_csv(args.reports / "benchmark.csv", index=False)
    write_markdown(table, sve, split_summary, store, args.reports / "benchmark.md")

    log_experiment(
        "benchmark",
        config={"model": store.meta["model"], "seed": args.seed,
                "corpus_sha256_16": store.meta["corpus_sha256_16"],
                "split_summary": split_summary, "pairs": args.pairs},
        metrics={"rows": len(rows), "seq_vs_embedding": sve},
    )
    print(f"Wrote {args.reports / 'benchmark.csv'} and benchmark.md")
    return 0


def write_markdown(table: pd.DataFrame, sve: dict, split_summary: dict,
                   store: EmbeddingStore, path: Path) -> None:
    lines = [
        "# ProteinLens representation benchmark",
        "",
        f"Model: `{store.meta['model']}` · corpus: {store.meta['n_proteins']} proteins "
        f"(`{store.meta['corpus_sha256_16']}`) · splits: family-grouped "
        f"{split_summary['split_sizes']} · generated by `scripts/run_benchmarks.py`.",
        "",
        "## Probes (test split, leakage-aware family-grouped)",
        "",
    ]
    probes = table[table.axis == "probe"]
    for task in probes["task"].unique():
        sub = probes[probes.task == task].sort_values("macro_f1", ascending=False)
        lines += [f"### {task}", "",
                  "| representation | accuracy | macro F1 | AUROC |",
                  "|---|---|---|---|"]
        for r in sub.itertuples():
            auroc = getattr(r, "auroc", float("nan"))
            if pd.isna(auroc):
                auroc = getattr(r, "auroc_macro_ovr", float("nan"))
            auroc_s = f"{auroc:.3f}" if pd.notna(auroc) else "—"
            lines.append(f"| {r.representation} | {r.accuracy:.3f} | {r.macro_f1:.3f} | {auroc_s} |")
        lines.append("")

    lines += ["## Retrieval (whole corpus)", "",
              "| representation | label | P@1 | P@5 | P@10 |", "|---|---|---|---|---|"]
    ret = table[table.axis == "retrieval"]
    for _, r in ret.iterrows():
        lines.append(f"| {r['representation']} | {r['task']} | {r['precision@1']:.3f} "
                     f"| {r['precision@5']:.3f} | {r['precision@10']:.3f} |")

    lines += ["", "## Clustering agreement (k-means, k=25, vs family)", "",
              "| representation | purity | NMI |", "|---|---|---|"]
    for r in table[table.axis == "clustering"].itertuples():
        lines.append(f"| {r.representation} | {r.purity:.3f} | {r.nmi:.3f} |")

    stab = table[table.axis == "stability"]
    if len(stab):
        lines += ["", "## Stability under one random substitution", "",
                  "| representation | cos mean | cos std | cos p05 |", "|---|---|---|---|"]
        for r in stab.itertuples():
            lines.append(f"| {r.representation} | {r.cosine_mean:.4f} "
                         f"| {r.cosine_std:.4f} | {r.cosine_p05:.4f} |")

    lines += ["", "## Sequence identity vs embedding cosine", "",
              f"- pairs: {sve['n_pairs']}, Pearson r = {sve['pearson_r']:.3f}",
              f"- low-identity (<20%) / high-cosine (>0.9) pairs: {sve['low_identity_high_cosine']}",
              f"- high-identity (>50%) / low-cosine (<0.7) pairs: {sve['high_identity_low_cosine']}",
              "",
              "High-cosine low-identity pairs are *representation-space neighbors*; "
              "no claim of biological convergence is made without evidence.", ""]
    path.write_text("\n".join(lines))
