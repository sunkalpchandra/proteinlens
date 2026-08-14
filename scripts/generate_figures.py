"""Regenerate every benchmark figure from artifacts on disk.

Inputs:  data/index/map_mean.json, reports/benchmark.csv, reports/seq_vs_emb.csv,
         plus one live mutation-landscape computation for the showcase protein.
Outputs: reports/figures/*.png (200 dpi) and *.pdf (vector).

All sampling is seeded; run twice, get identical figures. Style follows a
light-surface scientific palette; categorical hues are assigned in fixed
order, magnitudes use a single blue ramp.

Usage:
    python scripts/generate_figures.py [--skip-landscape]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- palette (light mode reference set) -------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#b9b7ae"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "plens_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8,
    "text.color": INK,
})

FIG_DIR = Path("reports/figures")
ESM_REPS = ["esm_mean", "esm_max", "esm_bos", "esm_attention"]
BASELINES = ["kmer3", "onehot"]
REP_ORDER = BASELINES + ESM_REPS


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.(png|pdf)")


def rep_color(rep: str) -> str:
    return SERIES[0] if rep.startswith("esm_") else OTHER


def load_map(pooling: str = "mean") -> pd.DataFrame:
    payload = json.loads(Path(f"data/index/map_{pooling}.json").read_text())
    return pd.DataFrame(payload["points"])


# --- 1. global embedding map --------------------------------------------------
def fig_embedding_map(points: pd.DataFrame) -> None:
    top = points["family"].value_counts().head(7).index.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 6))
    rest = points[~points["family"].isin(top)]
    ax.scatter(rest.x, rest.y, s=2.5, c=OTHER, alpha=0.35, linewidths=0, rasterized=True)
    for i, family in enumerate(top):
        sub = points[points["family"] == family]
        label = family if len(family) <= 42 else family[:40] + "…"
        ax.scatter(sub.x, sub.y, s=4.5, c=SERIES[i], alpha=0.85, linewidths=0,
                   label=f"{label} ({len(sub)})", rasterized=True)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Protein representation map — ESM-2 mean pooling, PCA→UMAP")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), markerscale=2.2)
    save(fig, "01_embedding_map")


# --- 2. sequence vs embedding similarity ---------------------------------------
def fig_seq_vs_emb() -> None:
    frame = pd.read_csv("reports/seq_vs_emb.csv")
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    same = frame[frame.same_family]
    diff = frame[~frame.same_family]
    ax.scatter(diff.identity, diff.cosine, s=5, c=SERIES[0], alpha=0.25,
               linewidths=0, label=f"different family (n={len(diff)})", rasterized=True)
    ax.scatter(same.identity, same.cosine, s=6, c=SERIES[1], alpha=0.6,
               linewidths=0, label=f"same family (n={len(same)})", rasterized=True)
    ax.axvspan(0, 0.20, ymin=0.9, alpha=0.06, color=SERIES[2])
    r = frame["identity"].corr(frame["cosine"])
    n_conv = len(frame[(frame.identity < 0.2) & (frame.cosine > 0.9)])
    ax.set_xlabel("pairwise sequence identity (global alignment)")
    ax.set_ylabel("embedding cosine similarity (mean pooling)")
    ax.set_title(f"Sequence identity vs representation similarity (r = {r:.2f})")
    ax.grid(True, alpha=0.6)
    ax.legend(loc="lower right")
    ax.annotate(
        f"{n_conv} representation-space neighbor pairs\n(<20% identity, cos > 0.9)",
        xy=(0.02, 0.965), xycoords="axes fraction", fontsize=8, color=INK2, va="top",
    )
    save(fig, "02_seq_vs_embedding")


# --- 3. mutation displacement heatmap -------------------------------------------
def fig_mutation_heatmap(skip: bool) -> None:
    from ml.sequence import CANONICAL_AA

    if skip:
        print("  skipping landscape figure (--skip-landscape)")
        return
    from ml.embeddings import EmbeddingPipeline
    from models.mutation import MutationAnalyzer

    corpus = pd.read_parquet("data/processed/proteins.parquet")
    showcase = corpus[corpus.accession == "P69905"]  # human hemoglobin α
    if showcase.empty:
        showcase = corpus[corpus.length.between(120, 200)].iloc[:1]
    row = showcase.iloc[0]
    seq = row.sequence
    center = 63 if len(seq) >= 80 else len(seq) // 2
    positions = list(range(max(1, center - 5), min(len(seq), center + 6)))

    pipeline = EmbeddingPipeline(cache_path=None)
    analyzer = MutationAnalyzer(pipeline)
    grid = np.full((len(positions), len(CANONICAL_AA)), np.nan)
    for r_i, pos in enumerate(positions):
        landscape = analyzer.landscape(seq, pos, "mean")
        for effect in landscape["effects"]:
            grid[r_i, CANONICAL_AA.index(effect["mutant"])] = effect["displacement"]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    im = ax.imshow(grid, aspect="auto", cmap=BLUES)
    ax.set_xticks(range(len(CANONICAL_AA)), list(CANONICAL_AA))
    ax.set_yticks(range(len(positions)),
                  [f"{seq[p-1]}{p}" for p in positions])
    ax.set_xlabel("substituted amino acid")
    ax.set_ylabel("position (wild-type)")
    ax.set_title(
        f"Representation displacement ‖Δz‖ under substitutions — "
        f"{row.protein_name} ({row.accession})"
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015)
    cbar.set_label("‖Δz‖₂ (mean pooling)", fontsize=8)
    cbar.outline.set_visible(False)
    save(fig, "03_mutation_heatmap")


# --- 4/5/6. benchmark-table figures ----------------------------------------------
def fig_retrieval(bench: pd.DataFrame) -> None:
    ret = bench[(bench.axis == "retrieval") & (bench.task == "same_pfam_primary")]
    ret = ret.set_index("representation").reindex(REP_ORDER).dropna(how="all")
    ks = ["precision@1", "precision@5", "precision@10"]
    x = np.arange(len(ret))
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    shades = ["#9ec5f4", "#3987e5", "#0d366b"]
    for i, k in enumerate(ks):
        vals = ret[k].to_numpy(dtype=float)
        bars = ax.bar(x + (i - 1) * width, vals, width * 0.92, color=shades[i], label=f"P@{k.split('@')[1]}")
        ax.bar_label(bars, fmt="%.2f", fontsize=6.5, color=INK2, padding=1)
    ax.set_xticks(x, ret.index)
    ax.set_ylabel("precision@k (same Pfam domain)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Nearest-neighbor retrieval by representation")
    ax.grid(True, axis="y", alpha=0.6)
    ax.legend(ncols=3, loc="upper left")
    save(fig, "04_retrieval_precision")


def fig_probes(bench: pd.DataFrame) -> None:
    probes = bench[bench.axis == "probe"]
    tasks = ["enzyme_vs_nonenzyme", "ec_class", "subcellular_localization"]
    titles = ["enzyme vs non-enzyme", "EC top class", "subcellular localization"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharey=False)
    for ax, task, title in zip(axes, tasks, titles):
        sub = probes[probes.task == task].set_index("representation").reindex(REP_ORDER)
        vals = sub["macro_f1"].to_numpy(dtype=float)
        colors = [rep_color(r) for r in sub.index]
        bars = ax.barh(np.arange(len(sub)), vals, color=colors, height=0.62)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, color=INK2, padding=2)
        ax.set_yticks(np.arange(len(sub)), sub.index)
        ax.invert_yaxis()
        ax.set_xlim(0, max(1.0, np.nanmax(vals) * 1.15))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("macro F1 (test)")
        ax.grid(True, axis="x", alpha=0.6)
    fig.suptitle("Linear probes on frozen representations — family-grouped splits", y=1.04)
    save(fig, "05_probe_results")


def fig_pooling_comparison(bench: pd.DataFrame) -> None:
    metrics = []
    probes = bench[bench.axis == "probe"]
    for rep in ESM_REPS:
        probe_f1 = probes[probes.representation == rep]["macro_f1"].astype(float).mean()
        ret = bench[(bench.axis == "retrieval") & (bench.task == "same_pfam_primary")
                    & (bench.representation == rep)]["precision@10"].astype(float)
        clus = bench[(bench.axis == "clustering") & (bench.representation == rep)]["nmi"].astype(float)
        metrics.append({
            "rep": rep.removeprefix("esm_"),
            "probe mean F1": probe_f1,
            "retrieval P@10": float(ret.iloc[0]) if len(ret) else np.nan,
            "cluster NMI": float(clus.iloc[0]) if len(clus) else np.nan,
        })
    frame = pd.DataFrame(metrics).set_index("rep")
    x = np.arange(len(frame))
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for i, col in enumerate(frame.columns):
        bars = ax.bar(x + (i - 1) * width, frame[col], width * 0.92,
                      color=["#9ec5f4", "#3987e5", "#0d366b"][i], label=col)
        ax.bar_label(bars, fmt="%.2f", fontsize=6.5, color=INK2, padding=1)
    ax.set_xticks(x, frame.index)
    ax.set_ylim(0, 1.05)
    ax.set_title("Pooling strategies compared (ESM-2 35M)")
    ax.grid(True, axis="y", alpha=0.6)
    ax.legend(ncols=3, loc="upper left", fontsize=7.5)
    save(fig, "06_pooling_comparison")


# --- 7/8. clustering + outliers ---------------------------------------------------
def fig_clusters(points: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    clusters = sorted(points["cluster"].unique())
    rng = np.random.default_rng(42)
    for cluster in clusters:
        sub = points[points.cluster == cluster]
        color = SERIES[cluster % len(SERIES)]
        ax.scatter(sub.x, sub.y, s=3, c=color, alpha=0.6, linewidths=0, rasterized=True)
        cx, cy = sub.x.median(), sub.y.median()
        ax.annotate(str(cluster), (cx, cy), fontsize=7.5, color=INK,
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.18", fc=SURFACE, ec=MUTED, lw=0.6, alpha=0.9))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"k-means clusters (k={len(clusters)}) on ESM-2 mean-pooled embeddings")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    save(fig, "07_clusters")
    _ = rng  # reserved for future jittered labels; keeps seed documented


def fig_outliers(points: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    order = points.sort_values("outlier")
    sc = ax.scatter(order.x, order.y, s=3.5, c=order.outlier, cmap=BLUES,
                    alpha=0.8, linewidths=0, rasterized=True)
    top = points.nlargest(8, "outlier")
    for row in top.itertuples():
        ax.annotate(row.id, (row.x, row.y), fontsize=6.5, color=INK2,
                    xytext=(4, 4), textcoords="offset points")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.015)
    cbar.set_label("representation outlier score (k-NN distance percentile)", fontsize=8)
    cbar.outline.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Representation-space isolation (not a biological anomaly claim)")
    save(fig, "08_outliers")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-landscape", action="store_true",
                        help="skip the figure that needs live ESM inference")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    points = load_map("mean")
    bench = pd.read_csv("reports/benchmark.csv")

    print("Generating figures…")
    fig_embedding_map(points)
    fig_seq_vs_emb()
    fig_retrieval(bench)
    fig_probes(bench)
    fig_pooling_comparison(bench)
    fig_clusters(points)
    fig_outliers(points)
    fig_mutation_heatmap(args.skip_landscape)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
