"""Figures for the extended benchmark (checkpoint scaling, objectives, ProstT5).

Reads reports/extended_benchmark.csv; writes:
    09_checkpoint_scaling.(png|pdf)   metric vs parameter count, log-x
    10_representation_comparison.(png|pdf)  grouped horizontal bars, all reps

Usage:
    python scripts/generate_extended_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_figures import INK2, MUTED, OTHER, SERIES, save  # noqa: E402

FIG_DIR = Path("reports/figures")


def fig_scaling(table: pd.DataFrame) -> None:
    scaling = table[(table.group == "esm2-scaling") & (table.pooling == "mean")]
    scaling = scaling.sort_values("params_m")
    if len(scaling) < 2:
        print("  skipping scaling figure (need ≥2 checkpoints)")
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    x = scaling["params_m"].to_numpy(dtype=float)
    for metric, label, color in [
        ("probe_f1_mean", "probe macro-F1 (mean of 3 tasks)", SERIES[0]),
        ("p_at_10", "retrieval P@10 (Pfam)", SERIES[2]),
        ("nmi", "cluster NMI (family)", SERIES[4]),
    ]:
        y = scaling[metric].to_numpy(dtype=float)
        ax.plot(x, y, "-o", color=color, markersize=5, linewidth=1.5, label=label)
        for xi, yi in zip(x, y, strict=True):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{int(v)}M" for v in x])
    ax.set_xlabel("ESM-2 parameters")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.6)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Checkpoint scale vs representation quality (mean pooling, 3k subset)")
    save(fig, "09_checkpoint_scaling")


def fig_comparison(table: pd.DataFrame) -> None:
    table = table.sort_values(["group", "params_m", "representation"])
    group_color = {"baseline": OTHER, "esm2-scaling": SERIES[0],
                   "pooling-objective": SERIES[2], "structure-aware": SERIES[1]}
    fig, ax = plt.subplots(figsize=(6.8, 0.34 * len(table) + 1.6))
    y = np.arange(len(table))
    values = table["probe_f1_mean"].to_numpy(dtype=float)
    colors = [group_color[g] for g in table["group"]]
    bars = ax.barh(y, values, color=colors, height=0.62)
    ax.bar_label(bars, fmt="%.3f", fontsize=7, color=INK2, padding=2)
    ax.set_yticks(y, table["representation"])
    ax.invert_yaxis()
    ax.set_xlabel("probe macro-F1 (mean of 3 tasks, shared subset)")
    ax.set_xlim(0, max(1.0, values.max() * 1.15))
    ax.grid(True, axis="x", alpha=0.6)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in group_color.values()]
    ax.legend(handles, group_color.keys(), loc="lower right", fontsize=7.5)
    ax.set_title("All representations on the shared evaluation subset")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED)
    save(fig, "10_representation_comparison")


def main() -> int:
    table = pd.read_csv("reports/extended_benchmark.csv")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating extended figures…")
    fig_scaling(table)
    fig_comparison(table)
    fig_dms()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def fig_dms(path: Path = Path("reports/dms_validation.csv")) -> None:
    """Figure 11: measured variant effect vs model statistics."""
    if not path.exists():
        print("  skipping DMS figure (no reports/dms_validation.csv)")
        return
    frame = pd.read_csv(path)
    if "displacement" not in frame:
        print("  skipping DMS figure (LLR-only run)")
        return
    from scipy.stats import spearmanr

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    for ax, (col, label, flip) in zip(axes, [
        ("llr", "LM log-likelihood ratio", False),
        ("displacement", "embedding displacement ‖Δz‖", True),
    ], strict=True):
        x = frame[col]
        rho = spearmanr(-x if flip else x, frame["score"]).statistic
        ax.scatter(x, frame["score"], s=4, c=SERIES[0], alpha=0.25,
                   linewidths=0, rasterized=True)
        ax.set_xlabel(label)
        ax.grid(True, alpha=0.6)
        sign_note = "−x vs score" if flip else "x vs score"
        ax.set_title(f"Spearman ρ = {rho:+.3f} ({sign_note})", fontsize=9)
    axes[0].set_ylabel("measured complementation score")
    fig.suptitle(
        "CALM1 DMS-TileSeq (2,525 variants) vs frozen ESM-2 35M statistics",
        y=1.02,
    )
    save(fig, "11_dms_validation")
