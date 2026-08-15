"""Inject headline benchmark numbers into README.md between the RESULTS markers.

Reads reports/benchmark.csv (never hand-entered numbers) and rewrites the
block so the README always reflects the last real run.

Usage:
    python scripts/update_readme_results.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

BEGIN = "<!-- RESULTS:BEGIN — filled by scripts/run_benchmarks.py output; see reports/benchmark.md -->"
END = "<!-- RESULTS:END -->"

REP_LABELS = {
    "kmer3": "3-mer frequencies (baseline)",
    "onehot": "one-hot composition (baseline)",
    "esm_mean": "ESM-2 mean pooling",
    "esm_max": "ESM-2 max pooling",
    "esm_bos": "ESM-2 BOS token",
    "esm_attention": "ESM-2 attention pooling (learned)",
}
ORDER = ["kmer3", "onehot", "esm_mean", "esm_max", "esm_bos", "esm_attention"]


def build_block(bench: pd.DataFrame) -> str:
    lines = [
        "Headline numbers from the last full run (`reports/benchmark.csv`; regenerate with",
        "`python scripts/run_benchmarks.py`). Probes use leakage-aware family-grouped splits.",
        "",
        "| Representation | Probe macro-F1 (mean of 3 tasks) | Retrieval P@10 (Pfam) | Cluster NMI (family) | Stability cos (1 sub) |",
        "|---|---|---|---|---|",
    ]
    probes = bench[bench.axis == "probe"]
    for rep in ORDER:
        f1 = probes[probes.representation == rep]["macro_f1"].astype(float).mean()
        ret = bench[(bench.axis == "retrieval") & (bench.representation == rep)
                    & (bench.task == "same_pfam_primary")]["precision@10"].astype(float)
        nmi = bench[(bench.axis == "clustering") & (bench.representation == rep)]["nmi"].astype(float)
        stab = bench[(bench.axis == "stability") & (bench.representation == rep)]["cosine_mean"].astype(float)

        def fmt(series_or_val, digits=3):
            try:
                v = float(series_or_val.iloc[0]) if hasattr(series_or_val, "iloc") else float(series_or_val)
                return f"{v:.{digits}f}"
            except (TypeError, ValueError, IndexError):
                return "—"

        lines.append(
            f"| {REP_LABELS[rep]} | {fmt(f1)} | {fmt(ret)} | {fmt(nmi)} | {fmt(stab, 4)} |"
        )
    lines += ["", "Full per-task tables: [`reports/benchmark.md`](reports/benchmark.md)."]
    return "\n".join(lines)


def replace_block(text: str, begin: str, end: str, body: str) -> str:
    """Swap the content between two markers (markers included in the result)."""
    block = f"{begin}\n{body}\n{end}"
    return re.sub(re.escape(begin) + r".*?" + re.escape(end), block, text, flags=re.S)


def main() -> int:
    readme = Path("README.md")
    text = readme.read_text()
    if BEGIN not in text or END not in text:
        print("RESULTS markers not found in README.md", file=sys.stderr)
        return 1

    text = replace_block(text, BEGIN, END, build_block(pd.read_csv("reports/benchmark.csv")))
    print("README results block updated.")

    extended = Path("reports/extended_benchmark.csv")
    if EXT_BEGIN in text and extended.exists():
        text = replace_block(text, EXT_BEGIN, EXT_END, build_extended_block(pd.read_csv(extended)))
        print("README extended block updated.")

    readme.write_text(text)
    return 0



EXT_BEGIN = "<!-- EXTENDED:BEGIN -->"
EXT_END = "<!-- EXTENDED:END -->"


def build_extended_block(table: pd.DataFrame) -> str:
    lines = [
        "Headline extended numbers (subset probes use the same leakage-aware splits):",
        "",
        "| representation | group | params (M) | probe F1 (mean) | P@1 (Pfam) | NMI |",
        "|---|---|---|---|---|---|",
    ]
    ordered = table.sort_values(["group", "params_m", "representation"])
    for r in ordered.itertuples():
        lines.append(
            f"| {r.representation} | {r.group} | {r.params_m} "
            f"| {r.probe_f1_mean:.3f} | {r.p_at_1:.3f} | {r.nmi:.3f} |"
        )
    lines += ["", "Full tables: [`reports/extended_benchmark.md`](reports/extended_benchmark.md)."]
    return "\n".join(lines)



if __name__ == "__main__":
    sys.exit(main())
