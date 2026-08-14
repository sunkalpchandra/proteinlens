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


def main() -> int:
    bench = pd.read_csv("reports/benchmark.csv")
    readme = Path("README.md")
    text = readme.read_text()
    if BEGIN not in text or END not in text:
        print("RESULTS markers not found in README.md", file=sys.stderr)
        return 1
    block = f"{BEGIN}\n{build_block(bench)}\n{END}"
    text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), block, text, flags=re.S)
    readme.write_text(text)
    print("README results block updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
