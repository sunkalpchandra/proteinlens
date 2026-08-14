"""Extended representation benchmark on the shared evaluation subset.

Compares, under identical probes/retrieval/clustering on the same ~3k
proteins:

    ESM-2 scaling:    8M vs 35M vs 150M (mean/max/bos each)
    Pooling training: 35M attention pooler, CE vs SupCon objectives
    Structure-aware:  ProstT5 encoder (mean), when embedded
    Baselines:        3-mer frequencies, one-hot composition

Candidates whose artifacts are missing are skipped with a notice — the table
never silently mixes stale rows. Writes reports/extended_benchmark.csv|md.

Usage:
    python scripts/run_extended_benchmarks.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.embeddings import EmbeddingStore  # noqa: E402
from ml.evaluation import clustering_agreement, retrieval_precision_at_k  # noqa: E402
from ml.probes import ProbeTask, run_probe  # noqa: E402
from ml.sequence import kmer_features, onehot_mean_features  # noqa: E402
from ml.splitting import load_splits  # noqa: E402
from ml.tracking import log_experiment  # noqa: E402

# (display name, family, params_m, loader) — loader returns dict pooling→matrix
CANDIDATES = [
    ("esm2-8M", "esm2-scaling", 8, "data/scaling/esm2_t6_8M"),
    ("esm2-35M", "esm2-scaling", 35, "store"),
    ("esm2-150M", "esm2-scaling", 150, "data/scaling/esm2_t30_150M"),
    ("prostt5", "structure-aware", 1208, "data/scaling/prostt5"),
]


def subset_probe_tasks(df: pd.DataFrame) -> list[ProbeTask]:
    """Probe tasks with class thresholds scaled to the subset size."""
    enzyme = df["is_enzyme"].map({True: "enzyme", False: "non-enzyme"})
    ec = df["ec_class"].where(df["is_enzyme"])
    return [
        ProbeTask("enzyme_vs_nonenzyme", enzyme, min_class_count=30),
        ProbeTask("ec_class", ec, min_class_count=12),
        ProbeTask("subcellular_localization", df["localization"], min_class_count=25),
    ]


def load_candidates(subset: list[str], store: EmbeddingStore) -> dict[str, dict]:
    """Returns name → {family, params_m, pooling, matrix} per available rep."""
    reps: dict[str, dict] = {}
    row_of = {acc: i for i, acc in enumerate(store.accessions)}
    subset_rows = np.array([row_of[a] for a in subset])

    for name, family, params_m, source in CANDIDATES:
        if source == "store":
            for pooling in store.poolings:
                reps[f"{name}-{pooling}"] = {
                    "family": family if pooling in ("mean", "max", "bos") else "pooling-objective",
                    "params_m": params_m, "pooling": pooling,
                    "matrix": np.asarray(store.matrix(pooling))[subset_rows],
                }
            continue
        meta_path = Path(source) / "meta.json"
        if not meta_path.exists():
            print(f"  skipping {name}: no artifacts at {source}")
            continue
        for pooling in json.loads(meta_path.read_text())["poolings"]:
            matrix_path = Path(source) / f"subset_{pooling}.npy"
            if matrix_path.exists():
                reps[f"{name}-{pooling}"] = {
                    "family": family, "params_m": params_m,
                    "pooling": pooling, "matrix": np.load(matrix_path),
                }

    supcon_path = Path("data/scaling/esm2_t12_35M/subset_attention_supcon.npy")
    if supcon_path.exists():
        reps["esm2-35M-attention-supcon"] = {
            "family": "pooling-objective", "params_m": 35,
            "pooling": "attention", "matrix": np.load(supcon_path),
        }
    else:
        print("  skipping esm2-35M-attention-supcon: not embedded yet")
    return reps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=Path("data/processed/eval_subset.json"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subset = json.loads(args.subset.read_text())["accessions"]
    corpus = pd.read_parquet("data/processed/proteins.parquet").set_index("accession")
    df = corpus.loc[subset].reset_index()
    split_map, _ = load_splits("data/processed/splits.json")
    splits = df["accession"].map(split_map)
    store = EmbeddingStore("data/embeddings")

    reps = load_candidates(subset, store)
    print(f"Baseline features on {len(df)} proteins…")
    reps["kmer3"] = {"family": "baseline", "params_m": 0, "pooling": "-",
                     "matrix": np.stack([kmer_features(s) for s in df["sequence"]])}
    reps["onehot"] = {"family": "baseline", "params_m": 0, "pooling": "-",
                      "matrix": np.stack([onehot_mean_features(s) for s in df["sequence"]])}

    tasks = subset_probe_tasks(df)
    rows = []
    for name, rep in sorted(reps.items()):
        matrix = rep["matrix"]
        probe_f1 = {}
        for task in tasks:
            metrics = run_probe(matrix, task, splits, seed=args.seed)
            probe_f1[task.name] = metrics["macro_f1"]
        retrieval = retrieval_precision_at_k(matrix, df["pfam_primary"])
        cluster = clustering_agreement(matrix, df["family"], n_clusters=15, seed=args.seed)
        rows.append({
            "representation": name, "group": rep["family"],
            "params_m": rep["params_m"], "pooling": rep["pooling"],
            "dim": matrix.shape[1],
            **{f"f1_{k}": round(v, 4) for k, v in probe_f1.items()},
            "probe_f1_mean": round(float(np.mean(list(probe_f1.values()))), 4),
            "p_at_1": round(retrieval["precision@1"], 4),
            "p_at_10": round(retrieval["precision@10"], 4),
            "nmi": round(cluster["nmi"], 4),
        })
        print(f"  {name:<28} F1 {rows[-1]['probe_f1_mean']:.3f}  "
              f"P@1 {rows[-1]['p_at_1']:.3f}  NMI {rows[-1]['nmi']:.3f}")

    table = pd.DataFrame(rows).sort_values(["group", "params_m", "representation"])
    args.reports.mkdir(exist_ok=True)
    table.to_csv(args.reports / "extended_benchmark.csv", index=False)

    lines = [
        "# Extended benchmark — checkpoint scale, pooling objective, structure-aware baseline",
        "",
        f"Shared evaluation subset: {len(df)} proteins; probes use the corpus's "
        "leakage-aware splits restricted to the subset; retrieval label = primary "
        "Pfam domain; clustering = k-means(15) NMI vs family.",
        "",
        "| representation | group | params (M) | dim | probe F1 (mean) | P@1 | P@10 | NMI |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        lines.append(f"| {r.representation} | {r.group} | {r.params_m} | {r.dim} "
                     f"| {r.probe_f1_mean:.3f} | {r.p_at_1:.3f} | {r.p_at_10:.3f} "
                     f"| {r.nmi:.3f} |")
    lines += ["", "ProstT5 carries structure supervision (3Di translation training) "
              "and ~35× the parameters of ESM-2 35M — treat its rows as a "
              "reference point, not a like-for-like pooling comparison.", ""]
    (args.reports / "extended_benchmark.md").write_text("\n".join(lines))
    print(f"Wrote {args.reports / 'extended_benchmark.csv'} and .md")

    log_experiment("extended_benchmark",
                   config={"subset": len(df), "seed": args.seed,
                           "candidates": sorted(reps)},
                   metrics={"rows": json.loads(table.to_json(orient="records"))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
