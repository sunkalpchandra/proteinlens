"""Validate model mutation statistics against measured DMS variant effects.

For every assayed single substitution, computes two model statistics —

    llr           wt-marginal log-likelihood ratio (masked-LM head; one forward)
    displacement  ‖z_mut − z_wt‖ (mean pooling; one forward per variant)

— and reports Spearman correlations against the experimental scores. This is
the honest test the README's future-work list asked for: does representation
displacement track measured functional effect, and how does it compare to the
field-standard likelihood score? The result is assay- and protein-specific;
nothing here turns either statistic into a fitness predictor.

Writes reports/dms_validation.{csv,md} and figure 11.

Usage:
    python scripts/run_dms_validation.py [--accession P0DP23] [--skip-displacement]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.tracking import log_experiment  # noqa: E402


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accession", default="P0DP23")
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/proteins.parquet"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--skip-displacement", action="store_true",
                        help="LLR only (skips ~2.5k embedding forwards)")
    args = parser.parse_args()

    dms_path = Path(f"data/processed/dms_{args.accession}.parquet")
    if not dms_path.exists():
        raise SystemExit(f"No DMS data at {dms_path}. Run scripts/download_dms.py.")
    dms = pd.read_parquet(dms_path)
    manifest = json.loads(
        Path(f"data/processed/dms_{args.accession}_manifest.json").read_text()
    )
    sequence = pd.read_parquet(args.corpus).set_index("accession").loc[
        args.accession, "sequence"
    ]
    print(f"{len(dms)} assayed variants for {args.accession} "
          f"({manifest['title']})")

    # --- LLR: one forward pass covers every variant --------------------------
    from ml.sequence import CANONICAL_AA
    from models.scoring import MaskedLMScorer

    scorer = MaskedLMScorer()
    log_probs = scorer.log_probs(sequence).numpy()  # [L, 20]
    aa_index = {aa: i for i, aa in enumerate(CANONICAL_AA)}
    dms["llr"] = [
        float(log_probs[row.position - 1, aa_index[row.mut]]
              - log_probs[row.position - 1, aa_index[row.wt]])
        for row in dms.itertuples()
    ]

    # --- Displacement: one embedding forward per variant -----------------------
    if not args.skip_displacement:
        from ml.embeddings import EmbeddingPipeline, cosine_similarity

        pipeline = EmbeddingPipeline(cache_path=None)
        z_wt = pipeline.embed(sequence, "mean").embedding
        mutants = [
            sequence[: row.position - 1] + row.mut + sequence[row.position :]
            for row in dms.itertuples()
        ]
        t0 = time.time()
        vectors = pipeline.embed_batch(mutants, "mean")
        print(f"Embedded {len(mutants)} variants in {(time.time()-t0)/60:.1f} min")
        dms["displacement"] = [float(np.linalg.norm(v - z_wt)) for v in vectors]
        dms["cosine"] = [cosine_similarity(z_wt, v) for v in vectors]

    args.reports.mkdir(exist_ok=True)
    dms.to_csv(args.reports / "dms_validation.csv", index=False)

    metrics = {"spearman_llr": spearman(dms["llr"], dms["score"])}
    if "displacement" in dms:
        metrics["spearman_neg_displacement"] = spearman(-dms["displacement"], dms["score"])
        metrics["spearman_cosine"] = spearman(dms["cosine"], dms["score"])
    for name, value in metrics.items():
        print(f"  {name}: {value:+.3f}")

    lines = [
        "# DMS validation — model statistics vs measured variant effects",
        "",
        f"Assay: {manifest['title']} ({manifest['urn']}), {len(dms)} single "
        f"substitutions on {args.accession} (complementation score; ≈1 = "
        "wild-type-like function). Model: `facebook/esm2_t12_35M_UR50D`, frozen.",
        "",
        "| model statistic | Spearman ρ vs assay score |",
        "|---|---|",
        f"| LM log-likelihood ratio (wt-marginal) | {metrics['spearman_llr']:+.3f} |",
    ]
    if "spearman_neg_displacement" in metrics:
        lines += [
            f"| −‖Δz‖ embedding displacement (mean pooling) | {metrics['spearman_neg_displacement']:+.3f} |",
            f"| cos(z_wt, z_mut) | {metrics['spearman_cosine']:+.3f} |",
        ]
    lines += [
        "",
        "Reading: the likelihood ratio is the field-standard zero-shot variant "
        "score; displacement measures representation movement. A positive ρ "
        "means the statistic tracks the assay (higher = more functional). "
        "Correlations are specific to this protein and assay; neither "
        "statistic is a fitness predictor.",
        "",
    ]
    (args.reports / "dms_validation.md").write_text("\n".join(lines))
    print(f"Wrote {args.reports / 'dms_validation.md'}")

    log_experiment("dms_validation",
                   config={"accession": args.accession, "urn": manifest["urn"],
                           "n_variants": len(dms)},
                   metrics=metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
