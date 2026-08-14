"""Registry of supported protein language-model checkpoints.

Every checkpoint is served through the exact same interface (``ESM2Encoder`` /
``EmbeddingPipeline``); this module centralizes what differs between them —
identity, dimensionality, and resource expectations — so scripts can validate
requests, size batches, and route artifacts without hard-coding model facts.

Artifact routing: the *serving* store stays at ``data/embeddings`` for the
configured default model; study artifacts for other checkpoints live under
``data/scaling/{slug}`` so checkpoints never overwrite each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    slug: str                # filesystem-safe short name
    family: str              # "esm2" | "prostt5"
    params_m: int            # parameters, millions
    hidden_size: int
    layers: int
    approx_fp32_gb: float    # weights only
    min_ram_gb: int          # practical minimum for CPU/MPS inference
    token_budget: int        # per-forward token budget tuned to the size
    notes: str = ""


REGISTRY: dict[str, ModelSpec] = {
    spec.hf_id: spec
    for spec in [
        # Token budgets are bounded by attention-activation memory (∝ batch·L²),
        # not weights — small models don't get to run huge batches on 8GB hosts.
        ModelSpec("facebook/esm2_t6_8M_UR50D", "esm2_t6_8M", "esm2",
                  8, 320, 6, 0.03, 2, 16384),
        ModelSpec("facebook/esm2_t12_35M_UR50D", "esm2_t12_35M", "esm2",
                  35, 480, 12, 0.14, 4, 16384,
                  "default serving checkpoint"),
        ModelSpec("facebook/esm2_t30_150M_UR50D", "esm2_t30_150M", "esm2",
                  150, 640, 30, 0.60, 8, 8192),
        ModelSpec("facebook/esm2_t33_650M_UR50D", "esm2_t33_650M", "esm2",
                  650, 1280, 33, 2.60, 16, 4096,
                  "needs a 16GB+ machine or GPU; not run on 8GB hosts"),
        ModelSpec("Rostlab/ProstT5", "prostt5", "prostt5",
                  1208, 1024, 24, 4.80, 16, 4096,
                  "structure-aware (3Di-supervised) T5 encoder; run fp16, "
                  "see ml/prost.py"),
    ]
}


def spec_for(model_id: str) -> ModelSpec:
    if model_id not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown model '{model_id}'. Registered: {known}")
    return REGISTRY[model_id]


def study_dir(model_id: str, root: str | Path = "data/scaling") -> Path:
    """Per-checkpoint artifact directory for scaling/baseline studies."""
    return Path(root) / spec_for(model_id).slug


def check_memory(model_id: str, available_gb: float) -> str | None:
    """Return a human-readable warning when the host is likely too small."""
    spec = spec_for(model_id)
    if available_gb < spec.min_ram_gb:
        return (
            f"{model_id} wants ≥{spec.min_ram_gb} GB RAM "
            f"(host has {available_gb:.0f} GB) — expect swapping or OOM. "
            f"Consider a smaller checkpoint or fp16."
        )
    return None
