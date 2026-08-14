"""Mutation analysis in representation space.

Given a wild-type sequence and a point substitution, re-encodes the mutant with
the frozen language model and measures how the representation moves:

  Δz = z_mut − z_wt          (protein level, chosen pooling)
  per-residue ‖Δh_i‖         (residue level, aligned positions)

These are *representation-space perturbations*. They quantify how the model's
embedding responds to a substitution — they are NOT fitness, stability, or
pathogenicity predictions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ml.embeddings import EmbeddingPipeline
from ml.sequence import CANONICAL_AA, SequenceValidationError, validate_sequence

MUTATION_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")


@dataclass(frozen=True)
class Mutation:
    wildtype: str   # single-letter AA expected at the position
    position: int   # 1-based
    mutant: str

    @property
    def label(self) -> str:
        return f"{self.wildtype}{self.position}{self.mutant}"


def parse_mutation(text: str) -> Mutation:
    match = MUTATION_PATTERN.match(text.strip().upper())
    if not match:
        raise SequenceValidationError(
            f"Cannot parse mutation {text!r}; expected e.g. 'H63Y' (WT, 1-based position, mutant)."
        )
    wt, pos, mut = match.group(1), int(match.group(2)), match.group(3)
    for aa, role in ((wt, "wild-type"), (mut, "mutant")):
        if aa not in CANONICAL_AA:
            raise SequenceValidationError(f"{role} residue {aa!r} is not a canonical amino acid.")
    return Mutation(wt, pos, mut)


def apply_mutation(sequence: str, mutation: Mutation) -> str:
    if not 1 <= mutation.position <= len(sequence):
        raise SequenceValidationError(
            f"Position {mutation.position} outside sequence of length {len(sequence)}."
        )
    found = sequence[mutation.position - 1]
    if found != mutation.wildtype:
        raise SequenceValidationError(
            f"Wild-type mismatch at position {mutation.position}: sequence has "
            f"{found}, mutation says {mutation.wildtype}."
        )
    if found == mutation.mutant:
        raise SequenceValidationError(f"{mutation.label} is not a substitution.")
    return sequence[: mutation.position - 1] + mutation.mutant + sequence[mutation.position:]


@dataclass
class MutationEffect:
    mutation: str
    pooling: str
    displacement: float          # ‖Δz‖₂
    relative_displacement: float  # ‖Δz‖₂ / ‖z_wt‖₂
    cosine_similarity: float     # cos(z_wt, z_mut)
    local_delta: float           # mean ‖Δh_i‖ within ±window of the site
    global_residue_delta: float  # mean ‖Δh_i‖ over all residues
    per_residue_delta: list[float]
    top_dimensions: list[dict]   # largest |Δz_d| entries
    mutant_embedding: list[float] | None = None
    wildtype_embedding: list[float] | None = None


class MutationAnalyzer:
    def __init__(self, pipeline: EmbeddingPipeline, window: int = 8) -> None:
        self.pipeline = pipeline
        self.window = window

    def _residue_deltas(self, wt_seq: str, mut_seq: str) -> np.ndarray:
        wt = self.pipeline.encode_residues(wt_seq).residue_embeddings.numpy()
        mut = self.pipeline.encode_residues(mut_seq).residue_embeddings.numpy()
        return np.linalg.norm(mut - wt, axis=1)

    def analyze(
        self,
        sequence: str,
        mutation: Mutation | str,
        pooling: str = "mean",
        include_embeddings: bool = False,
    ) -> MutationEffect:
        seq = validate_sequence(sequence)
        mut = parse_mutation(mutation) if isinstance(mutation, str) else mutation
        mut_seq = apply_mutation(seq, mut)

        z_wt = self.pipeline.embed(seq, pooling).embedding
        z_mut = self.pipeline.embed(mut_seq, pooling).embedding
        delta = z_mut - z_wt

        per_residue = self._residue_deltas(seq, mut_seq)
        lo = max(0, mut.position - 1 - self.window)
        hi = min(len(seq), mut.position + self.window)

        top = np.argsort(-np.abs(delta))[:10]
        displacement = float(np.linalg.norm(delta))
        return MutationEffect(
            mutation=mut.label,
            pooling=pooling,
            displacement=displacement,
            relative_displacement=displacement / max(float(np.linalg.norm(z_wt)), 1e-12),
            cosine_similarity=float(
                np.dot(z_wt, z_mut)
                / max(float(np.linalg.norm(z_wt) * np.linalg.norm(z_mut)), 1e-12)
            ),
            local_delta=float(per_residue[lo:hi].mean()),
            global_residue_delta=float(per_residue.mean()),
            per_residue_delta=[round(float(x), 5) for x in per_residue],
            top_dimensions=[
                {"dim": int(d), "delta": round(float(delta[d]), 5)} for d in top
            ],
            mutant_embedding=z_mut.tolist() if include_embeddings else None,
            wildtype_embedding=z_wt.tolist() if include_embeddings else None,
        )

    def landscape(
        self, sequence: str, position: int, pooling: str = "mean"
    ) -> dict:
        """Representation displacement for all 19 substitutions at one site.

        Mutants are encoded in a single batched pass; protein-level metrics use
        the requested pooling, and residue-level deltas use the mutation window.
        """
        seq = validate_sequence(sequence)
        if not 1 <= position <= len(seq):
            raise SequenceValidationError(
                f"Position {position} outside sequence of length {len(seq)}."
            )
        wildtype_aa = seq[position - 1]
        alternatives = [aa for aa in CANONICAL_AA if aa != wildtype_aa]
        mutants = {
            aa: apply_mutation(seq, Mutation(wildtype_aa, position, aa)) for aa in alternatives
        }

        z_wt = self.pipeline.embed(seq, pooling).embedding
        wt_residues = self.pipeline.encode_residues(seq).residue_embeddings.numpy()
        lo = max(0, position - 1 - self.window)
        hi = min(len(seq), position + self.window)

        encoded = self.pipeline.encoder.encode_batch(list(mutants.values()))
        effects = []
        for aa, enc in zip(mutants, encoded, strict=True):
            pooled, _ = self.pipeline.pooler.pool(
                enc.residue_embeddings, enc.bos_embedding, pooling
            )
            z_mut = pooled.numpy().astype(np.float32)
            delta = z_mut - z_wt
            per_res = np.linalg.norm(enc.residue_embeddings.numpy() - wt_residues, axis=1)
            effects.append(
                {
                    "mutant": aa,
                    "mutation": f"{wildtype_aa}{position}{aa}",
                    "displacement": float(np.linalg.norm(delta)),
                    "cosine_similarity": float(
                        np.dot(z_wt, z_mut)
                        / max(float(np.linalg.norm(z_wt) * np.linalg.norm(z_mut)), 1e-12)
                    ),
                    "local_delta": float(per_res[lo:hi].mean()),
                }
            )

        displacements = [e["displacement"] for e in effects]
        return {
            "position": position,
            "wildtype": wildtype_aa,
            "pooling": pooling,
            "effects": effects,
            "max_displacement": effects[int(np.argmax(displacements))]["mutation"],
            "min_displacement": effects[int(np.argmin(displacements))]["mutation"],
        }
