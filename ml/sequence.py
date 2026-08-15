"""Sequence-level utilities: validation, amino-acid vocabulary, baseline featurizers.

Everything here is model-free. The 20 canonical amino acids are the only accepted
alphabet; sequences containing ambiguity codes (B, Z, X, J) or non-canonical
residues (U = selenocysteine, O = pyrrolysine) are rejected at ingest so that
every downstream component can assume a clean alphabet.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = frozenset(CANONICAL_AA)
AA_TO_INDEX = {aa: i for i, aa in enumerate(CANONICAL_AA)}

# Physicochemical categories used for residue rendering and coarse analysis.
AA_CATEGORIES: dict[str, str] = {
    **dict.fromkeys("AVLIM", "hydrophobic"),
    **dict.fromkeys("FWY", "aromatic"),
    **dict.fromkeys("STNQ", "polar"),
    **dict.fromkeys("DE", "negative"),
    **dict.fromkeys("KRH", "positive"),
    **dict.fromkeys("CGP", "special"),
}

EC_CLASS_NAMES = {
    "1": "Oxidoreductase",
    "2": "Transferase",
    "3": "Hydrolase",
    "4": "Lyase",
    "5": "Isomerase",
    "6": "Ligase",
    "7": "Translocase",
}


class SequenceValidationError(ValueError):
    """Raised when a protein sequence fails validation."""


def clean_sequence(sequence: str) -> str:
    """Uppercase and strip whitespace/asterisks (common FASTA artifacts)."""
    return "".join(sequence.split()).upper().rstrip("*")


def validate_sequence(sequence: str, min_length: int = 10, max_length: int = 2048) -> str:
    """Return the cleaned sequence or raise ``SequenceValidationError``.

    Rejects empty input, out-of-range lengths, and any character outside the
    20 canonical amino acids.
    """
    seq = clean_sequence(sequence)
    if not seq:
        raise SequenceValidationError("Sequence is empty.")
    if len(seq) < min_length:
        raise SequenceValidationError(f"Sequence is too short ({len(seq)} aa; minimum {min_length}).")
    if len(seq) > max_length:
        raise SequenceValidationError(f"Sequence is too long ({len(seq)} aa; maximum {max_length}).")
    invalid = sorted(set(seq) - AA_SET)
    if invalid:
        raise SequenceValidationError(
            f"Sequence contains non-canonical symbols: {', '.join(invalid)}. "
            f"Allowed alphabet: {CANONICAL_AA}."
        )
    return seq


def is_valid_sequence(sequence: str, min_length: int = 10, max_length: int = 2048) -> bool:
    try:
        validate_sequence(sequence, min_length, max_length)
        return True
    except SequenceValidationError:
        return False


# ---------------------------------------------------------------------------
# Baseline featurizers (used by the benchmark suite, never by the main model)
# ---------------------------------------------------------------------------

def kmer_vocabulary(k: int) -> list[str]:
    return ["".join(p) for p in itertools.product(CANONICAL_AA, repeat=k)]


@functools.lru_cache(maxsize=4)
def _kmer_index(k: int) -> dict[str, int]:
    return {kmer: i for i, kmer in enumerate(kmer_vocabulary(k))}


def kmer_features(sequence: str, k: int = 3, normalize: bool = True) -> np.ndarray:
    """Frequency vector over all 20^k k-mers (8000 dims for k=3)."""
    index = _kmer_index(k)
    vec = np.zeros(len(index), dtype=np.float32)
    for i in range(len(sequence) - k + 1):
        idx = index.get(sequence[i : i + k])
        if idx is not None:
            vec[idx] += 1.0
    if normalize and vec.sum() > 0:
        vec /= vec.sum()
    return vec


def onehot_mean_features(sequence: str) -> np.ndarray:
    """Position-averaged one-hot composition (20 dims) — the weakest baseline."""
    vec = np.zeros(len(CANONICAL_AA), dtype=np.float32)
    for aa in sequence:
        idx = AA_TO_INDEX.get(aa)
        if idx is not None:
            vec[idx] += 1.0
    if len(sequence) > 0:
        vec /= len(sequence)
    return vec


@functools.lru_cache(maxsize=1)
def _blosum62_aligner():
    """Configured global BLOSUM62 aligner, built once — the matrix load reads
    a data file and per-call construction was pure overhead on /compare."""
    from Bio import Align
    from Bio.Align import substitution_matrices

    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.mode = "global"
    return aligner


def sequence_identity(a: str, b: str) -> float:
    """Global alignment identity via Biopython PairwiseAligner (BLOSUM62).

    Identity = matches / alignment length. This is a lightweight metric for
    analysis plots, not a substitute for a proper homology search tool.
    """
    alignment = _blosum62_aligner().align(a, b)[0]
    aligned_a, aligned_b = alignment[0], alignment[1]
    matches = sum(1 for x, y in zip(aligned_a, aligned_b, strict=True) if x == y and x != "-")
    return matches / max(len(aligned_a), 1)
