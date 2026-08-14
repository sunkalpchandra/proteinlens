"""Region-level embeddings: pool a contiguous residue span into one vector.

A region embedding is the mean of the span's residue representations —
directly comparable to protein-level *mean* embeddings (same construction,
shorter support). That makes "search the corpus with just this domain" a
well-posed cosine query, while remaining a cross-granularity comparison:
a region vector against whole-protein vectors, which the UI labels honestly.
"""

from __future__ import annotations

import numpy as np

from ml.sequence import SequenceValidationError


def validate_region(start: int, end: int, length: int, min_span: int = 5) -> None:
    """1-based inclusive coordinates, must lie inside the sequence."""
    if start < 1 or end > length:
        raise SequenceValidationError(
            f"Region {start}..{end} outside sequence of length {length}."
        )
    if end < start:
        raise SequenceValidationError(f"Region end {end} precedes start {start}.")
    if end - start + 1 < min_span:
        raise SequenceValidationError(
            f"Region {start}..{end} is shorter than {min_span} residues — too "
            "small for a meaningful pooled embedding."
        )


def region_embedding(
    residue_embeddings: np.ndarray, start: int, end: int, min_span: int = 5
) -> np.ndarray:
    """Mean-pool residues ``start..end`` (1-based, inclusive) into one vector."""
    validate_region(start, end, residue_embeddings.shape[0], min_span)
    return residue_embeddings[start - 1 : end].mean(axis=0).astype(np.float32)
