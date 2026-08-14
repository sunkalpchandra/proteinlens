import numpy as np
import pytest

from ml.regions import region_embedding, validate_region
from ml.sequence import SequenceValidationError


@pytest.fixture
def residues() -> np.ndarray:
    rng = np.random.default_rng(4)
    return rng.normal(size=(100, 16)).astype(np.float32)


class TestValidation:
    def test_full_sequence_is_valid(self):
        validate_region(1, 100, 100)

    @pytest.mark.parametrize("start,end", [(0, 10), (95, 101), (-3, 5)])
    def test_out_of_bounds_rejected(self, start, end):
        with pytest.raises(SequenceValidationError, match="outside"):
            validate_region(start, end, 100)

    def test_inverted_region_rejected(self):
        with pytest.raises(SequenceValidationError, match="precedes"):
            validate_region(50, 40, 100)

    def test_tiny_region_rejected(self):
        with pytest.raises(SequenceValidationError, match="too"):
            validate_region(10, 12, 100)


class TestEmbedding:
    def test_matches_manual_mean(self, residues):
        vec = region_embedding(residues, 11, 20)
        assert np.allclose(vec, residues[10:20].mean(axis=0), atol=1e-6)

    def test_full_region_equals_mean_pooling(self, residues):
        assert np.allclose(region_embedding(residues, 1, 100),
                           residues.mean(axis=0), atol=1e-6)

    def test_single_offset_changes_vector(self, residues):
        a = region_embedding(residues, 1, 50)
        b = region_embedding(residues, 2, 51)
        assert not np.allclose(a, b)
