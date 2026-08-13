import numpy as np
import pytest

from ml.sequence import (
    AA_CATEGORIES,
    CANONICAL_AA,
    SequenceValidationError,
    clean_sequence,
    kmer_features,
    onehot_mean_features,
    sequence_identity,
    validate_sequence,
)


class TestValidation:
    def test_accepts_canonical(self):
        assert validate_sequence("ACDEFGHIKLMNPQRSTVWY") == "ACDEFGHIKLMNPQRSTVWY"

    def test_cleans_whitespace_and_case(self):
        assert clean_sequence("mkt v\nlq*") == "MKTVLQ"

    def test_rejects_ambiguity_codes(self):
        for bad in ("MKTXVLQAAAAA", "MKTBVLQAAAAA", "MKTUVLQAAAAA"):
            with pytest.raises(SequenceValidationError, match="non-canonical"):
                validate_sequence(bad)

    def test_rejects_empty_and_short(self):
        with pytest.raises(SequenceValidationError):
            validate_sequence("")
        with pytest.raises(SequenceValidationError, match="too short"):
            validate_sequence("MKT")

    def test_rejects_too_long(self):
        with pytest.raises(SequenceValidationError, match="too long"):
            validate_sequence("A" * 5000)


class TestCategories:
    def test_every_amino_acid_categorized(self):
        assert set(AA_CATEGORIES) == set(CANONICAL_AA)

    def test_expected_assignments(self):
        assert AA_CATEGORIES["L"] == "hydrophobic"
        assert AA_CATEGORIES["K"] == "positive"
        assert AA_CATEGORIES["D"] == "negative"
        assert AA_CATEGORIES["W"] == "aromatic"
        assert AA_CATEGORIES["P"] == "special"


class TestFeaturizers:
    def test_kmer_normalized(self):
        vec = kmer_features("MKTVLQMKTVLQ", k=3)
        assert vec.shape == (8000,)
        assert np.isclose(vec.sum(), 1.0)

    def test_onehot_mean_is_composition(self):
        vec = onehot_mean_features("AAAA")
        assert np.isclose(vec.sum(), 1.0)
        assert vec[CANONICAL_AA.index("A")] == 1.0

    def test_identity_of_identical_sequences(self):
        assert sequence_identity("MKTVLQAAAA", "MKTVLQAAAA") == 1.0

    def test_identity_symmetric_and_bounded(self):
        a, b = "MKTVLQAAAAWWCH", "MKTVLQAAAAYYCH"
        ab, ba = sequence_identity(a, b), sequence_identity(b, a)
        assert ab == ba
        assert 0.0 < ab < 1.0
