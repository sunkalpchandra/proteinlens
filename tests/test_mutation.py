import pytest

from ml.sequence import SequenceValidationError
from models.mutation import Mutation, apply_mutation, parse_mutation


class TestParse:
    def test_parses_standard_notation(self):
        mut = parse_mutation("H63Y")
        assert mut == Mutation("H", 63, "Y")
        assert mut.label == "H63Y"

    def test_case_and_whitespace_tolerant(self):
        assert parse_mutation(" h63y ") == Mutation("H", 63, "Y")

    @pytest.mark.parametrize("bad", ["63Y", "H63", "HY", "H0.5Y", "H63Z_extra", ""])
    def test_rejects_malformed(self, bad):
        with pytest.raises(SequenceValidationError):
            parse_mutation(bad)

    def test_rejects_noncanonical_residues(self):
        with pytest.raises(SequenceValidationError, match="canonical"):
            parse_mutation("X10Y")


class TestApply:
    SEQ = "MKTVHQAAAA"

    def test_applies_substitution(self):
        assert apply_mutation(self.SEQ, Mutation("H", 5, "Y")) == "MKTVYQAAAA"

    def test_length_preserved(self):
        assert len(apply_mutation(self.SEQ, Mutation("H", 5, "Y"))) == len(self.SEQ)

    def test_rejects_wildtype_mismatch(self):
        with pytest.raises(SequenceValidationError, match="mismatch"):
            apply_mutation(self.SEQ, Mutation("W", 5, "Y"))

    def test_rejects_out_of_range(self):
        with pytest.raises(SequenceValidationError, match="outside"):
            apply_mutation(self.SEQ, Mutation("H", 99, "Y"))

    def test_rejects_identity_mutation(self):
        with pytest.raises(SequenceValidationError, match="not a substitution"):
            apply_mutation(self.SEQ, Mutation("H", 5, "H"))
