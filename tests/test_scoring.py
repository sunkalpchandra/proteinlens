import math
import os

import pytest
import torch

from ml.sequence import CANONICAL_AA
from models.scoring import MaskedLMScorer


class TestScoreMath:
    """The LLR arithmetic over a synthetic log-prob table (no model)."""

    def make_table(self, length: int = 6) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.log_softmax(torch.randn(length, 20), dim=-1)

    def test_wildtype_scores_zero(self):
        table = self.make_table()
        seq = "MKTVHQ"
        scores = MaskedLMScorer._scores_from_log_probs(table, seq, 3)
        assert scores["T"] == pytest.approx(0.0, abs=1e-6)  # position 3 is T

    def test_scores_are_log_ratios(self):
        table = self.make_table()
        seq = "MKTVHQ"
        scores = MaskedLMScorer._scores_from_log_probs(table, seq, 1)
        m_i, a_i = CANONICAL_AA.index("M"), CANONICAL_AA.index("A")
        expected = float(table[0, a_i] - table[0, m_i])
        assert scores["A"] == pytest.approx(expected, abs=1e-6)

    def test_all_twenty_residues_scored(self):
        scores = MaskedLMScorer._scores_from_log_probs(self.make_table(), "MKTVHQ", 2)
        assert set(scores) == set(CANONICAL_AA)
        assert all(math.isfinite(v) for v in scores.values())


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="loads ESM-2 LM head; set RUN_MODEL_TESTS=1",
)
class TestScorerModel:
    SEQ = (
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHV"
        "DDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
    )

    def test_log_probs_shape_and_normalization(self):
        scorer = MaskedLMScorer()
        table = scorer.log_probs(self.SEQ)
        assert table.shape == (len(self.SEQ), 20)
        # Rows are slices of a softmax over the full vocab; the 20 canonical
        # residues carry almost all mass, so sums land just under 1.
        sums = table.exp().sum(dim=-1)
        assert float(sums.min()) > 0.8
        assert float(sums.max()) <= 1.0 + 1e-5

    def test_conserved_proline_disfavors_substitution(self):
        scorer = MaskedLMScorer()
        scores = scorer.position_scores(self.SEQ, 59)  # distal His of hemoglobin α
        assert scores["H"] == pytest.approx(0.0, abs=1e-6)
        # The model should consider most substitutions at a conserved His
        # less likely than wild-type.
        negative = sum(1 for aa, s in scores.items() if aa != "H" and s < 0)
        assert negative >= 15
