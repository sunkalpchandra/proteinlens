"""Trajectory validation logic (no model needed — a stub pipeline suffices)."""

import pytest
import torch

from ml.sequence import SequenceValidationError
from models.encoder import EncodedProtein
from models.mutation import TrajectoryAnalyzer
from models.pooling import Pooler

SEQ = "MKTVHQAAAAWWCHACDEFGHIKLMNPQRSTVWY"


class StubEncoder:
    """Deterministic fake residues: value depends on the residue letter."""

    def encode_batch(self, sequences):
        out = []
        for seq in sequences:
            residues = torch.tensor(
                [[float(ord(a))] * 4 for a in seq], dtype=torch.float32
            )
            out.append(EncodedProtein(seq, residues, residues[0]))
        return out


class StubPipeline:
    encoder = StubEncoder()
    pooler = Pooler(None)

    def embed(self, sequence, pooling="mean"):
        enc = self.encoder.encode_batch([sequence])[0]
        pooled, _ = self.pooler.pool(enc.residue_embeddings, enc.bos_embedding, pooling)

        class Result:
            embedding = pooled.numpy()

        return Result()


@pytest.fixture
def analyzer() -> TrajectoryAnalyzer:
    return TrajectoryAnalyzer(StubPipeline())


class TestValidation:
    def test_empty_trajectory_rejected(self, analyzer):
        with pytest.raises(SequenceValidationError, match="at least one"):
            analyzer.trajectory(SEQ, [])

    def test_too_long_trajectory_rejected(self, analyzer):
        muts = ["M1A", "A1M"] * 6
        with pytest.raises(SequenceValidationError, match="maximum"):
            analyzer.trajectory(SEQ, muts)

    def test_labels_validate_against_cumulative_sequence(self, analyzer):
        # H5Y makes position 5 a Y; the next label must say Y5..., not H5...
        with pytest.raises(SequenceValidationError, match="mismatch"):
            analyzer.trajectory(SEQ, ["H5Y", "H5W"])
        result = analyzer.trajectory(SEQ, ["H5Y", "Y5W"])
        assert [s["mutation"] for s in result["steps"]] == ["H5Y", "Y5W"]


class TestGeometry:
    def test_round_trip_returns_to_wt(self, analyzer):
        result = analyzer.trajectory(SEQ, ["H5Y", "Y5H"])
        assert result["net_displacement"] == pytest.approx(0.0, abs=1e-5)
        assert result["path_length"] > 0
        assert result["directness"] == pytest.approx(0.0, abs=1e-5)

    def test_single_step_is_perfectly_direct(self, analyzer):
        result = analyzer.trajectory(SEQ, ["H5Y"])
        assert result["directness"] == pytest.approx(1.0, abs=1e-6)
        assert result["steps"][0]["displacement_from_wt"] == pytest.approx(
            result["steps"][0]["step_displacement"]
        )

    def test_cumulative_lists_grow(self, analyzer):
        result = analyzer.trajectory(SEQ, ["H5Y", "Y5W", "M1A"])
        assert result["steps"][2]["cumulative"] == ["H5Y", "Y5W", "M1A"]
