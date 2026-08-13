import numpy as np
import pytest

from ml.retrieval import ProteinIndex


@pytest.fixture
def toy_index() -> ProteinIndex:
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(50, 32)).astype(np.float32)
    # Make P001 and P002 near-duplicates so retrieval order is predictable.
    embeddings[1] = embeddings[0] + rng.normal(scale=0.01, size=32).astype(np.float32)
    accessions = [f"P{i:03d}" for i in range(50)]
    return ProteinIndex.build(embeddings, accessions, "mean")


class TestSearch:
    def test_self_is_top_hit_with_unit_similarity(self, toy_index):
        vec = toy_index.index.reconstruct(0)
        hits = toy_index.search(np.asarray(vec), k=3)
        assert hits[0].accession == "P000"
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)

    def test_exclude_removes_query_protein(self, toy_index):
        hits = toy_index.neighbors_of("P000", k=5)
        assert all(h.accession != "P000" for h in hits)
        assert hits[0].accession == "P001"  # the planted near-duplicate

    def test_scores_sorted_descending(self, toy_index):
        hits = toy_index.neighbors_of("P010", k=10)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
        assert len(hits) == 10

    def test_knn_distances_reasonable(self, toy_index):
        dists = toy_index.knn_distances(k=5)
        assert dists.shape == (50,)
        assert np.all(dists >= 0)
        # The near-duplicate pair should be among the least isolated points.
        assert dists[1] < np.median(dists)


class TestPersistence:
    def test_save_load_roundtrip(self, toy_index, tmp_path):
        toy_index.save(tmp_path)
        restored = ProteinIndex.load(tmp_path, "mean")
        assert restored.accessions == toy_index.accessions
        original = toy_index.neighbors_of("P000", k=3)
        loaded = restored.neighbors_of("P000", k=3)
        assert [h.accession for h in original] == [h.accession for h in loaded]

    def test_load_missing_pooling_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ProteinIndex.load(tmp_path, "max")
