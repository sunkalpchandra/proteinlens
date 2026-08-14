import numpy as np
import pytest

from ml.retrieval import AUTO_ANN_THRESHOLD, ProteinIndex, auto_backend


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(9)
    embeddings = rng.normal(size=(400, 48)).astype(np.float32)
    embeddings[1] = embeddings[0] + rng.normal(scale=0.01, size=48).astype(np.float32)
    accessions = [f"P{i:04d}" for i in range(400)]
    return embeddings, accessions


class TestBackendParity:
    @pytest.mark.parametrize("backend", ["hnsw", "ivf"])
    def test_ann_agrees_with_exact_on_easy_neighbors(self, data, backend):
        embeddings, accessions = data
        exact = ProteinIndex.build(embeddings, accessions, "mean", backend="flat")
        ann = ProteinIndex.build(embeddings, accessions, "mean", backend=backend)
        # The planted near-duplicate is unambiguous; any reasonable ANN finds it.
        assert ann.neighbors_of("P0000", k=1)[0].accession == "P0001"
        exact_top = {h.accession for h in exact.neighbors_of("P0100", k=10)}
        ann_top = {h.accession for h in ann.neighbors_of("P0100", k=10)}
        assert len(exact_top & ann_top) >= 7  # ≥70% overlap on random data

    @pytest.mark.parametrize("backend", ["hnsw", "ivf"])
    def test_reconstruct_works(self, data, backend):
        embeddings, accessions = data
        index = ProteinIndex.build(embeddings, accessions, "mean", backend=backend)
        vec = index.index.reconstruct(5)
        assert np.isfinite(vec).all() and vec.shape == (48,)

    @pytest.mark.parametrize("backend", ["hnsw", "ivf"])
    def test_save_load_roundtrip_preserves_backend(self, data, backend, tmp_path):
        embeddings, accessions = data
        ProteinIndex.build(embeddings, accessions, "mean", backend=backend).save(tmp_path)
        restored = ProteinIndex.load(tmp_path, "mean")
        assert restored.backend == backend
        assert restored.neighbors_of("P0000", k=1)[0].accession == "P0001"

    def test_unknown_backend_rejected(self, data):
        embeddings, accessions = data
        with pytest.raises(ValueError, match="backend"):
            ProteinIndex.build(embeddings, accessions, "mean", backend="lsh")


class TestAutoBackend:
    def test_small_corpora_stay_exact(self):
        assert auto_backend(12_000) == "flat"

    def test_large_corpora_go_ann(self):
        assert auto_backend(AUTO_ANN_THRESHOLD + 1) == "hnsw"
