import numpy as np

from ml.embeddings import ResidueLRU, SqliteVectorCache, embedding_cache_key, l2_normalize


class TestCacheKey:
    def test_deterministic(self):
        a = embedding_cache_key("MKTV", "model-a", "mean")
        assert a == embedding_cache_key("MKTV", "model-a", "mean")

    def test_sensitive_to_all_inputs(self):
        base = embedding_cache_key("MKTV", "model-a", "mean")
        assert base != embedding_cache_key("MKTA", "model-a", "mean")
        assert base != embedding_cache_key("MKTV", "model-b", "mean")
        assert base != embedding_cache_key("MKTV", "model-a", "max")


class TestSqliteVectorCache:
    def test_roundtrip(self, tmp_path):
        cache = SqliteVectorCache(tmp_path / "c.sqlite")
        vec = np.arange(8, dtype=np.float32)
        cache.put("k1", vec)
        out = cache.get("k1")
        assert out is not None and np.array_equal(out, vec)
        assert len(cache) == 1

    def test_miss_returns_none(self, tmp_path):
        cache = SqliteVectorCache(tmp_path / "c.sqlite")
        assert cache.get("missing") is None

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "c.sqlite"
        SqliteVectorCache(path).put("k", np.ones(4, dtype=np.float32))
        assert SqliteVectorCache(path).get("k") is not None


class TestResidueLRU:
    def test_evicts_oldest(self):
        lru = ResidueLRU(maxsize=2)
        lru.put("a", "A")  # type: ignore[arg-type]
        lru.put("b", "B")  # type: ignore[arg-type]
        lru.put("c", "C")  # type: ignore[arg-type]
        assert lru.get("a") is None
        assert lru.get("b") == "B"

    def test_get_refreshes_recency(self):
        lru = ResidueLRU(maxsize=2)
        lru.put("a", "A")  # type: ignore[arg-type]
        lru.put("b", "B")  # type: ignore[arg-type]
        lru.get("a")
        lru.put("c", "C")  # type: ignore[arg-type]
        assert lru.get("b") is None
        assert lru.get("a") == "A"


class TestNormalization:
    def test_rows_unit_norm(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(5, 16)).astype(np.float32)
        norms = np.linalg.norm(l2_normalize(x), axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_vector_input(self):
        v = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
        assert np.allclose(v, [0.6, 0.8])

    def test_zero_vector_safe(self):
        v = l2_normalize(np.zeros(4, dtype=np.float32))
        assert np.all(np.isfinite(v))
