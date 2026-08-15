import numpy as np
import pytest

from ml.projection import MAP_PRESETS, ProjectionCache, ProjectionParams, map_filename


class TestPresetRegistry:
    def test_default_preset_exists(self):
        assert "default" in MAP_PRESETS

    def test_presets_carry_umap_parameters(self):
        for params in MAP_PRESETS.values():
            assert set(params) == {"n_neighbors", "min_dist"}

    def test_map_filename_suffix_rule(self):
        assert map_filename("mean") == "map_mean.json"
        assert map_filename("mean", "local") == "map_mean_local.json"
        assert map_filename("attention", "global") == "map_attention_global.json"

    def test_unknown_preset_rejected(self):
        with pytest.raises(KeyError, match="huge"):
            map_filename("mean", "huge")


class TestProjectionCacheKeys:
    def test_key_differs_by_preset_parameters(self):
        base = ProjectionParams(pooling="mean", **MAP_PRESETS["default"])
        local = ProjectionParams(pooling="mean", **MAP_PRESETS["local"])
        assert base.cache_key("fp") != local.cache_key("fp")

    def test_key_differs_by_data_fingerprint(self):
        params = ProjectionParams(pooling="mean")
        assert params.cache_key("aaaa") != params.cache_key("bbbb")

    def test_stale_shape_forces_recompute(self, tmp_path):
        """A cached projection whose row count mismatches the matrix must be
        ignored (guards fingerprint collisions and hand-copied caches)."""
        cache = ProjectionCache(tmp_path)
        params = ProjectionParams(pooling="mean", pca_dim=4, n_neighbors=3, min_dist=0.1)
        rng = np.random.default_rng(0)
        small = rng.normal(size=(30, 8)).astype(np.float32)
        coords_a, _, hit_a = cache.load_or_compute(small, params, "same-fp")
        assert not hit_a and coords_a.shape == (30, 2)

        grown = rng.normal(size=(35, 8)).astype(np.float32)
        coords_b, _, hit_b = cache.load_or_compute(grown, params, "same-fp")
        assert not hit_b  # shape guard rejected the stale entry
        assert coords_b.shape == (35, 2)
