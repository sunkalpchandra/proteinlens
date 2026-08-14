import pytest

from models.registry import DEFAULT_MODEL, REGISTRY, check_memory, spec_for, study_dir


class TestRegistry:
    def test_default_model_registered(self):
        assert DEFAULT_MODEL in REGISTRY

    def test_specs_are_sane(self):
        for spec in REGISTRY.values():
            assert spec.hidden_size > 0
            assert spec.params_m > 0
            assert spec.token_budget >= 1024
            assert spec.approx_fp32_gb > 0

    def test_larger_models_get_smaller_token_budgets(self):
        esm = sorted(
            (s for s in REGISTRY.values() if s.family == "esm2"),
            key=lambda s: s.params_m,
        )
        budgets = [s.token_budget for s in esm]
        assert budgets == sorted(budgets, reverse=True)

    def test_unknown_model_raises_with_listing(self):
        with pytest.raises(KeyError, match="Registered:"):
            spec_for("facebook/esm2_t99_9B")

    def test_study_dir_uses_slug(self):
        assert study_dir("facebook/esm2_t6_8M_UR50D").name == "esm2_t6_8M"

    def test_memory_warning_thresholds(self):
        assert check_memory("facebook/esm2_t6_8M_UR50D", 8.0) is None
        warning = check_memory("facebook/esm2_t33_650M_UR50D", 8.0)
        assert warning is not None and "16 GB" in warning
