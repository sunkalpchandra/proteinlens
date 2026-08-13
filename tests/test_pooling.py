import torch

from models.pooling import AttentionPooling, Pooler, max_pool, mean_pool


def toy_residues() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(12, 16)


class TestParameterFreePooling:
    def test_mean_matches_manual(self):
        residues = toy_residues()
        assert torch.allclose(mean_pool(residues), residues.mean(0))

    def test_max_is_elementwise(self):
        residues = toy_residues()
        pooled = max_pool(residues)
        assert pooled.shape == (16,)
        assert torch.all(pooled >= residues.max(0).values - 1e-6)

    def test_pooler_dispatch_and_bos(self):
        residues, bos = toy_residues(), torch.ones(16)
        pooler = Pooler(None)
        assert torch.allclose(pooler.pool(residues, bos, "bos")[0], bos)
        assert "attention" not in pooler.available()


class TestAttentionPooling:
    def test_weights_sum_to_one(self):
        module = AttentionPooling(16, attn_hidden=8)
        pooled, weights = module(toy_residues())
        assert pooled.shape == (16,)
        assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)

    def test_masked_positions_get_zero_weight(self):
        module = AttentionPooling(16, attn_hidden=8)
        batch = torch.randn(2, 10, 16)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[0, 6:] = False
        _, weights = module(batch, mask)
        assert weights[0, 6:].abs().max() < 1e-6
        assert torch.isclose(weights.sum(-1), torch.ones(2), atol=1e-5).all()

    def test_pooled_is_convex_combination(self):
        module = AttentionPooling(16, attn_hidden=8)
        residues = toy_residues()
        pooled, weights = module(residues)
        assert torch.allclose(pooled, (weights.unsqueeze(-1) * residues).sum(0), atol=1e-5)

    def test_save_load_roundtrip(self, tmp_path):
        module = AttentionPooling(16, attn_hidden=8)
        path = tmp_path / "pooler.pt"
        module.save(path)
        restored = AttentionPooling.load(path)
        residues = toy_residues()
        assert torch.allclose(module(residues)[0], restored(residues)[0])
