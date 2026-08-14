import torch

from ml.losses import supcon_loss


def test_clustered_embeddings_score_lower_than_shuffled():
    torch.manual_seed(0)
    centers = torch.randn(4, 16) * 4
    labels = torch.arange(4).repeat_interleave(8)
    clustered = centers[labels] + 0.1 * torch.randn(32, 16)
    shuffled = clustered[torch.randperm(32)]
    assert supcon_loss(clustered, labels) < supcon_loss(shuffled, labels)


def test_perfectly_collapsed_classes_approach_lower_bound():
    centers = torch.nn.functional.normalize(torch.randn(2, 32), dim=1) * 10
    labels = torch.tensor([0] * 6 + [1] * 6)
    embeddings = centers[labels]
    loss = supcon_loss(embeddings, labels, temperature=0.1)
    assert loss.item() < 0.5


def test_anchor_without_positive_contributes_nothing():
    torch.manual_seed(1)
    embeddings = torch.randn(5, 8)
    labels = torch.tensor([0, 0, 1, 1, 2])  # class 2 is a singleton
    with_singleton = supcon_loss(embeddings, labels)
    without = supcon_loss(embeddings[:4], labels[:4])
    assert torch.isfinite(with_singleton)
    # Removing the positive-less anchor leaves the loss unchanged.
    assert torch.allclose(with_singleton, without, atol=1e-5)


def test_all_singletons_returns_zero_with_grad():
    embeddings = torch.randn(4, 8, requires_grad=True)
    labels = torch.arange(4)
    loss = supcon_loss(embeddings, labels)
    assert loss.item() == 0.0
    loss.backward()  # must be differentiable even in the degenerate case


def test_gradient_flows():
    torch.manual_seed(2)
    embeddings = torch.randn(12, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2] * 4)
    supcon_loss(embeddings, labels).backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
