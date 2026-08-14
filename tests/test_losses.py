import torch

from ml.losses import supcon_loss


def test_clustered_embeddings_score_lower_than_shuffled():
    torch.manual_seed(0)
    centers = torch.randn(4, 16) * 4
    labels = torch.arange(4).repeat_interleave(8)
    clustered = centers[labels] + 0.1 * torch.randn(32, 16)
    shuffled = clustered[torch.randperm(32)]
    assert supcon_loss(clustered, labels) < supcon_loss(shuffled, labels)


def test_perfectly_collapsed_classes_hit_the_analytic_floor():
    """With every class collapsed to a point and classes well separated, the
    loss floor is log(P) for P positives per anchor (softmax mass splits
    evenly across identical positives) — here log(5)."""
    import math

    torch.manual_seed(3)
    centers = torch.nn.functional.normalize(torch.randn(2, 32), dim=1)
    labels = torch.tensor([0] * 6 + [1] * 6)
    embeddings = centers[labels]
    loss = supcon_loss(embeddings, labels, temperature=0.1)
    assert abs(loss.item() - math.log(5)) < 0.3


def test_singleton_class_is_negative_only():
    """A positive-less anchor adds no anchor term (loss stays finite) but it
    still serves as a negative: perturbing it must change the loss."""
    torch.manual_seed(1)
    embeddings = torch.randn(5, 8)
    labels = torch.tensor([0, 0, 1, 1, 2])  # class 2 is a singleton
    base = supcon_loss(embeddings, labels)
    assert torch.isfinite(base)
    moved = embeddings.clone()
    moved[4] += 3.0
    assert not torch.allclose(base, supcon_loss(moved, labels), atol=1e-6)


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
