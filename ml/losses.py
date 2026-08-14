"""Training objectives for pooler fitting (the encoder stays frozen).

Supervised contrastive loss (SupCon, Khosla et al. 2020): pulls same-class
pooled embeddings together and pushes different classes apart on the unit
sphere — directly optimizing the cosine geometry that retrieval uses, where
cross-entropy only optimizes linear separability.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supcon_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive loss over a batch.

    Args:
        embeddings: [B, D] pooled vectors (any scale; normalized internally).
        labels: [B] integer class labels. Anchors whose class has no other
            member in the batch contribute nothing (standard SupCon handling —
            use a class-balanced sampler so this stays rare).
        temperature: softmax temperature τ.
    """
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings/labels batch mismatch")
    z = F.normalize(embeddings, dim=1)
    sim = z @ z.T / temperature                       # [B, B]

    batch = labels.shape[0]
    eye = torch.eye(batch, dtype=torch.bool, device=z.device)
    positives = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye

    # log-softmax over each row, excluding self-similarity from the denominator.
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    pos_counts = positives.sum(dim=1)
    has_pos = pos_counts > 0
    if not bool(has_pos.any()):
        return torch.zeros((), device=z.device, requires_grad=True)

    mean_pos_log_prob = (log_prob.masked_fill(~positives, 0.0).sum(dim=1)[has_pos]
                         / pos_counts[has_pos])
    return -mean_pos_log_prob.mean()
