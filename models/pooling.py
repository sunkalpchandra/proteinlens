"""Protein-level pooling strategies over residue representations.

Four strategies turn a variable-length matrix of residue embeddings ``H ∈ R^{L×D}``
into a fixed protein embedding ``z ∈ R^D``:

  mean       z = (1/L) Σ_i h_i
  max        z_d = max_i h_{i,d}
  bos        z = h_BOS                       (ESM's BOS/CLS token)
  attention  z = Σ_i α_i h_i,  α = softmax(score(h_i))    (learned)

The attention pooler is the only strategy with parameters. It is trained with a
lightweight supervised head on frozen embeddings (see ``ml/probes.py``); the
language model itself is never updated. Its per-residue weights ``α`` double as
a model-dependent interpretability signal.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

POOLING_STRATEGIES = ("mean", "max", "bos", "attention")


def mean_pool(residues: torch.Tensor) -> torch.Tensor:
    return residues.mean(dim=0)


def max_pool(residues: torch.Tensor) -> torch.Tensor:
    return residues.max(dim=0).values


class AttentionPooling(nn.Module):
    """Additive attention pooling: score(h) = w2ᵀ tanh(W1 h + b1).

    Works on a single [L, D] matrix or a padded batch [B, L, D] with a boolean
    mask. Returns the pooled vector(s) and the attention weights so callers can
    visualize which residues the pooled representation attends to.
    """

    def __init__(self, hidden_size: int, attn_hidden: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.attn_hidden = attn_hidden
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1, bias=False),
        )

    def forward(
        self, residues: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residues.dim() == 2:  # [L, D] → [1, L, D]
            pooled, weights = self.forward(residues.unsqueeze(0))
            return pooled.squeeze(0), weights.squeeze(0)

        scores = self.scorer(residues).squeeze(-1)  # [B, L]
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)  # [B, L]
        pooled = torch.einsum("bl,bld->bd", weights, residues)
        return pooled, weights

    # -- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hidden_size": self.hidden_size,
                "attn_hidden": self.attn_hidden,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "AttentionPooling":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        module = cls(payload["hidden_size"], payload["attn_hidden"])
        module.load_state_dict(payload["state_dict"])
        module.eval()
        return module


class Pooler:
    """Dispatches pooling by strategy name; owns the optional attention pooler."""

    def __init__(self, attention_pooler: AttentionPooling | None = None) -> None:
        self.attention_pooler = attention_pooler

    def available(self) -> list[str]:
        return [
            s for s in POOLING_STRATEGIES
            if s != "attention" or self.attention_pooler is not None
        ]

    @torch.inference_mode()
    def pool(
        self,
        residues: torch.Tensor,
        bos: torch.Tensor,
        strategy: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (protein_embedding, attention_weights_or_None)."""
        if strategy == "mean":
            return mean_pool(residues), None
        if strategy == "max":
            return max_pool(residues), None
        if strategy == "bos":
            return bos, None
        if strategy == "attention":
            if self.attention_pooler is None:
                raise ValueError(
                    "Attention pooling requested but no trained pooler is loaded. "
                    "Run scripts/train_attention_pooler.py first."
                )
            pooled, weights = self.attention_pooler(residues)
            return pooled, weights
        raise ValueError(f"Unknown pooling strategy: {strategy!r}. Options: {POOLING_STRATEGIES}")
