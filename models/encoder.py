"""ESM-2 encoder: pretrained protein language-model inference.

Wraps a HuggingFace ESM-2 checkpoint and exposes residue-level representations
with all special tokens (BOS/EOS/padding) stripped, so downstream code can index
representation ``i`` as residue ``i`` of the input sequence.

The model is frozen — ProteinLens never fine-tunes the language model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"

# Token budget per forward pass; batches are packed until they would exceed it.
# Conservative default keeps peak memory modest on CPU/MPS machines.
DEFAULT_TOKEN_BUDGET = 16384


@dataclass
class EncodedProtein:
    """Residue-level output for one sequence (special tokens removed)."""

    sequence: str
    residue_embeddings: torch.Tensor  # [L, D] float32, L == len(sequence)
    bos_embedding: torch.Tensor       # [D] representation of the BOS/CLS token

    @property
    def length(self) -> int:
        return self.residue_embeddings.shape[0]


def resolve_device(preferred: str | None = None) -> torch.device:
    """cuda > mps > cpu, overridable via argument or PROTEINLENS_DEVICE."""
    name = preferred or os.environ.get("PROTEINLENS_DEVICE")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ESM2Encoder:
    """Frozen ESM-2 inference with correct special-token handling."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        token_budget: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        if token_budget is None:
            # Larger checkpoints get smaller per-forward budgets (see registry);
            # unknown/custom checkpoints fall back to the conservative default.
            try:
                from models.registry import spec_for

                token_budget = spec_for(model_name).token_budget
            except KeyError:
                token_budget = DEFAULT_TOKEN_BUDGET
        self.token_budget = token_budget
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # add_pooling_layer=False: we pool ourselves; avoids a randomly
        # initialized (and unused) EsmPooler in the checkpoint load.
        self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.hidden_size: int = self.model.config.hidden_size
        self.num_layers: int = self.model.config.num_hidden_layers

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _forward(self, sequences: list[str]) -> list[EncodedProtein]:
        batch = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        special_mask = batch.pop("special_tokens_mask").bool()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        hidden = self.model(**batch).last_hidden_state.float().cpu()  # [B, T, D]
        if self.device.type == "mps":
            # The MPS caching allocator accumulates across long runs and tips
            # 8GB unified hosts into OOM mid-corpus; releasing after each
            # forward costs ~ms and bounds the footprint to one batch.
            torch.mps.empty_cache()

        outputs: list[EncodedProtein] = []
        attention_mask = batch["attention_mask"].bool().cpu()
        for i, seq in enumerate(sequences):
            valid = attention_mask[i] & ~special_mask[i]  # residues only
            residues = hidden[i][valid]
            if residues.shape[0] != len(seq):
                raise RuntimeError(
                    f"Residue/representation mismatch: {residues.shape[0]} vectors "
                    f"for a {len(seq)}-residue sequence. Tokenizer produced an "
                    f"unexpected tokenization; check for non-canonical symbols."
                )
            outputs.append(
                EncodedProtein(
                    sequence=seq,
                    residue_embeddings=residues,
                    bos_embedding=hidden[i, 0].clone(),
                )
            )
        return outputs

    def encode(self, sequence: str) -> EncodedProtein:
        return self._forward([sequence])[0]

    def encode_batch(self, sequences: list[str]) -> list[EncodedProtein]:
        """Encode many sequences, packing forward passes under the token budget.

        Sequences are sorted by length internally (minimizes padding waste) and
        results are returned in the original order.
        """
        if not sequences:
            return []
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        results: list[EncodedProtein | None] = [None] * len(sequences)

        chunk: list[int] = []
        max_len = 0
        for idx in order:
            candidate_max = max(max_len, len(sequences[idx]) + 2)  # +BOS/EOS
            if chunk and candidate_max * (len(chunk) + 1) > self.token_budget:
                for j, enc in zip(chunk, self._forward([sequences[j] for j in chunk]), strict=True):
                    results[j] = enc
                chunk, max_len = [], 0
                candidate_max = len(sequences[idx]) + 2
            chunk.append(idx)
            max_len = candidate_max
        if chunk:
            for j, enc in zip(chunk, self._forward([sequences[j] for j in chunk]), strict=True):
                results[j] = enc
        return results  # type: ignore[return-value]
