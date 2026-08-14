"""ProstT5 encoder: a structure-aware representation baseline.

ProstT5 (Rostlab/ProstT5) is a ProtT5-XL derivative fine-tuned to translate
between amino-acid sequences and 3Di structure tokens; its encoder therefore
carries structure supervision that pure sequence models lack. ProteinLens uses
the encoder half only, fp16, mean-pooled — as a *baseline representation* in
the extended benchmark, never as a serving encoder (1.2B parameters).

T5 tokenization differs from ESM: residues are space-separated single-token
symbols behind a task prefix, so masking must drop the prefix and the closing
``</s>`` before pooling.
"""

from __future__ import annotations

import re

import numpy as np
import torch

PROST_MODEL = "Rostlab/ProstT5"
# Amino-acid input prefix (the paired prefix, <fold2AA>, is for 3Di input).
AA_PREFIX = "<AA2fold>"


def prost_preprocess(sequence: str) -> str:
    """Uppercase, map rare/ambiguous residues to X, space-separate, prefix."""
    cleaned = re.sub(r"[UZOB]", "X", sequence.upper())
    return f"{AA_PREFIX} {' '.join(cleaned)}"


class ProstT5Encoder:
    """Frozen ProstT5 encoder producing mean-pooled per-protein embeddings."""

    def __init__(self, device: str | None = None, half: bool = True) -> None:
        from transformers import T5EncoderModel, T5Tokenizer

        from models.encoder import resolve_device

        self.device = resolve_device(device)
        self.tokenizer = T5Tokenizer.from_pretrained(PROST_MODEL, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(PROST_MODEL)
        # fp16 halves the 4.8GB fp32 footprint; CPU stays fp32 (no half kernels).
        self.half = half and self.device.type != "cpu"
        if self.half:
            self.model = self.model.half()
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.hidden_size: int = self.model.config.d_model

    @torch.inference_mode()
    def embed_mean(self, sequences: list[str]) -> np.ndarray:
        """Mean-pooled residue embeddings for a batch (call in small chunks)."""
        batch = self.tokenizer(
            [prost_preprocess(s) for s in sequences],
            return_tensors="pt", padding=True, add_special_tokens=True,
        ).to(self.device)
        hidden = self.model(**batch).last_hidden_state.float()  # [B, T, D]

        out = np.zeros((len(sequences), self.hidden_size), dtype=np.float32)
        mask = batch["attention_mask"].bool()
        for i, seq in enumerate(sequences):
            # Layout per row: [prefix] r_1 … r_L [</s>] [pad…]
            length = int(mask[i].sum())
            residues = hidden[i, 1 : length - 1]
            if residues.shape[0] != len(seq):
                raise RuntimeError(
                    f"ProstT5 tokenization mismatch: {residues.shape[0]} residue "
                    f"positions for a {len(seq)}-aa sequence."
                )
            out[i] = residues.mean(dim=0).cpu().numpy()
        return out
