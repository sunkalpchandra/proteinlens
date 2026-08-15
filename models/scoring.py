"""Zero-shot mutation-effect scoring with the language-model head.

The wild-type-marginal log-likelihood ratio (Meier et al. 2021):

    score(pos, wt→mut) = log P(x_pos = mut | x_wt) − log P(x_pos = wt | x_wt)

computed from ONE forward pass of the masked-LM head over the wild-type
sequence — every position's full 20-way distribution comes out at once, so a
whole 19-substitution landscape (or a full DMS-style scan) costs a single
inference. Negative scores mean the model finds the substitution less likely
than wild-type.

This is the field-standard zero-shot metric and is distinct from embedding
displacement ‖Δz‖: the DMS validation study (scripts/run_dms_validation.py)
measures how each correlates with experimentally assayed variant effects.
It remains a model likelihood statement, not a fitness prediction.
"""

from __future__ import annotations

import torch

from ml.sequence import CANONICAL_AA, validate_sequence
from models.encoder import DEFAULT_MODEL, resolve_device
from models.registry import spec_for


class MaskedLMScorer:
    """Frozen ESM-2 masked-LM head for wild-type-marginal scoring."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.model_name = model_name
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        try:
            self.token_budget = spec_for(model_name).token_budget
        except KeyError:
            self.token_budget = 8192
        # Residue-token ids in canonical order for slicing logits.
        self.aa_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(list(CANONICAL_AA))
        )

    @torch.inference_mode()
    def log_probs(self, sequence: str) -> torch.Tensor:
        """[L, 20] log-softmax over canonical residues at every position,
        from one forward pass over the (unmasked) wild-type sequence."""
        seq = validate_sequence(sequence)
        batch = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logits = self.model(**batch).logits.float().cpu()[0]  # [T, vocab]
        residue_logits = logits[1 : 1 + len(seq)]  # strip BOS; EOS trails
        log_probs = torch.log_softmax(residue_logits, dim=-1)
        if self.device.type == "mps":
            torch.mps.empty_cache()
        return log_probs[:, self.aa_ids]  # [L, 20] in CANONICAL_AA order

    def llr(self, sequence: str, position: int, mutant: str) -> float:
        """Wild-type-marginal LLR for one substitution (1-based position)."""
        scores = self.position_scores(sequence, position)
        return scores[mutant]

    def position_scores(self, sequence: str, position: int) -> dict[str, float]:
        """LLR for every substitution at one position (wild-type scores 0)."""
        seq = validate_sequence(sequence)
        if not 1 <= position <= len(seq):
            raise ValueError(f"Position {position} outside sequence of length {len(seq)}.")
        log_probs = self.log_probs(seq)
        return self._scores_from_log_probs(log_probs, seq, position)

    @staticmethod
    def _scores_from_log_probs(
        log_probs: torch.Tensor, sequence: str, position: int
    ) -> dict[str, float]:
        """Pure math over a precomputed [L, 20] table — testable without a model."""
        row = log_probs[position - 1]
        wt = sequence[position - 1]
        wt_lp = row[CANONICAL_AA.index(wt)]
        return {
            aa: float(row[i] - wt_lp) for i, aa in enumerate(CANONICAL_AA)
        }


def conservation_profile(log_probs: torch.Tensor, sequence: str) -> dict:
    """Per-position conservation signals from one wt-marginal forward pass.

    Two views over the [L, 20] log-prob table:
      entropy    Shannon entropy (nats) of the position's residue distribution
                 — low entropy = the model concentrates mass on few residues.
      wt_logprob log-probability of the wild-type residue — high = the model
                 "expects" what evolution put there.

    Model-dependent statistics about ESM-2's learned distribution, not
    evolutionary conservation scores (no alignment is involved).
    """
    probs = log_probs.exp()
    probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize the 20-slice
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    wt_idx = torch.tensor([CANONICAL_AA.index(a) for a in sequence])
    wt_logprob = log_probs.gather(1, wt_idx.unsqueeze(1)).squeeze(1)
    return {
        "entropy": [round(float(v), 4) for v in entropy],
        "wt_logprob": [round(float(v), 4) for v in wt_logprob],
        "max_entropy": round(float(torch.log(torch.tensor(20.0))), 4),
    }
