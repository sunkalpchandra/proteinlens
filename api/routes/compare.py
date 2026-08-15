"""Pairwise protein comparison: representation vs sequence similarity."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.schemas import CompareRequest, CompareResponse, ProteinSummary
from api.state import AppState, get_state
from ml.embeddings import cosine_similarity
from ml.sequence import sequence_identity

router = APIRouter(tags=["compare"])


@router.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest, state: AppState = Depends(get_state)) -> CompareResponse:
    row_a = state.protein_row(req.a)
    row_b = state.protein_row(req.b)

    cosines = {
        pooling: round(cosine_similarity(
            state.store.vector(req.a, pooling), state.store.vector(req.b, pooling)
        ), 4)
        for pooling in state.store.poolings
    }
    identity = sequence_identity(row_a["sequence"], row_b["sequence"])

    pfam_a = set(row_a["pfam_all"]) if row_a["pfam_all"] is not None else set()
    pfam_b = set(row_b["pfam_all"]) if row_b["pfam_all"] is not None else set()
    family_a, family_b = row_a["family"], row_b["family"]

    return CompareResponse(
        a=ProteinSummary(**state.summary_dict(req.a)),
        b=ProteinSummary(**state.summary_dict(req.b)),
        cosine_by_pooling=cosines,
        sequence_identity=round(identity, 4),
        same_family=bool(
            isinstance(family_a, str) and isinstance(family_b, str) and family_a == family_b
        ),
        shared_pfam=sorted(pfam_a & pfam_b),
        a_domains=state.domains_for(req.a),
        b_domains=state.domains_for(req.b),
    )
