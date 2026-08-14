"""Domain listings and region-level (per-domain) similarity search."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.schemas import (
    DomainsResponse,
    RegionSearchRequest,
    RegionSearchResponse,
    SearchHitOut,
)
from api.state import AppState, get_state
from ml.regions import region_embedding, validate_region

router = APIRouter(tags=["regions"])


@router.get("/protein/{accession}/domains", response_model=DomainsResponse)
def protein_domains(
    accession: str, state: AppState = Depends(get_state)
) -> DomainsResponse:
    row = state.protein_row(accession)
    return DomainsResponse(
        accession=accession,
        length=int(row["length"]),
        domains=state.domains_for(accession),
    )


@router.post("/region-search", response_model=RegionSearchResponse)
def region_search(
    req: RegionSearchRequest, state: AppState = Depends(get_state)
) -> RegionSearchResponse:
    row = state.protein_row(req.accession)
    seq = row["sequence"]
    validate_region(req.start, req.end, len(seq))

    with state.encoder_lock:
        encoded = state.pipeline.encode_residues(seq)
    vector = region_embedding(encoded.residue_embeddings.numpy(), req.start, req.end)

    hits = state.index("mean").search(vector, k=req.k, exclude=req.accession)
    return RegionSearchResponse(
        accession=req.accession,
        start=req.start,
        end=req.end,
        span_length=req.end - req.start + 1,
        hits=[
            SearchHitOut(rank=h.rank, similarity=round(h.score, 4),
                         protein=state.summary_dict(h.accession))
            for h in hits
        ],
    )
