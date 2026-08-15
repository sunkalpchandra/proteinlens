"""On-demand UniProt fetch: embed any accession and search the corpus."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.schemas import FetchProteinRequest, FetchProteinResponse, SearchHitOut
from api.state import AppState, get_state
from ml.uniprot_fetch import UniProtFetchError, fetch_entry, validate_accession

router = APIRouter(tags=["fetch"])


@router.post("/fetch-protein", response_model=FetchProteinResponse)
def fetch_protein(
    req: FetchProteinRequest, state: AppState = Depends(get_state)
) -> FetchProteinResponse:
    # The corpus is authoritative for its own identifiers — check it before
    # enforcing UniProt's accession grammar (which only matters for fetching).
    accession = req.accession.strip().upper()
    if accession in state.by_accession.index:
        hits = state.index("mean").neighbors_of(accession, k=10)
        return FetchProteinResponse(
            source="corpus",
            protein=state.summary_dict(accession),
            hits=[SearchHitOut(rank=h.rank, similarity=round(h.score, 4),
                               protein=state.summary_dict(h.accession))
                  for h in hits],
        )

    try:
        entry = fetch_entry(validate_accession(accession))
    except UniProtFetchError as exc:
        if "does not look like" in str(exc):
            raise HTTPException(422, str(exc)) from exc
        status = 404 if "no entry" in str(exc) else 502
        raise HTTPException(status, str(exc)) from exc

    sequence = entry.pop("sequence")
    with state.encoder_lock:
        vector = state.pipeline.embed(sequence, "mean").embedding
    hits = state.index("mean").search(vector, k=10)
    return FetchProteinResponse(
        source="uniprot",
        protein=entry,
        hits=[SearchHitOut(rank=h.rank, similarity=round(h.score, 4),
                           protein=state.summary_dict(h.accession))
              for h in hits],
    )
