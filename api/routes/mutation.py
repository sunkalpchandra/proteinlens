"""Mutation analysis endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.schemas import (
    LandscapeRequest,
    LandscapeResponse,
    MutationRequest,
    MutationResponse,
    TrajectoryRequest,
    TrajectoryResponse,
)
from api.state import AppState, get_state
from ml.sequence import validate_sequence
from models.mutation import TrajectoryAnalyzer

router = APIRouter(tags=["mutation"])


def _resolve_sequence(state: AppState, accession: str | None, sequence: str | None) -> str:
    if accession is not None:
        return state.protein_row(accession)["sequence"]
    return validate_sequence(sequence or "")


@router.post("/mutation", response_model=MutationResponse)
def mutation(req: MutationRequest, state: AppState = Depends(get_state)) -> MutationResponse:
    seq = _resolve_sequence(state, req.accession, req.sequence)
    with state.encoder_lock:
        effect = state.analyzer.analyze(seq, req.mutation, req.pooling)
    return MutationResponse(
        mutation=effect.mutation,
        pooling=req.pooling,
        displacement=round(effect.displacement, 5),
        relative_displacement=round(effect.relative_displacement, 5),
        cosine_similarity=round(effect.cosine_similarity, 6),
        local_delta=round(effect.local_delta, 5),
        global_residue_delta=round(effect.global_residue_delta, 5),
        per_residue_delta=effect.per_residue_delta,
        top_dimensions=effect.top_dimensions,
    )


@router.post("/mutation-landscape", response_model=LandscapeResponse)
def mutation_landscape(
    req: LandscapeRequest, state: AppState = Depends(get_state)
) -> LandscapeResponse:
    seq = _resolve_sequence(state, req.accession, req.sequence)
    with state.encoder_lock:
        landscape = state.analyzer.landscape(seq, req.position, req.pooling)
    return LandscapeResponse(**landscape)


@router.post("/trajectory", response_model=TrajectoryResponse)
def trajectory(
    req: TrajectoryRequest, state: AppState = Depends(get_state)
) -> TrajectoryResponse:
    seq = _resolve_sequence(state, req.accession, req.sequence)
    # Lazy factory: the analyzer validates the chain before resolving it, so
    # invalid requests never load the encoder.
    analyzer = TrajectoryAnalyzer(lambda: state.pipeline)
    with state.encoder_lock:
        result = analyzer.trajectory(seq, req.mutations, req.pooling)
    return TrajectoryResponse(**result)
