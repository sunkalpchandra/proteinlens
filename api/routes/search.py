"""Embedding and semantic search endpoints."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends

from api.schemas import EmbedRequest, EmbedResponse, SearchHitOut, SearchRequest, SearchResponse
from api.state import AppState, get_state
from ml.embeddings import EMBEDDING_VERSION
from ml.sequence import validate_sequence

router = APIRouter(tags=["search"])


@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest, state: AppState = Depends(get_state)) -> EmbedResponse:
    seq = validate_sequence(req.sequence)
    with state.encoder_lock:
        result = state.pipeline.embed(seq, req.pooling, with_residues=True)

    residue_norms = None
    if req.include_residue_norms and result.residue_embeddings is not None:
        residue_norms = np.linalg.norm(result.residue_embeddings, axis=1).round(4).tolist()

    attention = None
    if req.include_attention:
        pooler = state.pipeline.pooler.attention_pooler
        if pooler is not None:
            import torch

            with state.encoder_lock, torch.inference_mode():
                encoded = state.pipeline.encode_residues(seq)
                _, weights = pooler(encoded.residue_embeddings)
            attention = weights.numpy().round(6).tolist()

    return EmbedResponse(
        embedding=result.embedding.round(5).tolist(),
        length=result.length,
        model=result.model,
        pooling=req.pooling,
        embedding_version=EMBEDDING_VERSION,
        cache_hit=result.cache_hit,
        embedding_norm=round(float(np.linalg.norm(result.embedding)), 4),
        residue_norms=residue_norms,
        attention_weights=attention,
    )


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, state: AppState = Depends(get_state)) -> SearchResponse:
    if req.accession is not None:
        query = state.store.vector(req.accession, req.pooling)
        query_length = int(state.protein_row(req.accession)["length"])
        exclude = req.accession
    else:
        seq = validate_sequence(req.sequence or "")
        with state.encoder_lock:
            query = state.pipeline.embed(seq, req.pooling).embedding
        query_length = len(seq)
        exclude = None

    hits = state.index(req.pooling).search(query, k=req.k, exclude=exclude)
    return SearchResponse(
        pooling=req.pooling,
        query_length=query_length,
        hits=[
            SearchHitOut(
                rank=h.rank,
                similarity=round(h.score, 4),
                protein=state.summary_dict(h.accession),
            )
            for h in hits
        ],
    )
