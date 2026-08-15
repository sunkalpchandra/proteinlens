"""Protein lookup, profile, and attention endpoints."""

from __future__ import annotations

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Query

from api.schemas import (
    AttentionResponse,
    ConservationResponse,
    ProteinProfile,
    ProteinSummary,
    RepresentationStats,
    SearchHitOut,
)
from api.state import AppState, get_state

router = APIRouter(tags=["proteins"])


@router.get("/proteins", response_model=list[ProteinSummary])
def find_proteins(
    q: str = Query(..., min_length=1, max_length=120),
    limit: int = Query(20, ge=1, le=100),
    state: AppState = Depends(get_state),
) -> list[ProteinSummary]:
    """Name/gene/accession text lookup (not semantic search)."""
    needle = q.strip().lower()
    df = state.df

    name = df["protein_name"].str.lower()
    gene = df["gene"].fillna("").str.lower()
    accession = df["accession"].str.lower()
    family = df["family"].fillna("").str.lower()

    score = np.zeros(len(df))
    score += np.where(accession == needle, 120.0, 0.0)
    score += np.where(gene == needle, 100.0, 0.0)
    score += np.where(name.str.startswith(needle), 80.0, 0.0)
    score += np.where(name.str.contains(needle, regex=False), 40.0, 0.0)
    score += np.where(gene.str.contains(needle, regex=False), 25.0, 0.0)
    score += np.where(family.str.contains(needle, regex=False), 10.0, 0.0)
    score += np.where(accession.str.contains(needle, regex=False), 8.0, 0.0)

    order = np.argsort(-score, kind="stable")[:limit]
    rows = [i for i in order if score[i] > 0]
    return [ProteinSummary(**state.summary_dict(df["accession"].iat[i])) for i in rows]


@router.get("/protein/{accession}", response_model=ProteinProfile)
def protein_profile(
    accession: str,
    pooling: str = Query("mean"),
    k: int = Query(10, ge=1, le=50),
    state: AppState = Depends(get_state),
) -> ProteinProfile:
    row = state.protein_row(accession)
    vector = state.store.vector(accession, pooling)

    point = None
    try:
        point = state.map_point(pooling, accession)
    except FileNotFoundError:
        pass  # map payload not built for this pooling — stats degrade gracefully

    neighbors = state.index(pooling).neighbors_of(accession, k=k)
    nn_distance = round(1.0 - neighbors[0].score, 5) if neighbors else None

    keywords = row["keywords"] if isinstance(row["keywords"], str) else ""
    return ProteinProfile(
        protein=ProteinSummary(**state.summary_dict(accession)),
        protein_name_full=row["protein_name_full"],
        keywords=[k.strip() for k in keywords.split(";") if k.strip()][:20],
        pdb=state.pdb_for(accession)[:12],
        sequence=row["sequence"],
        model=state.store.meta["model"],
        stats=RepresentationStats(
            embedding_norm=round(float(np.linalg.norm(vector)), 4),
            dim=int(vector.shape[0]),
            nn_distance=nn_distance,
            knn_mean_distance=point["knn_dist"] if point else None,
            cluster=point["cluster"] if point else None,
            outlier_score=point["outlier"] if point else None,
            x=point["x"] if point else None,
            y=point["y"] if point else None,
        ),
        neighbors=[
            SearchHitOut(
                rank=h.rank,
                similarity=round(h.score, 4),
                protein=state.summary_dict(h.accession),
            )
            for h in neighbors
        ],
    )


@router.get("/protein/{accession}/attention", response_model=AttentionResponse)
def protein_attention(
    accession: str, state: AppState = Depends(get_state)
) -> AttentionResponse:
    pooler = state.pipeline.pooler.attention_pooler
    if pooler is None:
        raise HTTPException(
            status_code=404,
            detail="No trained attention pooler available; run scripts/train_attention_pooler.py.",
        )
    row = state.protein_row(accession)
    with state.encoder_lock, torch.inference_mode():
        encoded = state.pipeline.encode_residues(row["sequence"])
        _, weights = pooler(encoded.residue_embeddings)
    w = weights.numpy()
    top = np.argsort(-w)[:10]
    return AttentionResponse(
        accession=accession,
        length=int(row["length"]),
        weights=w.round(6).tolist(),
        top_positions=[int(p) + 1 for p in top],  # 1-based positions
    )


@router.get("/protein/{accession}/conservation", response_model=ConservationResponse)
def protein_conservation(
    accession: str, state: AppState = Depends(get_state)
) -> ConservationResponse:
    from models.scoring import conservation_profile

    row = state.protein_row(accession)
    with state.encoder_lock:
        log_probs = state.scorer.log_probs(row["sequence"])
    profile = conservation_profile(log_probs, row["sequence"])
    return ConservationResponse(
        accession=accession,
        length=int(row["length"]),
        **profile,
    )
