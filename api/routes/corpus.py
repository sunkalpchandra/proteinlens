"""Corpus-level endpoints: health, map, clusters, benchmark."""

from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response

from api.schemas import HealthResponse
from api.state import AppState, get_state

router = APIRouter(tags=["corpus"])


@router.get("/health", response_model=HealthResponse)
def health(state: AppState = Depends(get_state)) -> HealthResponse:
    from models.encoder import resolve_device

    return HealthResponse(
        status="ok",
        model=state.store.meta["model"],
        corpus_size=len(state.df),
        poolings=state.store.poolings,
        device=str(resolve_device()),
        encoder_loaded=state.encoder_loaded,
    )


MAP_PRESET_NAMES = ("default", "local", "global")


@router.get("/map")
def embedding_map(
    pooling: str = Query("mean"),
    preset: str = Query("default"),
    state: AppState = Depends(get_state),
) -> Response:
    if preset not in MAP_PRESET_NAMES:
        raise HTTPException(422, f"Unknown preset '{preset}'. Options: {MAP_PRESET_NAMES}")
    suffix = "" if preset == "default" else f"_{preset}"
    path = state.index_dir / f"map_{pooling}{suffix}.json"
    if not path.exists():
        raise HTTPException(
            404,
            f"No map payload for pooling '{pooling}' preset '{preset}'. Run scripts/build_index.py.",
        )
    # Serve the prebuilt file verbatim — no per-request recomputation.
    return Response(content=path.read_bytes(), media_type="application/json")


@router.get("/clusters")
def clusters(
    pooling: str = Query("mean"), state: AppState = Depends(get_state)
) -> Response:
    path = state.index_dir / f"clusters_{pooling}.json"
    if not path.exists():
        raise HTTPException(404, f"No cluster summary for pooling '{pooling}'. Run scripts/build_index.py.")
    return Response(content=path.read_bytes(), media_type="application/json")


@router.get("/benchmark")
def benchmark(state: AppState = Depends(get_state)) -> JSONResponse:
    csv_path = state.reports_dir / "benchmark.csv"
    if not csv_path.exists():
        raise HTTPException(404, "No benchmark results. Run scripts/run_benchmarks.py.")
    table = pd.read_csv(csv_path)
    payload: dict = {"rows": json.loads(table.to_json(orient="records"))}

    sve_path = state.reports_dir / "seq_vs_emb.csv"
    if sve_path.exists():
        sve = pd.read_csv(sve_path)
        if len(sve) > 4000:
            sve = sve.sample(4000, random_state=0)
        payload["seq_vs_emb"] = json.loads(sve.to_json(orient="records"))

    md_path = state.reports_dir / "benchmark.md"
    if md_path.exists():
        payload["markdown"] = md_path.read_text()

    extended_path = state.reports_dir / "extended_benchmark.csv"
    if extended_path.exists():
        payload["extended"] = json.loads(
            pd.read_csv(extended_path).to_json(orient="records")
        )
    return JSONResponse(payload)
