"""Corpus-level endpoints: health, map, clusters, benchmark."""

from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response

from api.schemas import HealthResponse, StatsResponse
from api.state import AppState, get_state
from ml.projection import MAP_PRESETS, map_filename

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


@router.get("/map")
def embedding_map(
    pooling: str = Query("mean"),
    preset: str = Query("default"),
    state: AppState = Depends(get_state),
) -> Response:
    if preset not in MAP_PRESETS:
        raise HTTPException(422, f"Unknown preset '{preset}'. Options: {sorted(MAP_PRESETS)}")
    path = state.index_dir / map_filename(pooling, preset)
    if not path.exists():
        if preset != "default" and pooling != "mean":
            raise HTTPException(
                404,
                f"Alternative presets are built only for mean pooling; "
                f"'{pooling}' serves preset 'default' only.",
            )
        raise HTTPException(
            404,
            f"No map payload for pooling '{pooling}' preset '{preset}'. Run scripts/build_index.py.",
        )
    # Serve the prebuilt file verbatim — no per-request recomputation.
    return Response(content=path.read_bytes(), media_type="application/json")


@router.get("/clusters")
def clusters(
    pooling: str = Query("mean"),
    algorithm: str = Query("kmeans"),
    state: AppState = Depends(get_state),
) -> Response:
    if algorithm not in ("kmeans", "hdbscan"):
        raise HTTPException(422, f"Unknown algorithm '{algorithm}'. Options: kmeans, hdbscan")
    suffix = "" if algorithm == "kmeans" else "_hdbscan"
    path = state.index_dir / f"clusters_{pooling}{suffix}.json"
    if not path.exists():
        raise HTTPException(
            404, f"No {algorithm} summary for pooling '{pooling}'. Run scripts/build_index.py."
        )
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


@router.get("/stats", response_model=StatsResponse)
def stats(state: AppState = Depends(get_state)) -> StatsResponse:
    """Operational snapshot: corpus composition, cache size, artifact vintage."""
    # Count cache entries without leaking connections: reuse the pipeline's
    # open handle when the encoder is loaded, else one short-lived connection.
    cache_path = state.embeddings_dir / "adhoc_cache.sqlite"
    cache_entries = 0
    if state.encoder_loaded and state.pipeline.cache is not None:
        cache_entries = len(state.pipeline.cache)
    elif cache_path.exists():
        import sqlite3

        with sqlite3.connect(cache_path) as conn:
            try:
                (cache_entries,) = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
            except sqlite3.OperationalError:
                cache_entries = 0  # cache file exists but table not created yet

    # Backend name from the sidecar written at build time — reporting it must
    # not force the FAISS index into memory on a cold deployment.
    backend = "flat"
    backend_meta = state.index_dir / "index_mean_meta.json"
    if backend_meta.exists():
        backend = json.loads(backend_meta.read_text()).get("backend", "flat")

    return StatsResponse(
        corpus_size=len(state.df),
        n_families=int(state.df["family"].nunique()),
        n_with_domains=state.n_proteins_with_domains(),
        poolings=state.store.poolings,
        index_backend=backend,
        adhoc_cache_entries=int(cache_entries),
        encoder_loaded=state.encoder_loaded,
        embeddings_created_at=state.store.meta.get("created_at"),
        appended_proteins=state.store.meta.get("appended", []),
    )
