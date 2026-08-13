"""ProteinLens API entrypoint.

    uvicorn api.main:app --reload --port 8000

Run from the repository root so relative data paths resolve. Startup loads
corpus metadata and precomputed embeddings; ESM-2 weights load lazily on the
first request that actually needs inference.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.routes import corpus, mutation, proteins, search
from api.state import get_state
from ml.sequence import SequenceValidationError


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = get_state()  # loads corpus + embedding store; encoder stays lazy
    print(f"ProteinLens API: {len(state.df)} proteins, poolings {state.store.poolings}")
    yield


app = FastAPI(
    title="ProteinLens API",
    description=(
        "Representation-space analysis of proteins with a frozen ESM-2 encoder: "
        "embeddings, semantic retrieval, mutation perturbation analysis, and "
        "corpus-level geometry."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1024)

cors_origins = os.environ.get(
    "PROTEINLENS_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(SequenceValidationError)
async def sequence_error_handler(_: Request, exc: SequenceValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(KeyError)
async def key_error_handler(_: Request, exc: KeyError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc.args[0]) if exc.args else "Not found"})


@app.exception_handler(FileNotFoundError)
async def artifact_error_handler(_: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": f"Server artifact not built yet: {exc}"},
    )


app.include_router(corpus.router)
app.include_router(proteins.router)
app.include_router(search.router)
app.include_router(mutation.router)
