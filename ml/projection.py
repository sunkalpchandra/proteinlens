"""2D projection of protein embeddings: PCA → UMAP, with on-disk caching.

UMAP is never run on raw high-dimensional embeddings: PCA first reduces to
``pca_dim`` components (denoises and makes neighbor graphs cheaper), then UMAP
produces 2D coordinates. Every parameter set is cached under a content hash so
the API can serve alternative projections without recomputation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ProjectionParams:
    pooling: str = "mean"
    pca_dim: int = 50
    n_neighbors: int = 15
    min_dist: float = 0.1
    seed: int = 42

    def cache_key(self, data_fingerprint: str) -> str:
        payload = json.dumps({**asdict(self), "data": data_fingerprint}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def compute_projection(
    embeddings: np.ndarray, params: ProjectionParams
) -> tuple[np.ndarray, dict]:
    """Returns ([N, 2] float32 coordinates, info dict with PCA variance)."""
    from sklearn.decomposition import PCA
    from umap import UMAP

    x = np.asarray(embeddings, dtype=np.float32)
    pca_dim = min(params.pca_dim, x.shape[1], x.shape[0])
    pca = PCA(n_components=pca_dim, random_state=params.seed)
    reduced = pca.fit_transform(x)

    reducer = UMAP(
        n_components=2,
        n_neighbors=params.n_neighbors,
        min_dist=params.min_dist,
        metric="cosine",
        random_state=params.seed,
    )
    coords = reducer.fit_transform(reduced).astype(np.float32)
    info = {
        "pca_dim": pca_dim,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "n": int(x.shape[0]),
    }
    return coords, info


class ProjectionCache:
    def __init__(self, directory: str | Path = "data/index/projections") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def load_or_compute(
        self,
        embeddings: np.ndarray,
        params: ProjectionParams,
        data_fingerprint: str,
    ) -> tuple[np.ndarray, dict, bool]:
        """Returns (coords, info, cache_hit)."""
        key = params.cache_key(data_fingerprint)
        path = self.directory / f"projection_{key}.npz"
        if path.exists():
            payload = np.load(path, allow_pickle=False)
            info = json.loads(str(payload["info"]))
            return payload["coords"], info, True
        coords, info = compute_projection(embeddings, params)
        np.savez_compressed(
            path,
            coords=coords,
            info=np.array(json.dumps({**info, **asdict(params)})),
        )
        return coords, {**info, **asdict(params)}, False
