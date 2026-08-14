"""FAISS-based semantic protein retrieval.

Embeddings are L2-normalized so inner product equals cosine similarity. Three
interchangeable backends sit behind one interface:

  flat   IndexFlatIP — exact; the default up to ~50k vectors
  hnsw   IndexHNSWFlat — graph ANN; sub-millisecond queries past 100k vectors
  ivf    IndexIVFFlat — clustered ANN; smallest memory of the three

``auto_backend`` picks by corpus size; ``scripts/benchmark_ann.py`` measures
the recall/QPS trade-off that justifies the thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from ml.embeddings import l2_normalize

BACKENDS = ("flat", "hnsw", "ivf")

# Exact search stays affordable well past this corpus's size; the graph index
# earns its build cost once brute force stops being interactive.
AUTO_ANN_THRESHOLD = 50_000


def auto_backend(n_vectors: int) -> str:
    return "flat" if n_vectors < AUTO_ANN_THRESHOLD else "hnsw"


@dataclass
class SearchHit:
    accession: str
    score: float  # cosine similarity in [-1, 1]
    rank: int


class ProteinIndex:
    """Cosine-similarity index over one pooling strategy's embeddings."""

    def __init__(
        self,
        index: faiss.Index,
        accessions: list[str],
        pooling: str,
        backend: str = "flat",
    ) -> None:
        self.index = index
        self.accessions = accessions
        self.pooling = pooling
        self.backend = backend
        self.row_of = {acc: i for i, acc in enumerate(accessions)}

    # -- construction ------------------------------------------------------
    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        accessions: list[str],
        pooling: str,
        backend: str = "flat",
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        nlist: int | None = None,
        nprobe: int = 16,
    ) -> ProteinIndex:
        if embeddings.shape[0] != len(accessions):
            raise ValueError("embeddings/accessions length mismatch")
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}; options: {BACKENDS}")
        normalized = l2_normalize(np.ascontiguousarray(embeddings, dtype=np.float32))
        dim = normalized.shape[1]

        if backend == "flat":
            index: faiss.Index = faiss.IndexFlatIP(dim)
        elif backend == "hnsw":
            index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
        else:  # ivf
            nlist = nlist or max(16, int(np.sqrt(len(accessions))) * 4)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(normalized)
            index.nprobe = nprobe

        index.add(normalized)
        if backend == "ivf":
            # reconstruct() needs a direct map on IVF (neighbors_of, kNN stats).
            index.make_direct_map()
        return cls(index, list(accessions), pooling, backend)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / f"index_{self.pooling}.faiss"))
        (directory / f"index_{self.pooling}_accessions.json").write_text(
            json.dumps(self.accessions)
        )
        (directory / f"index_{self.pooling}_meta.json").write_text(
            json.dumps({"backend": self.backend})
        )

    @classmethod
    def load(cls, directory: str | Path, pooling: str) -> ProteinIndex:
        directory = Path(directory)
        index_path = directory / f"index_{pooling}.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"No FAISS index for pooling '{pooling}' in {directory}")
        index = faiss.read_index(str(index_path))
        accessions = json.loads((directory / f"index_{pooling}_accessions.json").read_text())
        meta_path = directory / f"index_{pooling}_meta.json"
        backend = "flat"
        if meta_path.exists():
            backend = json.loads(meta_path.read_text()).get("backend", "flat")
        if backend == "ivf":
            index.make_direct_map()  # not persisted by write_index
        return cls(index, accessions, pooling, backend)

    # -- queries -------------------------------------------------------------
    def search(self, query: np.ndarray, k: int = 10, exclude: str | None = None) -> list[SearchHit]:
        q = l2_normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))
        # +1 so that excluding the query protein itself still yields k hits.
        scores, rows = self.index.search(q, min(k + 1, self.index.ntotal))
        hits: list[SearchHit] = []
        for score, row in zip(scores[0], rows[0], strict=True):
            if row < 0:
                continue
            accession = self.accessions[row]
            if exclude is not None and accession == exclude:
                continue
            hits.append(SearchHit(accession, float(score), len(hits) + 1))
            if len(hits) == k:
                break
        return hits

    def neighbors_of(self, accession: str, k: int = 10) -> list[SearchHit]:
        row = self.row_of.get(accession)
        if row is None:
            raise KeyError(f"{accession} not in index")
        vec = self.index.reconstruct(row)
        return self.search(np.asarray(vec), k, exclude=accession)

    def knn_distances(self, k: int = 10, batch: int = 2048) -> np.ndarray:
        """Mean cosine *distance* (1 − similarity) to each protein's k nearest
        neighbors (self excluded). Basis for density and outlier scores."""
        n = self.index.ntotal
        out = np.zeros(n, dtype=np.float32)
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            block = np.vstack([self.index.reconstruct(i) for i in range(start, stop)])
            scores, rows = self.index.search(block, k + 1)
            for i in range(stop - start):
                mask = rows[i] != (start + i)
                sims = scores[i][mask][:k]
                out[start + i] = float((1.0 - sims).mean())
        return out
