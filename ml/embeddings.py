"""Embedding pipeline: encoder + pooling + persistent caching.

Two storage layers with different jobs:

  * ``EmbeddingStore`` — consolidated ``.npy`` matrices for the precomputed
    corpus (one matrix per pooling strategy, rows aligned with an accession
    index). Read-only at serving time.
  * ``SqliteVectorCache`` — persistent cache for ad-hoc requests (user-pasted
    sequences, mutants). Keys are ``sha256(model | pooling | sequence)``, so a
    cache entry is invalidated automatically by changing model or pooling.

An in-process LRU keeps recently used residue-level encodings warm for the API
(mutation landscapes hit the same wild-type repeatedly).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from models.encoder import DEFAULT_MODEL, EncodedProtein, ESM2Encoder
from models.pooling import AttentionPooling, Pooler
from ml.sequence import validate_sequence

EMBEDDING_VERSION = "1"


def embedding_cache_key(sequence: str, model_name: str, pooling: str) -> str:
    payload = f"{model_name}|{pooling}|v{EMBEDDING_VERSION}|{sequence}"
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class EmbeddingResult:
    protein_id: str | None
    embedding: np.ndarray                    # [D] float32
    length: int
    model: str
    pooling: str
    embedding_version: str
    cache_hit: bool
    residue_embeddings: np.ndarray | None = None  # [L, D] when requested
    attention_weights: np.ndarray | None = None   # [L] for attention pooling


class SqliteVectorCache:
    """Tiny persistent key→vector store (thread-safe, WAL mode)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            " key TEXT PRIMARY KEY, dim INTEGER NOT NULL,"
            " vector BLOB NOT NULL, created REAL NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT dim, vector FROM vectors WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        dim, blob = row
        return np.frombuffer(blob, dtype=np.float32, count=dim).copy()

    def put(self, key: str, vector: np.ndarray) -> None:
        vec = np.ascontiguousarray(vector, dtype=np.float32)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO vectors (key, dim, vector, created) VALUES (?,?,?,?)",
                (key, vec.shape[0], vec.tobytes(), time.time()),
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            (n,) = self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
        return int(n)


class ResidueLRU:
    """In-memory LRU of residue-level encodings keyed by sequence hash."""

    def __init__(self, maxsize: int = 16) -> None:
        self.maxsize = maxsize
        self._store: OrderedDict[str, EncodedProtein] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> EncodedProtein | None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
        return None

    def put(self, key: str, value: EncodedProtein) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)


class EmbeddingPipeline:
    """sequence → validate → encode → pool, with caching at each level."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        cache_path: str | Path | None = "data/embeddings/adhoc_cache.sqlite",
        attention_pooler_path: str | Path | None = "data/embeddings/attention_pooler.pt",
        residue_lru_size: int = 16,
    ) -> None:
        self.encoder = ESM2Encoder(model_name, device=device)
        pooler_path = Path(attention_pooler_path) if attention_pooler_path else None
        attention = (
            AttentionPooling.load(pooler_path) if pooler_path and pooler_path.exists() else None
        )
        self.pooler = Pooler(attention)
        self.cache = SqliteVectorCache(cache_path) if cache_path else None
        self.residue_lru = ResidueLRU(residue_lru_size)

    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return self.encoder.model_name

    def _sequence_key(self, sequence: str) -> str:
        return hashlib.sha256(f"{self.model_name}|{sequence}".encode()).hexdigest()

    def encode_residues(self, sequence: str) -> EncodedProtein:
        """Residue-level encoding with an in-memory LRU (no disk persistence)."""
        key = self._sequence_key(sequence)
        cached = self.residue_lru.get(key)
        if cached is not None:
            return cached
        encoded = self.encoder.encode(sequence)
        self.residue_lru.put(key, encoded)
        return encoded

    def embed(
        self,
        sequence: str,
        pooling: str = "mean",
        protein_id: str | None = None,
        with_residues: bool = False,
    ) -> EmbeddingResult:
        seq = validate_sequence(sequence)
        cache_key = embedding_cache_key(seq, self.model_name, pooling)

        if not with_residues and self.cache is not None:
            hit = self.cache.get(cache_key)
            if hit is not None:
                return EmbeddingResult(
                    protein_id=protein_id,
                    embedding=hit,
                    length=len(seq),
                    model=self.model_name,
                    pooling=pooling,
                    embedding_version=EMBEDDING_VERSION,
                    cache_hit=True,
                )

        encoded = self.encode_residues(seq)
        pooled, attn = self.pooler.pool(encoded.residue_embeddings, encoded.bos_embedding, pooling)
        embedding = pooled.numpy().astype(np.float32)
        if self.cache is not None:
            self.cache.put(cache_key, embedding)

        return EmbeddingResult(
            protein_id=protein_id,
            embedding=embedding,
            length=len(seq),
            model=self.model_name,
            pooling=pooling,
            embedding_version=EMBEDDING_VERSION,
            cache_hit=False,
            residue_embeddings=encoded.residue_embeddings.numpy() if with_residues else None,
            attention_weights=attn.numpy() if attn is not None else None,
        )

    def embed_batch(
        self, sequences: list[str], pooling: str = "mean"
    ) -> list[np.ndarray]:
        """Pooled embeddings for many sequences (no per-item caching)."""
        encoded = self.encoder.encode_batch([validate_sequence(s) for s in sequences])
        out = []
        for enc in encoded:
            pooled, _ = self.pooler.pool(enc.residue_embeddings, enc.bos_embedding, pooling)
            out.append(pooled.numpy().astype(np.float32))
        return out


class EmbeddingStore:
    """Read-only access to precomputed corpus embeddings.

    Layout under ``data/embeddings/``:
        corpus_{pooling}.npy   float32 [N, D], rows aligned with accessions.json
        accessions.json        ["P69905", ...]
        store_meta.json        model name, embedding version, poolings, dims
    """

    def __init__(self, directory: str | Path = "data/embeddings") -> None:
        self.directory = Path(directory)
        meta_path = self.directory / "store_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No embedding store at {self.directory}. Run scripts/precompute_embeddings.py."
            )
        self.meta = json.loads(meta_path.read_text())
        self.accessions: list[str] = json.loads((self.directory / "accessions.json").read_text())
        self.row_of = {acc: i for i, acc in enumerate(self.accessions)}
        self._matrices: dict[str, np.ndarray] = {}

    @property
    def poolings(self) -> list[str]:
        return self.meta["poolings"]

    def matrix(self, pooling: str) -> np.ndarray:
        if pooling not in self._matrices:
            path = self.directory / f"corpus_{pooling}.npy"
            if not path.exists():
                raise FileNotFoundError(f"No precomputed matrix for pooling '{pooling}'")
            self._matrices[pooling] = np.load(path, mmap_mode="r")
        return self._matrices[pooling]

    def vector(self, accession: str, pooling: str) -> np.ndarray:
        row = self.row_of.get(accession)
        if row is None:
            raise KeyError(f"Accession {accession} not in embedding store")
        return np.asarray(self.matrix(pooling)[row], dtype=np.float32)


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization (cosine similarity becomes inner product)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        return matrix / max(float(np.linalg.norm(matrix)), eps)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)
