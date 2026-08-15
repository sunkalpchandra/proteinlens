"""Application state: corpus, embedding store, indexes, and the (lazy) encoder.

Everything cheap loads at startup; the ESM-2 encoder loads on first use so that
metadata-only deployments (browse/map/search-by-accession) never pay for model
weights. A threading lock serializes encoder access — model inference is the
only mutable, non-thread-safe resource.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pandas as pd

from ml.embeddings import EmbeddingPipeline, EmbeddingStore
from ml.retrieval import ProteinIndex
from models.mutation import MutationAnalyzer
from models.scoring import MaskedLMScorer


class AppState:
    def __init__(
        self,
        corpus_path: str | Path = "data/processed/proteins.parquet",
        embeddings_dir: str | Path = "data/embeddings",
        index_dir: str | Path = "data/index",
        reports_dir: str | Path = "reports",
        domains_path: str | Path = "data/processed/domains.parquet",
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.embeddings_dir = Path(embeddings_dir)
        self.index_dir = Path(index_dir)
        self.reports_dir = Path(reports_dir)
        self.domains_path = Path(domains_path)
        self._domains: pd.DataFrame | None = None
        self.model_name = os.environ.get("PROTEINLENS_MODEL", "facebook/esm2_t12_35M_UR50D")

        self.df = pd.read_parquet(self.corpus_path)
        self.by_accession = self.df.set_index("accession", drop=False)
        self.store = EmbeddingStore(self.embeddings_dir)

        self._indexes: dict[str, ProteinIndex] = {}
        self._maps: dict[str, dict] = {}
        self._map_points_by_id: dict[str, dict[str, dict]] = {}
        self._pipeline: EmbeddingPipeline | None = None
        self._analyzer: MutationAnalyzer | None = None
        self._scorer: MaskedLMScorer | None = None
        self.encoder_lock = threading.Lock()
        self._init_lock = threading.Lock()

    # -- lazy resources ----------------------------------------------------
    def index(self, pooling: str) -> ProteinIndex:
        if pooling not in self._indexes:
            self._indexes[pooling] = ProteinIndex.load(self.index_dir, pooling)
        return self._indexes[pooling]

    def map_payload(self, pooling: str) -> dict:
        if pooling not in self._maps:
            path = self.index_dir / f"map_{pooling}.json"
            if not path.exists():
                raise FileNotFoundError(f"No map payload for pooling '{pooling}'")
            self._maps[pooling] = json.loads(path.read_text())
        return self._maps[pooling]

    def map_point(self, pooling: str, accession: str) -> dict | None:
        if pooling not in self._map_points_by_id:
            payload = self.map_payload(pooling)
            self._map_points_by_id[pooling] = {p["id"]: p for p in payload["points"]}
        return self._map_points_by_id[pooling].get(accession)

    @property
    def pipeline(self) -> EmbeddingPipeline:
        # Double-checked locking: two concurrent cold requests must not both
        # load the ESM-2 checkpoint.
        if self._pipeline is None:
            with self._init_lock:
                if self._pipeline is None:
                    self._pipeline = EmbeddingPipeline(
                        model_name=self.model_name,
                        cache_path=self.embeddings_dir / "adhoc_cache.sqlite",
                        attention_pooler_path=self.embeddings_dir / "attention_pooler.pt",
                    )
        return self._pipeline

    @property
    def analyzer(self) -> MutationAnalyzer:
        if self._analyzer is None:
            self._analyzer = MutationAnalyzer(self.pipeline)
        return self._analyzer

    @property
    def scorer(self) -> MaskedLMScorer:
        if self._scorer is None:
            with self._init_lock:
                if self._scorer is None:
                    self._scorer = MaskedLMScorer(self.model_name)
        return self._scorer

    @property
    def encoder_loaded(self) -> bool:
        return self._pipeline is not None

    def _domains_frame(self) -> pd.DataFrame:
        if self._domains is None:
            if self.domains_path.exists():
                self._domains = pd.read_parquet(self.domains_path)
            else:
                self._domains = pd.DataFrame(columns=["accession", "name", "start", "end"])
        return self._domains

    def domains_for(self, accession: str) -> list[dict]:
        """UniProt-curated DOMAIN features; empty when no data file exists."""
        frame = self._domains_frame()
        rows = frame[frame["accession"] == accession]
        return [{"name": r.name_, "start": int(r.start), "end": int(r.end)}
                for r in rows.rename(columns={"name": "name_"}).itertuples()]

    def n_proteins_with_domains(self) -> int:
        # Intersect with the corpus: a stale domains file must not count
        # proteins that a rebuilt corpus no longer contains.
        frame = self._domains_frame()
        in_corpus = frame["accession"].isin(self.by_accession.index)
        return int(frame.loc[in_corpus, "accession"].nunique())

    # -- helpers ----------------------------------------------------------
    def protein_row(self, accession: str) -> pd.Series:
        try:
            return self.by_accession.loc[accession]
        except KeyError as exc:
            raise KeyError(f"Unknown accession '{accession}'") from exc

    def summary_dict(self, accession: str) -> dict:
        row = self.protein_row(accession)
        return {
            "accession": row["accession"],
            "name": row["protein_name"],
            "gene": row["gene"] if isinstance(row["gene"], str) else None,
            "organism": row["organism_short"],
            "length": int(row["length"]),
            "family": row["family"] if isinstance(row["family"], str) else None,
            "pfam": row["pfam_primary"] if isinstance(row["pfam_primary"], str) else None,
            "ec_class": row["ec_class"] if isinstance(row["ec_class"], str) else None,
            "localization": row["localization"] if isinstance(row["localization"], str) else None,
        }


_state: AppState | None = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state
