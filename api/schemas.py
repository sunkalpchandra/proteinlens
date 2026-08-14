"""Pydantic schemas for the ProteinLens API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Pooling = Literal["mean", "max", "bos", "attention"]

MAX_SEQUENCE_LENGTH = 2048


class EmbedRequest(BaseModel):
    sequence: str = Field(..., min_length=1, max_length=MAX_SEQUENCE_LENGTH * 2)
    pooling: Pooling = "mean"
    include_residue_norms: bool = False
    include_attention: bool = False


class EmbedResponse(BaseModel):
    embedding: list[float]
    length: int
    model: str
    pooling: Pooling
    embedding_version: str
    cache_hit: bool
    embedding_norm: float
    residue_norms: list[float] | None = None
    attention_weights: list[float] | None = None


class SearchRequest(BaseModel):
    """Search by raw sequence OR by corpus accession (exactly one)."""

    sequence: str | None = Field(None, max_length=MAX_SEQUENCE_LENGTH * 2)
    accession: str | None = None
    pooling: Pooling = "mean"
    k: int = Field(10, ge=1, le=100)

    @model_validator(mode="after")
    def exactly_one_query(self) -> SearchRequest:
        if bool(self.sequence) == bool(self.accession):
            raise ValueError("Provide exactly one of 'sequence' or 'accession'.")
        return self


class ProteinSummary(BaseModel):
    accession: str
    name: str
    gene: str | None
    organism: str
    length: int
    family: str | None
    pfam: str | None
    ec_class: str | None
    localization: str | None


class SearchHitOut(BaseModel):
    rank: int
    similarity: float
    protein: ProteinSummary


class SearchResponse(BaseModel):
    pooling: Pooling
    query_length: int
    hits: list[SearchHitOut]


class RepresentationStats(BaseModel):
    embedding_norm: float
    dim: int
    nn_distance: float | None
    knn_mean_distance: float | None
    cluster: int | None
    outlier_score: float | None
    x: float | None
    y: float | None


class ProteinProfile(BaseModel):
    protein: ProteinSummary
    protein_name_full: str
    keywords: list[str]
    sequence: str
    model: str
    stats: RepresentationStats
    neighbors: list[SearchHitOut]


class MutationRequest(BaseModel):
    accession: str | None = None
    sequence: str | None = Field(None, max_length=MAX_SEQUENCE_LENGTH * 2)
    mutation: str = Field(..., examples=["H63Y"])
    pooling: Pooling = "mean"

    @model_validator(mode="after")
    def exactly_one_target(self) -> MutationRequest:
        if bool(self.sequence) == bool(self.accession):
            raise ValueError("Provide exactly one of 'sequence' or 'accession'.")
        return self


class MutationResponse(BaseModel):
    mutation: str
    pooling: Pooling
    displacement: float
    relative_displacement: float
    cosine_similarity: float
    local_delta: float
    global_residue_delta: float
    per_residue_delta: list[float]
    top_dimensions: list[dict]
    note: str = (
        "Representation-space perturbation of a frozen protein language model; "
        "not a fitness, stability, or pathogenicity prediction."
    )


class LandscapeRequest(BaseModel):
    accession: str | None = None
    sequence: str | None = Field(None, max_length=MAX_SEQUENCE_LENGTH * 2)
    position: int = Field(..., ge=1)
    pooling: Pooling = "mean"

    @model_validator(mode="after")
    def exactly_one_target(self) -> LandscapeRequest:
        if bool(self.sequence) == bool(self.accession):
            raise ValueError("Provide exactly one of 'sequence' or 'accession'.")
        return self


class LandscapeEffect(BaseModel):
    mutant: str
    mutation: str
    displacement: float
    cosine_similarity: float
    local_delta: float


class LandscapeResponse(BaseModel):
    position: int
    wildtype: str
    pooling: Pooling
    effects: list[LandscapeEffect]
    max_displacement: str
    min_displacement: str
    note: str = (
        "Representation-space perturbation of a frozen protein language model; "
        "not a fitness, stability, or pathogenicity prediction."
    )


class AttentionResponse(BaseModel):
    accession: str
    length: int
    weights: list[float]
    top_positions: list[int]
    note: str = (
        "Learned attention-pooling weights: a model-dependent interpretability "
        "signal, not a functional-residue annotation."
    )


class HealthResponse(BaseModel):
    status: str
    model: str
    corpus_size: int
    poolings: list[str]
    device: str
    encoder_loaded: bool


class DomainOut(BaseModel):
    name: str
    start: int  # 1-based, inclusive
    end: int


class DomainsResponse(BaseModel):
    accession: str
    length: int
    domains: list[DomainOut]
    note: str = (
        "UniProt-curated DOMAIN features; coverage is partial (~30% of the "
        "corpus carries curated domain coordinates)."
    )


class RegionSearchRequest(BaseModel):
    accession: str
    start: int = Field(..., ge=1)
    end: int = Field(..., ge=1)
    k: int = Field(10, ge=1, le=100)


class RegionSearchResponse(BaseModel):
    accession: str
    start: int
    end: int
    span_length: int
    hits: list[SearchHitOut]
    note: str = (
        "A mean-pooled residue-span embedding queried against protein-level "
        "mean embeddings — a cross-granularity cosine comparison, not a "
        "domain-database search."
    )


class TrajectoryRequest(BaseModel):
    accession: str | None = None
    sequence: str | None = Field(None, max_length=MAX_SEQUENCE_LENGTH * 2)
    mutations: list[str] = Field(..., min_length=1, max_length=10,
                                 examples=[["E7V", "V7A"]])
    pooling: Pooling = "mean"

    @model_validator(mode="after")
    def exactly_one_target(self) -> TrajectoryRequest:
        if bool(self.sequence) == bool(self.accession):
            raise ValueError("Provide exactly one of 'sequence' or 'accession'.")
        return self


class TrajectoryStep(BaseModel):
    step: int
    mutation: str
    cumulative: list[str]
    step_displacement: float
    displacement_from_wt: float
    cosine_to_wt: float


class TrajectoryResponse(BaseModel):
    pooling: Pooling
    n_steps: int
    steps: list[TrajectoryStep]
    path_length: float
    net_displacement: float
    directness: float
    note: str = (
        "Sequential representation-space movement of a frozen protein language "
        "model; not an evolutionary path or fitness trajectory."
    )


class CompareRequest(BaseModel):
    a: str
    b: str

    @model_validator(mode="after")
    def distinct(self) -> CompareRequest:
        if self.a == self.b:
            raise ValueError("Choose two different proteins to compare.")
        return self


class CompareResponse(BaseModel):
    a: ProteinSummary
    b: ProteinSummary
    cosine_by_pooling: dict[str, float]
    sequence_identity: float
    same_family: bool
    shared_pfam: list[str]
    a_domains: list[DomainOut]
    b_domains: list[DomainOut]
    note: str = (
        "Embedding cosine is representation similarity under a frozen model; "
        "high cosine at low sequence identity marks representation-space "
        "neighbors, not evidence of homology or shared function."
    )
