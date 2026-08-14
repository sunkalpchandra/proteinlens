"""Shared fixtures: a fully synthetic mini-deployment of ProteinLens.

Builds a 12-protein corpus with random embeddings, a FAISS index, and a map
payload in a temp directory, then wires an ``AppState`` to it. API routes that
don't need the language model are tested against this; model-dependent routes
are exercised only when RUN_MODEL_TESTS=1 (they download ESM-2 weights).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml.retrieval import ProteinIndex

DIM = 24
N = 12


@pytest.fixture(scope="session")
def synthetic_state(tmp_path_factory):
    root = tmp_path_factory.mktemp("plens")
    processed = root / "processed"
    embeddings = root / "embeddings"
    index_dir = root / "index"
    reports = root / "reports"
    for d in (processed, embeddings, index_dir, reports):
        d.mkdir()

    rng = np.random.default_rng(11)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    rows = []
    for i in range(N):
        rows.append({
            "accession": f"T{i:04d}",
            "entry_name": f"TEST{i}_HUMAN",
            "protein_name": f"Test protein {i}" if i else "Hemoglobin subunit test",
            "protein_name_full": f"Test protein {i} (full)",
            "gene": f"TG{i}",
            "organism": "Homo sapiens (Human)",
            "organism_short": "H. sapiens",
            "taxon_id": "9606",
            "length": 60,
            "sequence": "".join(rng.choice(aas, size=60)),
            "family": "Globin family" if i < 4 else None,
            "pfam_all": ["PF00042"] if i < 4 else [],
            "pfam_primary": "PF00042" if i < 4 else None,
            "ec": None,
            "ec_class": "Hydrolase" if i % 3 == 0 else None,
            "is_enzyme": i % 3 == 0,
            "keywords": "Oxygen transport;Heme",
            "localization": "Cytoplasm",
            "subcellular_location": "SUBCELLULAR LOCATION: Cytoplasm.",
        })
    df = pd.DataFrame(rows)
    df.to_parquet(processed / "proteins.parquet", index=False)

    matrix = rng.normal(size=(N, DIM)).astype(np.float32)
    np.save(embeddings / "corpus_mean.npy", matrix)
    accessions = df["accession"].tolist()
    (embeddings / "accessions.json").write_text(json.dumps(accessions))
    (embeddings / "store_meta.json").write_text(json.dumps({
        "model": "synthetic/test-model",
        "embedding_version": "1",
        "dim": DIM,
        "poolings": ["mean"],
        "n_proteins": N,
        "corpus_sha256_16": "deadbeefdeadbeef",
    }))

    ProteinIndex.build(matrix, accessions, "mean").save(index_dir)

    points = [{
        "id": acc, "name": df["protein_name"].iat[i], "gene": df["gene"].iat[i],
        "org": "H. sapiens", "len": 60, "family": df["family"].iat[i],
        "pfam": df["pfam_primary"].iat[i], "ec": df["ec_class"].iat[i],
        "enzyme": bool(df["is_enzyme"].iat[i]), "loc": "Cytoplasm",
        "x": float(i), "y": float(-i), "cluster": i % 3,
        "knn_dist": 0.1, "outlier": i / (N - 1),
    } for i, acc in enumerate(accessions)]
    (index_dir / "map_mean.json").write_text(json.dumps({
        "pooling": "mean", "model": "synthetic/test-model",
        "projection": {"pca_dim": 8}, "clustering": {"n_clusters": 3},
        "points": points,
    }))

    domains = pd.DataFrame([
        {"accession": "T0000", "name": "EF-hand 1", "start": 5, "end": 25},
        {"accession": "T0000", "name": "EF-hand 2", "start": 30, "end": 55},
    ])
    domains.to_parquet(processed / "domains.parquet", index=False)

    from api.state import AppState

    return AppState(
        corpus_path=processed / "proteins.parquet",
        embeddings_dir=embeddings,
        index_dir=index_dir,
        reports_dir=reports,
        domains_path=processed / "domains.parquet",
    )


@pytest.fixture(scope="session")
def client(synthetic_state):
    from fastapi.testclient import TestClient

    from api.main import app
    from api.state import get_state

    app.dependency_overrides[get_state] = lambda: synthetic_state
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
