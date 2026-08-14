"""API contract tests against the synthetic mini-deployment (no model needed)."""

from __future__ import annotations

import os

import pytest


class TestHealth:
    def test_health_reports_corpus(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["corpus_size"] == 12
        assert body["poolings"] == ["mean"]
        assert body["encoder_loaded"] is False


class TestProteinLookup:
    def test_text_search_finds_by_name(self, client):
        hits = client.get("/proteins", params={"q": "hemoglobin"}).json()
        assert hits and hits[0]["accession"] == "T0000"

    def test_text_search_finds_by_gene(self, client):
        hits = client.get("/proteins", params={"q": "TG3"}).json()
        assert any(h["accession"] == "T0003" for h in hits)

    def test_no_match_returns_empty_list(self, client):
        assert client.get("/proteins", params={"q": "zzzzzz"}).json() == []

    def test_profile_contains_stats_and_neighbors(self, client):
        body = client.get("/protein/T0001").json()
        assert body["protein"]["accession"] == "T0001"
        assert body["stats"]["dim"] == 24
        assert body["stats"]["cluster"] == 1
        assert len(body["neighbors"]) == 10
        assert all(n["protein"]["accession"] != "T0001" for n in body["neighbors"])

    def test_unknown_accession_is_404(self, client):
        assert client.get("/protein/NOPE99").status_code == 404


class TestSearch:
    def test_search_by_accession(self, client):
        body = client.post("/search", json={"accession": "T0002", "k": 5}).json()
        assert len(body["hits"]) == 5
        assert body["hits"][0]["similarity"] <= 1.0001
        sims = [h["similarity"] for h in body["hits"]]
        assert sims == sorted(sims, reverse=True)

    def test_requires_exactly_one_query(self, client):
        assert client.post("/search", json={"k": 5}).status_code == 422
        assert client.post(
            "/search", json={"accession": "T0001", "sequence": "MKTVLQ" * 3}
        ).status_code == 422

    def test_unknown_accession_404(self, client):
        assert client.post("/search", json={"accession": "NOPE"}).status_code == 404


class TestCorpus:
    def test_map_payload_served(self, client):
        body = client.get("/map").json()
        assert body["pooling"] == "mean"
        assert len(body["points"]) == 12

    def test_missing_pooling_404(self, client):
        assert client.get("/map", params={"pooling": "max"}).status_code == 404

    def test_benchmark_missing_is_404(self, client):
        assert client.get("/benchmark").status_code == 404


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="model tests download ESM-2 weights; set RUN_MODEL_TESTS=1",
)
class TestModelEndpoints:
    SEQ = (
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHV"
        "DDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
    )

    def test_embed_returns_vector_and_norms(self, client):
        body = client.post(
            "/embed",
            json={"sequence": self.SEQ, "pooling": "mean", "include_residue_norms": True},
        ).json()
        assert body["length"] == len(self.SEQ)
        assert len(body["residue_norms"]) == len(self.SEQ)
        assert body["embedding_norm"] > 0

    def test_embed_rejects_invalid_alphabet(self, client):
        response = client.post("/embed", json={"sequence": "MKTXXBBZZ123"})
        assert response.status_code == 422

    def test_mutation_analysis(self, client):
        body = client.post(
            "/mutation", json={"sequence": self.SEQ, "mutation": "H59Y"}
        ).json()
        assert body["mutation"] == "H59Y"
        assert body["displacement"] > 0
        assert 0.9 < body["cosine_similarity"] <= 1.0
        assert len(body["per_residue_delta"]) == len(self.SEQ)

    def test_mutation_wildtype_mismatch_422(self, client):
        response = client.post(
            "/mutation", json={"sequence": self.SEQ, "mutation": "W59Y"}
        )
        assert response.status_code == 422

    def test_landscape_has_19_effects(self, client):
        body = client.post(
            "/mutation-landscape", json={"sequence": self.SEQ, "position": 59}
        ).json()
        assert body["wildtype"] == "H"
        assert len(body["effects"]) == 19


class TestDomains:
    def test_domains_listed_with_coordinates(self, client):
        body = client.get("/protein/T0000/domains").json()
        assert body["length"] == 60
        assert [d["name"] for d in body["domains"]] == ["EF-hand 1", "EF-hand 2"]
        assert body["domains"][0]["start"] == 5

    def test_protein_without_domains_gets_empty_list(self, client):
        assert client.get("/protein/T0005/domains").json()["domains"] == []

    def test_unknown_accession_404(self, client):
        assert client.get("/protein/NOPE/domains").status_code == 404

    def test_region_search_validates_bounds(self, client):
        response = client.post(
            "/region-search", json={"accession": "T0000", "start": 10, "end": 500}
        )
        assert response.status_code == 422
        assert "outside" in response.json()["detail"]

    def test_region_search_rejects_tiny_span(self, client):
        response = client.post(
            "/region-search", json={"accession": "T0000", "start": 10, "end": 12}
        )
        assert response.status_code == 422
