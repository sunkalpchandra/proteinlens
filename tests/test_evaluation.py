import numpy as np
import pandas as pd

from ml.evaluation import (
    clustering_agreement,
    perturbation_pairs,
    retrieval_precision_at_k,
    stability_from_vectors,
)


def clustered_embeddings(n_per: int = 20, n_classes: int = 4, dim: int = 16):
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(n_classes, dim)) * 6
    labels = np.repeat(np.arange(n_classes), n_per)
    points = centers[labels] + 0.2 * rng.normal(size=(n_per * n_classes, dim))
    return points.astype(np.float32), pd.Series([f"fam-{c}" for c in labels])


class TestRetrievalPrecision:
    def test_well_separated_classes_score_high(self):
        embeddings, labels = clustered_embeddings()
        metrics = retrieval_precision_at_k(embeddings, labels)
        assert metrics["precision@1"] > 0.95
        assert metrics["n_evaluated"] == len(labels)

    def test_unlabeled_proteins_stay_in_the_candidate_pool(self):
        """Regression (review finding): removing unlabeled proteins from the
        index inflates precision. An unlabeled near-duplicate planted next to
        every query must depress precision@1."""
        embeddings, labels = clustered_embeddings(n_per=6)
        noisy = np.concatenate([embeddings, embeddings + 1e-4])
        noisy_labels = pd.concat(
            [labels, pd.Series([None] * len(labels))], ignore_index=True
        )
        metrics = retrieval_precision_at_k(noisy, noisy_labels)
        assert metrics["n_index"] == len(noisy)          # pool includes unlabeled
        assert metrics["n_evaluated"] == len(labels)     # queries exclude them
        assert metrics["precision@1"] < 0.2              # duplicates steal rank 1

    def test_singleton_labels_not_queried(self):
        embeddings, labels = clustered_embeddings(n_per=2)
        labels.iloc[0] = "one-off"
        metrics = retrieval_precision_at_k(embeddings, labels)
        assert metrics["n_evaluated"] == len(labels) - 2  # singleton + its orphaned pair


class TestClusteringAgreement:
    def test_separable_classes_give_high_nmi(self):
        embeddings, labels = clustered_embeddings()
        metrics = clustering_agreement(embeddings, labels, n_clusters=4)
        assert metrics["nmi"] > 0.9

    def test_null_categories_excluded(self):
        embeddings, labels = clustered_embeddings()
        labels.iloc[:10] = None
        metrics = clustering_agreement(embeddings, labels, n_clusters=4)
        assert np.isfinite(metrics["nmi"])


class TestStability:
    def test_pairs_are_single_substitutions(self):
        sequences = ["ACDEFGHIKLMNPQRSTVWY" * 3] * 30
        for idx, wt, mut in perturbation_pairs(sequences, n=10, seed=1):
            assert wt == sequences[idx]
            diffs = [i for i, (a, b) in enumerate(zip(wt, mut, strict=True)) if a != b]
            assert len(diffs) == 1

    def test_identical_vectors_have_unit_cosine(self):
        x = np.random.default_rng(2).normal(size=(8, 12)).astype(np.float32)
        metrics = stability_from_vectors(x, x.copy())
        assert metrics["cosine_mean"] > 1.0 - 1e-5  # float32 normalization
        assert metrics["cosine_min"] > 1.0 - 1e-5
