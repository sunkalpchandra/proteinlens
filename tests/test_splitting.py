import numpy as np
import pandas as pd
import pytest

from ml.splitting import assign_groups, make_splits


@pytest.fixture
def corpus() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for fam in range(30):
        for member in range(rng.integers(2, 12)):
            seq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=80))
            rows.append({
                "accession": f"F{fam:02d}M{member}",
                "family": f"family-{fam}" if fam < 24 else None,
                "pfam_primary": f"PF{fam:05d}" if 24 <= fam < 28 else None,
                "sequence": seq,
            })
    return pd.DataFrame(rows)


class TestGrouping:
    def test_every_protein_grouped(self, corpus):
        groups, stats = assign_groups(corpus)
        assert groups.notna().all()
        assert stats["by_family"] + stats["by_pfam"] + stats["by_kmer_cluster"] == len(corpus)

    def test_family_members_share_group(self, corpus):
        groups, _ = assign_groups(corpus)
        by_family = corpus.groupby("family").groups
        for indices in by_family.values():
            assert groups.loc[indices].nunique() == 1


class TestSplits:
    def test_groups_never_straddle_splits(self, corpus):
        splits, _ = make_splits(corpus, seed=0)
        groups, _ = assign_groups(corpus)
        frame = pd.DataFrame({"group": groups, "split": splits})
        assert (frame.groupby("group")["split"].nunique() == 1).all()

    def test_ratios_approximately_honored(self, corpus):
        splits, _ = make_splits(corpus, ratios=(0.7, 0.15, 0.15), seed=0)
        fractions = splits.value_counts(normalize=True)
        assert fractions["train"] == pytest.approx(0.7, abs=0.15)
        assert fractions["test"] == pytest.approx(0.15, abs=0.12)

    def test_deterministic_given_seed(self, corpus):
        a, _ = make_splits(corpus, seed=5)
        b, _ = make_splits(corpus, seed=5)
        assert a.equals(b)
