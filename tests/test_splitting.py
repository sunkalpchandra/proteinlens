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
            has_family = fam < 24
            has_pfam = 20 <= fam < 28  # fams 20-23 carry BOTH annotations
            rows.append({
                "accession": f"F{fam:02d}M{member}",
                "family": f"family-{fam}" if has_family else None,
                "pfam_primary": f"PF{fam:05d}" if has_pfam else None,
                "pfam_all": [f"PF{fam:05d}"] if has_pfam else [],
                "sequence": seq,
            })
    return pd.DataFrame(rows)


class TestGrouping:
    def test_every_protein_grouped(self, corpus):
        groups, stats = assign_groups(corpus)
        assert groups.notna().all()
        assert (
            stats["annotated"]
            + stats["unannotated_linked_to_annotated"]
            + stats["unannotated_own_cluster"]
        ) == len(corpus)

    def test_family_members_share_group(self, corpus):
        groups, _ = assign_groups(corpus)
        by_family = corpus.groupby("family").groups
        for indices in by_family.values():
            assert groups.loc[indices].nunique() == 1

    def test_shared_pfam_bridges_family_and_familyless(self):
        """Regression: a family-annotated protein and a family-less protein
        sharing a Pfam domain must land in ONE group (the old tiered fallback
        split them and leaked homologs across splits)."""
        df = pd.DataFrame([
            {"accession": "A1", "family": "Globin family",
             "pfam_all": ["PF00042"], "pfam_primary": "PF00042",
             "sequence": "ACDEFGHIKLMNPQRSTVWY" * 4},
            {"accession": "A2", "family": None,
             "pfam_all": ["PF00042"], "pfam_primary": "PF00042",
             "sequence": "MKTVLQACDEFGHIKLMNPQ" * 4},
        ])
        groups, _ = assign_groups(df)
        assert groups.iloc[0] == groups.iloc[1]

    def test_multidomain_protein_bridges_groups(self):
        """A protein carrying two domains merges both domains' groups."""
        df = pd.DataFrame([
            {"accession": "B1", "family": None, "pfam_primary": "PF00001",
             "pfam_all": ["PF00001"], "sequence": "ACDEFGHIKLMNPQRSTVWY" * 4},
            {"accession": "B2", "family": None, "pfam_primary": "PF00002",
             "pfam_all": ["PF00002"], "sequence": "MKTVLQACDEFGHIKLMNPQ" * 4},
            {"accession": "B3", "family": None, "pfam_primary": "PF00001",
             "pfam_all": ["PF00001", "PF00002"], "sequence": "WYACDEFGHIKLMNPQRSTV" * 4},
        ])
        groups, _ = assign_groups(df)
        assert groups.nunique() == 1

    def test_identical_orphan_joins_annotated_group(self):
        """An unannotated near-duplicate of an annotated protein shares its group."""
        base = "ACDEFGHIKLMNPQRSTVWY" * 5
        df = pd.DataFrame([
            {"accession": "C1", "family": "Some family", "pfam_primary": None,
             "pfam_all": [], "sequence": base},
            {"accession": "C2", "family": None, "pfam_primary": None,
             "pfam_all": [], "sequence": base[:-1] + "A"},
        ])
        groups, stats = assign_groups(df)
        assert groups.iloc[0] == groups.iloc[1]
        assert stats["unannotated_linked_to_annotated"] == 1


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


class TestMMseqsGroups:
    mmseqs_missing = __import__("shutil").which("mmseqs") is None

    @pytest.mark.skipif(mmseqs_missing, reason="mmseqs binary not installed")
    def test_identical_sequences_share_cluster(self):
        from ml.splitting import mmseqs_groups

        base = "MKTVLQAAAAWWCHACDEFGHIKLMNPQRSTVWY" * 3
        df = pd.DataFrame([
            {"accession": "A1", "sequence": base},
            {"accession": "A2", "sequence": base[:-2] + "AA"},   # ~98% identical
            {"accession": "B1", "sequence": "GPGPGPWWNNQQRRKKDDEEFFYYHH" * 4},
        ])
        groups, stats = mmseqs_groups(df)
        assert groups.iloc[0] == groups.iloc[1]
        assert groups.iloc[0] != groups.iloc[2]
        assert stats["method"] == "mmseqs"

    @pytest.mark.skipif(mmseqs_missing, reason="mmseqs binary not installed")
    def test_make_splits_accepts_mmseqs_method(self, corpus):
        splits, summary = make_splits(corpus, seed=0, method="mmseqs")
        assert summary["method"] == "mmseqs"
        assert set(splits.unique()) <= {"train", "val", "test"}

    def test_unknown_method_rejected(self, corpus):
        with pytest.raises(ValueError, match="method"):
            make_splits(corpus, method="blastclust")
