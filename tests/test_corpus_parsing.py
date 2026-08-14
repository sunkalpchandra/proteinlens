import pandas as pd

from ml.corpus import (
    SHOWCASE_ACCESSIONS,
    clean_protein_name,
    derive_fields,
    parse_ec_class,
    parse_family,
    parse_localization,
    parse_pfam,
    short_organism,
)


class TestParsers:
    def test_family_takes_first_segment(self):
        assert parse_family("Globin family") == "Globin family"
        assert parse_family("TRAFAC class GTPase superfamily, Bms1 family") == (
            "TRAFAC class GTPase superfamily"
        )
        assert parse_family(None) is None
        assert parse_family("   ") is None

    def test_pfam_splits_and_strips(self):
        assert parse_pfam("PF00042;PF00043;") == ["PF00042", "PF00043"]
        assert parse_pfam(float("nan")) == []

    def test_ec_class_maps_top_digit(self):
        assert parse_ec_class("2.7.11.1") == "Transferase"
        assert parse_ec_class("7.1.1.-; 1.6.5.3") == "Translocase"
        assert parse_ec_class("9.9.9.9") is None
        assert parse_ec_class(None) is None

    def test_localization_priority_order(self):
        text = "SUBCELLULAR LOCATION: Secreted. Note=also in cytoplasm"
        assert parse_localization(text) == "Secreted"
        assert parse_localization("SUBCELLULAR LOCATION: Cytoplasm.") == "Cytoplasm"
        assert parse_localization("nothing recognizable") is None

    def test_protein_name_keeps_head(self):
        assert clean_protein_name("Hemoglobin subunit alpha (Alpha-globin) (EC 1.1)") == (
            "Hemoglobin subunit alpha"
        )

    def test_short_organism(self):
        assert short_organism("Homo sapiens (Human)") == "H. sapiens"
        assert short_organism("Escherichia coli (strain K12)") == "E. coli"


class TestDeriveFields:
    def test_end_to_end_derivation(self):
        df = pd.DataFrame([{
            "accession": "P00001",
            "protein_name": "Test kinase (EC 2.7.11.1)",
            "protein_families": "Protein kinase superfamily, Ser/Thr family",
            "pfam": "PF00069;",
            "ec": "2.7.11.1",
            "subcellular_location": "SUBCELLULAR LOCATION: Nucleus.",
            "organism": "Homo sapiens (Human)",
            "length": "300",
        }])
        out = derive_fields(df)
        row = out.iloc[0]
        assert row["protein_name"] == "Test kinase"
        assert row["protein_name_full"].startswith("Test kinase (EC")
        assert row["family"] == "Protein kinase superfamily"
        assert row["pfam_primary"] == "PF00069"
        assert row["ec_class"] == "Transferase"
        assert bool(row["is_enzyme"]) is True
        assert row["localization"] == "Nucleus"
        assert row["organism_short"] == "H. sapiens"
        assert row["length"] == 300


class TestShowcase:
    def test_showcase_list_is_nonempty_and_unique(self):
        assert len(SHOWCASE_ACCESSIONS) >= 8
        assert len(set(SHOWCASE_ACCESSIONS)) == len(SHOWCASE_ACCESSIONS)
