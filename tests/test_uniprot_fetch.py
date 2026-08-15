import pytest

from ml.uniprot_fetch import UniProtFetchError, parse_entry_tsv, validate_accession

SAMPLE_TSV = (
    "Entry\tEntry Name\tProtein names\tGene Names (primary)\tOrganism\t"
    "Organism (ID)\tLength\tSequence\tPfam\tEC number\tKeywords\t"
    "Subcellular location [CC]\tProtein families\n"
    "P99999\tTEST_HUMAN\tTest protein (EC 2.7.11.1) (Synonym)\tTST1\t"
    "Homo sapiens (Human)\t9606\t24\tMKTVHQAAAAWWCHACDEFGHIKL\tPF00069;\t"
    "2.7.11.1\tKinase;ATP-binding\tSUBCELLULAR LOCATION: Nucleus.\t"
    "Protein kinase superfamily, Ser/Thr family\n"
)


class TestValidateAccession:
    def test_accepts_standard_forms(self):
        assert validate_accession(" p69905 ") == "P69905"
        assert validate_accession("A0A023PXB0") == "A0A023PXB0"

    @pytest.mark.parametrize("bad", ["", "p!", "../etc", "P69905; DROP", "X" * 20])
    def test_rejects_malformed(self, bad):
        with pytest.raises(UniProtFetchError):
            validate_accession(bad)


class TestParseEntry:
    def test_full_derivation(self):
        entry = parse_entry_tsv(SAMPLE_TSV)
        assert entry["accession"] == "P99999"
        assert entry["name"] == "Test protein"
        assert entry["gene"] == "TST1"
        assert entry["organism"] == "H. sapiens"
        assert entry["family"] == "Protein kinase superfamily"
        assert entry["pfam"] == "PF00069"
        assert entry["ec_class"] == "Transferase"
        assert entry["localization"] == "Nucleus"
        assert entry["sequence"] == "MKTVHQAAAAWWCHACDEFGHIKL"

    def test_empty_response_rejected(self):
        header_only = SAMPLE_TSV.split("\n")[0] + "\n"
        with pytest.raises(UniProtFetchError, match="empty"):
            parse_entry_tsv(header_only)
