from models.prost_encoder import AA_PREFIX, prost_preprocess


class TestProstPreprocess:
    def test_spaces_and_prefix(self):
        assert prost_preprocess("MKT") == f"{AA_PREFIX} M K T"

    def test_maps_noncanonical_to_x(self):
        assert prost_preprocess("MUZOB") == f"{AA_PREFIX} M X X X X"

    def test_uppercases(self):
        assert prost_preprocess("mkt") == f"{AA_PREFIX} M K T"
