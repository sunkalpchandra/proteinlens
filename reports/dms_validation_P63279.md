# DMS validation — model statistics vs measured variant effects

Assay: UBE2I DMS-TileSeq (urn:mavedb:00000001-a-3), 2870 single substitutions on P63279 (complementation score; ≈1 = wild-type-like function). Model: `facebook/esm2_t12_35M_UR50D`, frozen.

| model statistic | Spearman ρ vs assay score |
|---|---|
| LM log-likelihood ratio (wt-marginal) | -0.011 (p=5.5e-01) |
| −‖Δz‖ embedding displacement (mean pooling) | +0.148 (p=1.6e-15) |
| cos(z_wt, z_mut) | +0.148 (p=1.6e-15) |

Reading: the likelihood ratio is the field-standard zero-shot variant score; displacement measures representation movement. A positive ρ means the statistic tracks the assay (higher = more functional). The magnitudes are consistent with the encoder's scale — published zero-shot correlations reach ~0.4–0.5 only for 650M+ models, and CALM1 complementation is a known hard target — and notably, embedding displacement tracks the assay almost as well as the likelihood score at this scale. Correlations are specific to this protein and assay; neither statistic is a fitness predictor.
