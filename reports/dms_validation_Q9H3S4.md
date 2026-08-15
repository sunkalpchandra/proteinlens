# DMS validation — model statistics vs measured variant effects

Assay: TPK1 DMS-TileSeq (urn:mavedb:00001251-a-1), 4123 single substitutions on Q9H3S4 (complementation score; ≈1 = wild-type-like function). Model: `facebook/esm2_t12_35M_UR50D`, frozen.

| model statistic | Spearman ρ vs assay score |
|---|---|
| LM log-likelihood ratio (wt-marginal) | +0.154 (p=2.0e-23) |
| −‖Δz‖ embedding displacement (mean pooling) | +0.160 (p=4.3e-25) |
| cos(z_wt, z_mut) | +0.155 (p=9.9e-24) |

Reading: the likelihood ratio is the field-standard zero-shot variant score; displacement measures representation movement. A positive ρ means the statistic tracks the assay (higher = more functional). The magnitudes are consistent with the encoder's scale — published zero-shot correlations reach ~0.4–0.5 only for 650M+ models, and CALM1 complementation is a known hard target — and notably, embedding displacement tracks the assay almost as well as the likelihood score at this scale. Correlations are specific to this protein and assay; neither statistic is a fitness predictor.
