# DMS validation summary — four assays, two model statistics

All assays: Weile et al.-style complementation TileSeq from MaveDB; model: frozen `facebook/esm2_t12_35M_UR50D`. Spearman ρ vs measured score (higher score = more functional).

| protein | assay | variants | ρ LLR | ρ −‖Δz‖ |
|---|---|---|---|---|
| P0DP23 | Human Calmodulin DMS-TileSeq | 2,525 | +0.187 | +0.170 |
| P63165 | SUMO1 DMS-TileSeq | 1,778 | +0.477 | +0.244 |
| P63279 | UBE2I DMS-TileSeq | 2,870 | -0.011 | +0.148 |
| Q9H3S4 | TPK1 DMS-TileSeq | 4,123 | +0.154 | +0.160 |

Mean across assays: ρ(LLR) = +0.202, ρ(−‖Δz‖) = +0.180. The two statistics behave differently: the likelihood ratio is high-variance across assays (from strong on SUMO1 to indistinguishable from zero on UBE2I), while embedding displacement is more consistent and matches or exceeds it on half the assays. Magnitudes overall fit the 35M-parameter encoder (published zero-shot correlations reach ~0.4–0.5 only at 650M+). Per-assay details in `dms_validation_{accession}.md`. Model statistics, not fitness predictions.
