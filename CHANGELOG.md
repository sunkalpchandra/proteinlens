# Changelog

## 0.4.0 — 2026-08-14

Model conservation, a four-assay DMS study, and product quantization.

- **Model-conservation track**: per-position entropy and wild-type
  log-probability of the masked-LM distribution (one forward pass), on every
  profile beside the attention track — explicitly a model signal, not an
  alignment-based conservation score.
- **Four-assay DMS study**: SUMO1, UBE2I, and TPK1 join CALM1 (11,296 measured
  substitutions from MaveDB). Finding: the likelihood ratio is high-variance
  across assays (+0.477 on SUMO1, ~0 on UBE2I) while embedding displacement is
  consistent (+0.15 to +0.24) and matches or beats it on half the assays.
- **IVFPQ backend**: product quantization for the million-vector regime — 22x
  smaller indexes at top QPS, with the recall collapse on clustered data
  documented (candidate generation, not a drop-in index).
- Sitemap + robots for the static site; DMS targets added to the corpus
  (12,008 proteins).

## 0.3.0 — 2026-08-14

Mutation scoring, experimental validation, and external integrations.

- **LM log-likelihood scoring**: wild-type-marginal LLR (Meier et al. style)
  from the masked-LM head; whole landscapes from one forward pass. Surfaced in
  /mutation, /mutation-landscape, the heatmap metric toggle, and the detail
  panel.
- **DMS validation (MaveDB)**: human Calmodulin DMS-TileSeq, 2,525 measured
  substitutions — LLR ρ = +0.187, −‖Δz‖ ρ = +0.170 (p < 1e-17); report +
  figure 11. Honest scale framing; displacement tracks the assay nearly as
  well as the likelihood score at 35M parameters.
- **Live UniProt fetch**: POST /fetch-protein embeds any accession on demand
  and searches the corpus; search page offers it on empty results.
- **PDB cross-references**: 41k structure links across 24% of the corpus,
  rendered as RCSB chips on profiles and in the demo bundle.

## 0.2.0 — 2026-08-14

Extended-studies release: multi-checkpoint scaling, contrastive pooling,
structure-aware baseline, identity-validated splits, ANN indexes, and
per-domain views.

### Studies (shared 3k evaluation subset; `reports/extended_benchmark.md`)
- ESM-2 checkpoint scaling 8M → 35M → 150M through a registry-driven interface;
  monotone probe-F1 gains (0.535 → 0.569, mean pooling).
- SupCon-trained attention pooler: retrieval P@1 0.769 vs 0.611 for the CE
  pooler on the same frozen encoder — contrastive training reshapes the cosine
  geometry retrieval uses.
- ProstT5 (1.2B, 3Di-supervised, fp16 on an 8GB host) as a structure-aware
  reference: F1 0.551 / P@1 0.773 — comparable to ESM-2 150M here.
- Split validation: annotation union-find grouping strictly subsumes MMseqs2
  30%-identity clustering (0 identity-joined pairs separated in 200k sampled).
- Index scaling: HNSW at 150k vectors = 2.6× exact-search QPS at 0.9995
  recall@10; automatic backend selection past 50k.

### Features
- Representation trajectories: cumulative mutation chains with per-step
  displacement, path length, and directness (API + builder UI).
- Per-domain views: 5,204 UniProt DOMAIN coordinates, domain tracks on
  profiles, and region-level similarity search.
- Pairwise comparison page: identity vs cosine across all four poolings.
- Cluster browser with k-means and HDBSCAN density-island views.
- UMAP neighborhood presets (balanced / local / global), precomputed.
- Sequence exports (copy/FASTA), search-result JSON export, shareable search
  URLs, keyboard navigation for residue selection, `/stats` endpoint.

### Fixed (own adversarial review, 29 findings across seven angles)
- Explorer no longer locks out attention pooling after a preset 404.
- Trajectory results invalidate on pooling change; steps hit the embedding
  cache; validation runs before any encoder load.
- Preset vocabulary unified into `ml.projection` (was five copies).
- `/stats` connection leak and eager index load removed; BLOSUM62 aligner
  built once; downloads survive Firefox/Safari URL-revocation timing.
- HDBSCAN moved to PCA-50 + leaf selection (raw-space density collapse).

## 0.1.0 — 2026-08-13/14

Initial system: 12k-protein Swiss-Prot corpus with provenance manifests,
frozen ESM-2 encoder with four poolings (learned attention pooler at 91.5%
holdout), leakage-audited family-grouped splits, FAISS retrieval, PCA→UMAP
map, mutation landscapes, benchmark suite (probes/retrieval/clustering/
stability vs k-mer and one-hot baselines), FastAPI backend, Next.js frontend,
static demo on GitHub Pages, CI.
