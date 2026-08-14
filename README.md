# ProteinLens

An interactive representation-learning platform for exploring the learned geometry of protein
sequence space. A frozen [ESM-2](https://github.com/facebookresearch/esm) protein language model
embeds a real Swiss-Prot corpus; ProteinLens turns those representations into semantic retrieval,
mutation perturbation analysis, interpretability views, and a rigorously benchmarked, interactive
map of protein space.

ProteinLens analyzes how pretrained protein representations organize sequence and functional
metadata. It deliberately makes **no** biological discovery claims — see
[Scientific framing](#scientific-framing).

## Overview

```text
Protein sequence (UniProtKB/Swiss-Prot, 9 organisms, ~12k proteins)
      ↓
ESM-2 tokenizer → frozen transformer (facebook/esm2_t12_35M_UR50D)
      ↓
Residue-level representations  (special tokens stripped)
      ↓
Pooling: mean · max · BOS · learned additive attention
      ↓
Protein-level embedding  (480-d, cached, versioned)
      ↓
┌────────────────┬─────────────────────┬────────────────────┐
│ FAISS retrieval│ Mutation analysis   │ Probes & benchmarks│
│ (exact cosine) │ Δz = z_mut − z_wt   │ (frozen encoder)   │
└────────────────┴─────────────────────┴────────────────────┘
      ↓
PCA(50) → UMAP(2) → interactive representation map (Next.js)
```

## Demo

- **Frontend**: Next.js app with a canvas-rendered embedding map (12k points, 60 fps zoom/pan),
  protein profiles, semantic search, and a mutation simulator.
- **Static demo mode**: the deployed site runs entirely from a precomputed 1,200-protein bundle —
  browsing, retrieval, attention views, and showcase mutation landscapes need **no backend**.
- **Live mode**: point `NEXT_PUBLIC_API_URL` at the FastAPI server and every feature works on
  arbitrary sequences (ESM inference server-side).

The signature workflow: search *hemoglobin* → open the protein → the map highlights its
representation-space neighbors → click residue 63 → compute the 19-substitution mutation
landscape → inspect which substitutions perturb the representation most.

## Architecture

```mermaid
flowchart LR
    subgraph data [Data pipeline]
        A[UniProt REST\nSwiss-Prot TSV] --> B[preprocess.py\nvalidate · dedup · cap]
        B --> C[proteins.parquet\n11,999 proteins]
        C --> D[make_splits.py\nfamily-grouped 70/15/15]
    end
    subgraph ml [Representation engine]
        C --> E[ESM2Encoder\nfrozen, batched]
        D --> F[train_attention_pooler.py]
        E --> F
        E --> G[precompute_embeddings.py\nmean/max/bos/attention]
        F --> G
        G --> H[build_index.py\nFAISS + PCA→UMAP + k-means + outliers]
        G --> I[run_benchmarks.py\nprobes · retrieval · purity · stability]
    end
    subgraph serve [Serving]
        H --> J[FastAPI\n/search /mutation /map …]
        J --> K[Next.js frontend]
        H --> L[demo bundle\nfrontend/public/demo]
        L --> K
    end
```

## ML method

**Encoder.** `facebook/esm2_t12_35M_UR50D` (12 layers, 480-d, ~35M parameters), always frozen.
Residue representations exclude BOS/EOS/padding via the tokenizer's special-token mask, so
representation *i* corresponds to residue *i* — verified at runtime. Batches pack sequences under
a token budget after length sorting to minimize padding waste.

**Pooling.** Four strategies over residue matrix `H ∈ R^{L×D}`:

```text
mean:       z = (1/L) Σᵢ hᵢ
max:        z_d = maxᵢ h_{i,d}
bos:        z = h_BOS
attention:  αᵢ = softmax(w₂ᵀ tanh(W₁hᵢ + b₁)),   z = Σᵢ αᵢ hᵢ
```

The attention pooler is the only learned component: trained jointly with a linear classifier on
UniProt family labels (train split only, encoder frozen), then reused corpus-wide. Its weights
`αᵢ` double as a model-dependent interpretability signal.

**Retrieval.** Embeddings are L2-normalized so cosine similarity equals inner product, then
indexed with FAISS `IndexFlatIP` — exact search (no ANN recall loss at 12k scale; the interface
is unchanged if swapped for IVF/HNSW at larger scale).

```text
sim(zᵢ, zⱼ) = (zᵢ · zⱼ) / (‖zᵢ‖ ‖zⱼ‖)
```

**Map.** PCA to 50 components, then UMAP to 2D (cosine metric, fixed seed). UMAP never runs on
raw 480-d vectors, and every parameter set is cached under a content hash.

**Mutation analysis.** For substitution `X{pos}Y`, both sequences are re-encoded and compared:

```text
Δz = z_mut − z_wt          ‖Δz‖₂, cos(z_wt, z_mut)
per-residue: ‖Δhᵢ‖₂        local window ±8 around the site
```

Displayed strictly as **representation-space perturbation** — not fitness, stability, or
pathogenicity. The landscape view computes all 19 substitutions at a site in one batched pass.

**Clustering & outliers.** K-means (k=25) and HDBSCAN on normalized embeddings; outlier score is
the percentile of mean cosine distance to the 10 nearest neighbors — a geometric isolation
statement, not a biological anomaly claim.

## Data

| | |
|---|---|
| Source | UniProtKB/Swiss-Prot (reviewed), REST stream API, release recorded in manifest |
| License | [CC BY 4.0](https://www.uniprot.org/help/license) |
| Organisms | human, mouse, zebrafish, fly, worm, arabidopsis, yeast, *E. coli* K-12, *B. subtilis* |
| Filters | length 50–512, canonical alphabet only, exact-duplicate removal, uncharacterized entries dropped |
| Corpus | 11,999 proteins, 94.5% with Pfam domains, ~3.2k family labels |
| Sampling | ≤80 proteins per family (anti-dominance), then organism-proportional to 12k, seed 42 |

Raw downloads land in `data/raw/` with a manifest (queries, UniProt release, SHA-256 per file).
Nothing under `data/` is committed; `scripts/download_data.py` reproduces it.

## Splits (leakage control)

Random splits let homologs straddle train/test and turn probe metrics into homology detection.
ProteinLens groups proteins by **UniProt family → Pfam domain → greedy 5-mer Jaccard cluster**
(fallback chain), then assigns whole groups to train/val/test (70/15/15). A leakage audit samples
cross-split pairs: 4-mer cosine similarity at p99 is 0.019 across splits vs 0.018 within train —
i.e., cross-split similarity is indistinguishable from background. Residual risk (remote homology
across families) is documented in [Limitations](#limitations).

Because families *are* the grouping unit, probe tasks target labels that cut across families
(enzyme/non-enzyme, EC top class, subcellular localization) — a genuine generalization test, not
family memorization.

## Installation

```bash
git clone https://github.com/sunkalpchandra/proteinlens
cd proteinlens
make venv install          # Python 3.11+; installs torch, transformers, faiss, …
```

macOS arm64 note: torch and faiss-cpu each vendor a libomp; loading both segfaults. `make install`
runs `scripts/fix_macos_libomp.sh`, which points FAISS at torch's copy.

## Quickstart

```bash
# Full pipeline (download → preprocess → splits → pooler → embeddings → index → demo bundle)
make setup

# Or step by step:
python scripts/download_data.py            # ~15 MB from UniProt
python scripts/preprocess.py               # → data/processed/proteins.parquet
python scripts/make_splits.py              # → leakage-audited splits
python scripts/train_attention_pooler.py   # ~15 min on Apple Silicon / GPU
python scripts/precompute_embeddings.py    # ~30 min on MPS, one pass, 4 poolings
python scripts/build_index.py              # FAISS + UMAP + clusters + outliers
python scripts/run_benchmarks.py           # full evaluation suite
python scripts/generate_figures.py         # reports/figures/*.png|pdf

# Serve
uvicorn api.main:app --reload              # backend on :8000
cd frontend && npm install && NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Environment knobs: `PROTEINLENS_MODEL` (any ESM-2 checkpoint), `PROTEINLENS_DEVICE`
(cuda/mps/cpu), `PROTEINLENS_CORS_ORIGINS`. See `.env.example`.

## Embedding pipeline

- **Cache keys**: `sha256(model | pooling | version | sequence)` in a SQLite vector cache —
  ad-hoc requests (pasted sequences, mutants) never recompute. Corpus embeddings live in
  consolidated `.npy` matrices, one per pooling, memory-mapped at serve time.
- **Versioning**: `embedding_version` invalidates caches when representation semantics change.
- **Residue-level**: computed on demand with an in-process LRU (mutation landscapes hit the same
  wild-type repeatedly).

## API

`GET /health` · `POST /embed` · `POST /search` · `GET /proteins?q=` · `GET /protein/{id}` ·
`GET /protein/{id}/attention` · `POST /mutation` · `POST /mutation-landscape` · `GET /map` ·
`GET /clusters` · `GET /benchmark`

Pydantic schemas validate everything; invalid amino-acid symbols return a clean 422 with the
offending characters named. Missing artifacts return 503 with the script to run. No filesystem
paths are exposed.

## Benchmarking

Four representation families × four evaluation axes, regenerated end-to-end by
`scripts/run_benchmarks.py` (nothing hand-entered):

| Axis | Metric | Question |
|---|---|---|
| Probes | accuracy, macro-F1, AUROC | Is function linearly accessible from frozen embeddings? |
| Retrieval | precision@1/5/10 (Pfam, family) | Do nearest neighbors share annotations? |
| Clustering | purity, NMI vs family | Does unsupervised structure recover annotations? |
| Stability | cos(z, z′) under 1 substitution | How smooth is the representation? |

Baselines: 3-mer frequency vectors (8000-d) and mean one-hot composition (20-d) — the bar any
learned representation must clear.

## Results

<!-- RESULTS:BEGIN — filled by scripts/run_benchmarks.py output; see reports/benchmark.md -->
Benchmarks are being regenerated; see `reports/benchmark.md` after running
`python scripts/run_benchmarks.py`.
<!-- RESULTS:END -->

Figures in `reports/figures/` (regenerate with `python scripts/generate_figures.py`): global map,
sequence-vs-embedding similarity, mutation displacement heatmap, retrieval precision@k, probe
results, pooling comparison, clusters, outliers.

## Reproducibility

- Deterministic seeds everywhere (download → sampling → splits → UMAP → benchmarks).
- `experiments/` records every run: config, metrics, timestamp, git commit.
- Data manifests pin the UniProt release and file hashes.
- `make setup` rebuilds the entire pipeline from a clean clone.

## Project structure

```text
proteinlens/
├── models/            # encoder.py (frozen ESM-2) · pooling.py · mutation.py
├── ml/                # embeddings, retrieval, projection, clustering,
│                      # splitting, probes, evaluation, tracking, sequence
├── api/               # FastAPI app: schemas, state, routes/
├── scripts/           # download → preprocess → splits → pooler → embed →
│                      # index → benchmarks → figures → demo bundle
├── data/              # raw/ processed/ embeddings/ index/   (never committed)
├── experiments/       # one JSON per run
├── reports/           # benchmark.md/csv + figures/
├── tests/             # ML unit tests + API contract tests
└── frontend/          # Next.js 15 + TS + Tailwind: explorer, search,
                       # protein/[id], mutation, benchmarks, about
```

## Scientific framing

- Embedding similarity is called **representation similarity**; unusually similar low-identity
  pairs are **representation-space neighbors** — no convergent-evolution claims.
- Mutation effects are **representation-space perturbations** — no fitness or pathogenicity
  prediction exists in this codebase.
- Attention weights are a **model-dependent interpretability signal**, not functional-residue
  annotations.
- Outliers are **geometrically isolated in the selected representation space**, nothing more.

## Limitations

- 35M-parameter encoder: representation quality is well below ESM-2 650M/3B; the config accepts
  larger checkpoints if you have the memory.
- Family/Pfam grouping controls the dominant leakage mode but cannot eliminate remote homology
  across families; the k-mer audit bounds only what k-mers can see.
- UMAP distorts global distances; the map is for structure browsing, not distance measurement.
- Annotation sparsity: ~24% of the corpus has no family label; localization labels are coarse.
- Attention pooling was trained on family classification; its weights reflect that objective.
- No structural (3D) information is used anywhere.

## Future work

Larger checkpoints behind the same interface · MMseqs2-based identity splits · structure-aware
baselines (e.g., ProstT5 tokens) · contrastive fine-tuning of the pooler · ANN indexes past 100k
proteins · per-domain embedding views.

## License

MIT for the code (see `LICENSE`). Protein data © UniProt Consortium, CC BY 4.0.
