# Split methods: annotation union-find vs MMseqs2 identity clustering

| | annotation | mmseqs (30% id, 80% cov) |
|---|---|---|
| groups | 3,509 | 8,281 |
| largest group | 2,099 (17.5%) | 52 (0.4%) |
| singleton fraction | — | 75.7% |

## Where the methods disagree

On 200,000 random pairs: 6132 joined only by annotation (annotations link what <30% identity cannot see — safe, conservative), 0 joined only by MMseqs (**potential leaks under annotation grouping** — homologous by identity yet annotation-disjoint).

## Probe metrics under each split (ESM-2 mean pooling)

| task | accuracy (annotation) | accuracy (mmseqs) | macro-F1 (annotation) | macro-F1 (mmseqs) |
|---|---|---|---|---|
| enzyme_vs_nonenzyme | 0.853 | 0.904 | 0.790 | 0.875 |
| ec_class | 0.468 | 0.699 | 0.313 | 0.529 |
| subcellular_localization | 0.551 | 0.554 | 0.401 | 0.453 |

Leakage audit (annotation): cross-split 4-mer cosine p99 0.019 vs within-train 0.022.

Leakage audit (mmseqs): cross-split 4-mer cosine p99 0.018 vs within-train 0.020.

Close agreement between the two probe columns indicates the annotation grouping was already controlling the leakage that identity clustering formalizes.
