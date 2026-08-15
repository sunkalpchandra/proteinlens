# ANN index benchmark

Host: arm64 · Darwin · single process, CPU. Recall is against exact (`flat`) search on identical queries (k=10, n=1000).

| regime | backend | vectors | build s | size MB | QPS | recall@10 |
|---|---|---|---|---|---|---|
| corpus-12k | flat | 12,008 | 0.06 | 23.1 | 742.0 | 1.0 |
| corpus-12k | hnsw | 12,008 | 8.53 | 26.3 | 4,342.0 | 0.9982 |
| corpus-12k | ivf | 12,008 | 1.2 | 24.1 | 9,049.1 | 0.9706 |
| corpus-12k | ivfpq | 12,008 | 19.48 | 2.1 | 17,332.8 | 0.5699 |
| synthetic-150k | flat | 150,000 | 0.19 | 288.0 | 4,088.4 | 1.0 |
| synthetic-150k | hnsw | 150,000 | 48.82 | 328.8 | 6,123.6 | 0.9995 |
| synthetic-150k | ivf | 150,000 | 16.38 | 293.4 | 12,481.7 | 1.0 |
| synthetic-150k | ivfpq | 150,000 | 59.48 | 13.1 | 13,568.1 | 0.1759 |

Reading: at 12k vectors exact search is already fast — the ANN backends exist for the 100k+ regime, where HNSW trades a one-time build cost for a query speedup at ~0.999 recall. IVFPQ is the memory tool: ~22x smaller at the highest QPS, but recall collapses on tightly clustered data without a reranking stage — treat it as candidate generation feeding an exact re-scorer, not a drop-in index. `auto_backend` switches to HNSW at 50k vectors.
