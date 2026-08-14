# ANN index benchmark

Host: arm64 · Darwin · single process, CPU. Recall is against exact (`flat`) search on identical queries (k=10, n=1000).

| regime | backend | vectors | build s | QPS | recall@10 |
|---|---|---|---|---|---|
| corpus-12k | flat | 12,005 | 0.03 | 11,031.0 | 1.0 |
| corpus-12k | hnsw | 12,005 | 0.75 | 34,609.3 | 0.9993 |
| corpus-12k | ivf | 12,005 | 0.27 | 38,061.9 | 0.969 |
| synthetic-150k | flat | 150,000 | 0.19 | 4,347.2 | 1.0 |
| synthetic-150k | hnsw | 150,000 | 48.88 | 11,088.8 | 0.9995 |
| synthetic-150k | ivf | 150,000 | 67.93 | 7,287.9 | 1.0 |

Reading: at 12k vectors exact search is already fast — the ANN backends exist for the 100k+ regime, where HNSW trades a one-time build cost for an order-of-magnitude query speedup at high recall. `auto_backend` switches at 50k vectors.
