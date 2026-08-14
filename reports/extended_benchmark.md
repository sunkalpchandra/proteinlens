# Extended benchmark — checkpoint scale, pooling objective, structure-aware baseline

Shared evaluation subset: 2997 proteins; probes use the corpus's leakage-aware splits restricted to the subset; retrieval label = primary Pfam domain; clustering = k-means(15) NMI vs family.

| representation | group | params (M) | dim | probe F1 (mean) | P@1 | P@10 | NMI |
|---|---|---|---|---|---|---|---|
| kmer3 | baseline | 0 | 8000 | 0.319 | 0.399 | 0.104 | 0.437 |
| onehot | baseline | 0 | 20 | 0.368 | 0.215 | 0.062 | 0.455 |
| esm2-8M-bos | esm2-scaling | 8 | 320 | 0.476 | 0.416 | 0.130 | 0.473 |
| esm2-8M-max | esm2-scaling | 8 | 320 | 0.452 | 0.658 | 0.260 | 0.477 |
| esm2-8M-mean | esm2-scaling | 8 | 320 | 0.535 | 0.591 | 0.213 | 0.464 |
| esm2-35M-bos | esm2-scaling | 35 | 480 | 0.499 | 0.497 | 0.161 | 0.455 |
| esm2-35M-max | esm2-scaling | 35 | 480 | 0.479 | 0.778 | 0.293 | 0.474 |
| esm2-35M-mean | esm2-scaling | 35 | 480 | 0.545 | 0.648 | 0.215 | 0.470 |
| esm2-150M-bos | esm2-scaling | 150 | 640 | 0.547 | 0.550 | 0.170 | 0.469 |
| esm2-150M-max | esm2-scaling | 150 | 640 | 0.513 | 0.805 | 0.276 | 0.455 |
| esm2-150M-mean | esm2-scaling | 150 | 640 | 0.569 | 0.635 | 0.192 | 0.469 |
| esm2-35M-attention | pooling-objective | 35 | 480 | 0.463 | 0.611 | 0.223 | 0.480 |
| esm2-35M-attention-supcon | pooling-objective | 35 | 480 | 0.520 | 0.769 | 0.292 | 0.488 |

ProstT5 carries structure supervision (3Di translation training) and ~35× the parameters of ESM-2 35M — treat its rows as a reference point, not a like-for-like pooling comparison.
