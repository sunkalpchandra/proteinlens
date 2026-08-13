# ProteinLens

An interactive representation-learning platform for exploring the learned geometry of protein
sequence space — comparing proteins and mutations, retrieving representation-space neighbors,
and interpreting pretrained protein language-model representations.

> Work in progress. Full documentation lands with the benchmark suite.

```text
Protein sequence
      ↓
ESM-2 (frozen)
      ↓
Residue-level representations
      ↓
Pooling (mean / max / BOS / learned attention)
      ↓
Protein-level embedding
      ↓
┌───────────────┬────────────────┬──────────────────┐
│ Similarity    │ Mutation       │ Probe-based      │
│ search (FAISS)│ analysis       │ evaluation       │
└───────────────┴────────────────┴──────────────────┘
                         ↓
            Interactive representation map
```
