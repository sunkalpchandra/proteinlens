# Contributing

## Setup

```bash
make venv install      # Python venv + deps (+ macOS libomp fix)
cd frontend && npm install
```

Artifacts (corpus, embeddings, indexes) build with `make setup`; see the README's
Quickstart for the step-by-step pipeline. Nothing under `data/` is committed.

## Working rules

- **Determinism**: every script that samples takes `--seed`; keep new randomness
  seeded and recorded in the `experiments/` log via `ml.tracking.log_experiment`.
- **Leakage discipline**: anything that trains or evaluates must respect the
  grouped splits (`data/processed/splits.json`). If your change alters grouping,
  rerun `scripts/make_splits.py` *and* `scripts/compare_split_methods.py`, and
  retrain the pooler before quoting benchmark numbers.
- **Scientific framing**: embedding effects are representation-space statements.
  Copy the phrasing patterns in the README (Scientific framing section) — no
  fitness/pathogenicity/function claims without a validated predictor.
- **Caches are keyed**: if you change what a representation means, bump
  `EMBEDDING_VERSION` (ml/embeddings.py) or include the new parameter in the
  cache key, as the attention pooler fingerprint does.

## Checks before a PR

```bash
ruff check models ml api scripts tests
pytest                               # +RUN_MODEL_TESTS=1 if you touched inference
cd frontend && npm run typecheck && npm test && npm run build
```

CI runs the same on Ubuntu. Commit style follows the history: small, scoped
commits with `feat:` / `fix:` / `test:` / `docs:` / `perf:` / `chore:` prefixes.

## Hardware notes

8GB unified-memory hosts run everything (the encoder flushes the MPS cache per
forward; token budgets are attention-bounded — see `models/registry.py`).
ProstT5 needs the fp16 path and patience; ESM-2 650M wants 16GB+.
