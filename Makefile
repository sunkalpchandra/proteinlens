PYTHON ?= .venv/bin/python
PIP    ?= .venv/bin/pip

.PHONY: setup venv install data splits pooler embeddings index benchmarks figures demo api frontend test lint clean

setup: venv install data splits pooler embeddings index demo   ## Full local pipeline through the demo bundle

venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e . --no-deps
	@if [ "$$(uname)" = "Darwin" ]; then bash scripts/fix_macos_libomp.sh || true; fi

data:
	$(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/preprocess.py

splits:
	$(PYTHON) scripts/make_splits.py

pooler:
	$(PYTHON) scripts/train_attention_pooler.py

embeddings:
	$(PYTHON) scripts/precompute_embeddings.py

index:
	$(PYTHON) scripts/build_index.py

benchmarks:
	$(PYTHON) scripts/run_benchmarks.py

figures:
	$(PYTHON) scripts/generate_figures.py

demo:
	$(PYTHON) scripts/build_demo_bundle.py

api:
	.venv/bin/uvicorn api.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	$(PYTHON) -m pytest

lint:
	.venv/bin/ruff check models ml api scripts tests

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
