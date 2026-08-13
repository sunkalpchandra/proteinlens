PYTHON ?= .venv/bin/python
PIP    ?= .venv/bin/pip

.PHONY: setup venv install data embeddings index benchmarks figures demo api frontend test lint clean

setup: venv install data embeddings index demo   ## Full local setup: env + data + embeddings + index + demo bundle

venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip

install:
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/preprocess.py

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
