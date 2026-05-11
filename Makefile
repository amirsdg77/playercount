# playercount — convenience targets.
# Run `make help` for the full list. Targets are intentionally small wrappers
# so the underlying commands remain greppable.

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.PHONY: help
help:
	@echo "playercount — common targets"
	@echo ""
	@echo "  install         pip install -e .[cpu,dev]"
	@echo "  install-gpu     pip install -e .[gpu,dev]"
	@echo "  lint            ruff check"
	@echo "  format          ruff format"
	@echo "  typecheck       mypy --strict (config in pyproject.toml)"
	@echo "  test            pytest (unit only — skips @slow)"
	@echo "  test-all        pytest (everything, including integration)"
	@echo "  cov             pytest with coverage"
	@echo ""
	@echo "  serve           uvicorn (reload mode, dev)"
	@echo "  run VIDEO=...   playercount run on the supplied video"
	@echo "  calibrate VIDEO=...  fit team clusterer on the supplied video"
	@echo ""
	@echo "  docker-build       docker build (cpu)"
	@echo "  docker-build-gpu   docker build (gpu)"
	@echo "  docker-run         docker compose up api"
	@echo "  download-weights   python scripts/download_weights.py"
	@echo "  clean              remove build artifacts and __pycache__"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

.PHONY: install install-gpu
install:
	$(PIP) install -e ".[cpu,dev]" --extra-index-url https://download.pytorch.org/whl/cpu

install-gpu:
	$(PIP) install -e ".[gpu,dev]" --extra-index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------

.PHONY: lint format typecheck test test-all cov
lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest -m "not slow"

test-all:
	$(PYTHON) -m pytest

cov:
	$(PYTHON) -m pytest --cov --cov-report=term-missing -m "not slow"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

VIDEO ?= data/sample.mp4
OUT   ?= results/out.ndjson

.PHONY: serve run calibrate
serve:
	$(PYTHON) -m uvicorn playercount.api.main:create_app --factory --reload \
		--host 0.0.0.0 --port 8000

run:
	$(PYTHON) -m playercount run "$(VIDEO)" --out "$(OUT)"

calibrate:
	$(PYTHON) -m playercount calibrate "$(VIDEO)" --out models/teams.joblib

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

.PHONY: docker-build docker-build-gpu docker-run docker-stop
docker-build:
	docker build -t playercount:cpu --build-arg BASE=cpu .

docker-build-gpu:
	docker build -t playercount:gpu --build-arg BASE=gpu .

docker-run:
	docker compose up api

docker-stop:
	docker compose down

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

.PHONY: download-weights clean
download-weights:
	$(PYTHON) scripts/download_weights.py

clean:
	rm -rf build dist *.egg-info .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
