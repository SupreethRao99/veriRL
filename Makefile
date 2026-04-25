# ============================================================
#  VeriRL — Developer Makefile
#  Usage: make <target>   |   make help
# ============================================================

# Overridable defaults (mirror config.yaml values)
PORT    ?= 8000
IMAGE   ?= verirl:latest
CONFIG  ?= config.yaml

# Detect uv; fall back to plain python
UV := $(shell command -v uv 2>/dev/null)
PYTHON := $(if $(UV),uv run python,python3)

.DEFAULT_GOAL := help

.PHONY: help \
        install install-dev install-training \
        serve serve-dev \
        validate test test-cov \
        docker-build docker-run docker-push \
        train train-vllm smoke-test \
        clean

# ── Self-documenting help ────────────────────────────────────────────────────

help: ## Show this help message
	@printf "\n\033[1mVeriRL — available targets\033[0m\n\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	    | sort \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@printf "\nOverridable variables: PORT=$(PORT)  IMAGE=$(IMAGE)  CONFIG=$(CONFIG)\n\n"

# ── Dependencies ────────────────────────────────────────────────────────────

install: ## Install core runtime dependencies
	uv sync

install-dev: ## Install dev extras (pytest, towncrier, ...)
	uv sync --extra dev

install-training: ## Install training extras (torch, trl, vllm, omegaconf, ...)
	uv sync --extra training

# ── Server ──────────────────────────────────────────────────────────────────

serve: ## Start the server in production mode (PORT=8000)
	uv run --project . server --port $(PORT)

serve-dev: ## Start the server with auto-reload for local development
	uvicorn server.app:app --reload --host 0.0.0.0 --port $(PORT)

# ── Quality gates ───────────────────────────────────────────────────────────

validate: ## Run openenv validate (required before every PR/release)
	openenv validate

test: ## Run the full test suite
	uv run pytest -v

test-cov: ## Run tests and print a coverage report
	uv run pytest --cov --cov-report=term-missing

# ── Docker ──────────────────────────────────────────────────────────────────

docker-build: ## Build the Docker image (IMAGE=verirl:latest)
	docker build -t $(IMAGE) -f server/Dockerfile .

docker-run: ## Run the Docker image locally on PORT
	docker run --rm -p $(PORT):8000 $(IMAGE)

docker-push: ## Push image to GHCR (requires docker login ghcr.io first)
	docker push $(IMAGE)

# ── Training (Modal) ────────────────────────────────────────────────────────

train: ## Run standard QLoRA GRPO training on Modal H100-80 GB
	modal run training/train.py

train-vllm: ## Run GRPO training with vLLM colocate mode for faster generation
	modal run training/train.py::train_vllm

smoke-test: ## Validate env connectivity and one full episode (no GPU needed)
	modal run training/train.py::smoke_test

# ── Config helper ───────────────────────────────────────────────────────────

config-show: ## Print the resolved config.yaml via OmegaConf
	@$(PYTHON) -c "\
from omegaconf import OmegaConf; \
import sys; \
cfg = OmegaConf.load('$(CONFIG)'); \
print(OmegaConf.to_yaml(cfg))"

# ── Housekeeping ────────────────────────────────────────────────────────────

clean: ## Remove build artefacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.xml
