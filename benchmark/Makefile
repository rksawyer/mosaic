# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

# Distributed GPU Inference Benchmark Orchestration
# See Plan.md and Research.md for design details.

# ---------- Configuration ----------
IMAGE_NAME    := mosaic/trtllm-bench
IMAGE_TAG     := latest
MODEL         := openai/gpt-oss-120b
RESULTS_DIR   := ./results
LGTM_COMPOSE  := ../deployments/docker-compose.yml
BENCHMARK_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# ---------- Build ----------

.PHONY: build
build: ## Build the TensorRT-LLM + profiler + InferenceX image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile.trtllm ../

# ---------- Telemetry Stack ----------

.PHONY: telemetry-up
telemetry-up: ## Start the LGTM observability stack
	docker compose -f $(LGTM_COMPOSE) up -d

.PHONY: telemetry-down
telemetry-down: ## Stop the LGTM observability stack
	docker compose -f $(LGTM_COMPOSE) down

# ---------- Sweep ----------

.PHONY: sweep
sweep: telemetry-up ## Run full parameter sweep
	./sweep.sh --model $(MODEL) --results-dir $(RESULTS_DIR)

.PHONY: sweep-tp
sweep-tp: telemetry-up ## Sweep TP-only configurations
	./sweep.sh --model $(MODEL) --results-dir $(RESULTS_DIR) --mode tp

.PHONY: sweep-ep
sweep-ep: telemetry-up ## Sweep EP-only configurations
	./sweep.sh --model $(MODEL) --results-dir $(RESULTS_DIR) --mode ep

.PHONY: sweep-hybrid
sweep-hybrid: telemetry-up ## Sweep hybrid TP+EP configurations
	./sweep.sh --model $(MODEL) --results-dir $(RESULTS_DIR) --mode hybrid

# ---------- Results ----------

.PHONY: results
results: ## Summarize sweep results
	python3 summarize_results.py $(RESULTS_DIR)

# ---------- Validation & Testing ----------

.PHONY: validate-configs
validate-configs: ## Generate and validate all parallelism config YAML files
	python3 -m pytest tests/test_generate_config.py tests/test_configs.py -v

.PHONY: test
test: ## Run all non-GPU tests
	python3 -m pytest tests/ -v --ignore=tests/integration

.PHONY: lint
lint: ## Run linters on Dockerfile and shell scripts
	@echo "--- Dockerfile lint ---"
	hadolint Dockerfile.trtllm --ignore DL3008 --ignore DL3013
	@echo "--- Shell script lint ---"
	shellcheck sweep.sh

# ---------- Clean ----------

.PHONY: clean
clean: ## Remove result files and stopped containers
	rm -rf $(RESULTS_DIR)
	docker rm -f trtllm-bench-server 2>/dev/null || true

# ---------- Help ----------

.PHONY: help
help: ## Show this help
	@echo "Distributed GPU Inference Benchmark"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables:"
	@echo "  IMAGE_NAME    Docker image name    (default: $(IMAGE_NAME))"
	@echo "  IMAGE_TAG     Docker image tag     (default: $(IMAGE_TAG))"
	@echo "  MODEL         HuggingFace model ID (default: $(MODEL))"
	@echo "  RESULTS_DIR   Results output dir   (default: $(RESULTS_DIR))"
