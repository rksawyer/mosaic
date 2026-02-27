#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

# Container smoke tests for the TensorRT-LLM benchmark image.
# Requires: Docker, GPU access (nvidia-container-toolkit).
# Run after `make build` to verify the image is correctly assembled.
#
# Usage:
#   bash tests/integration/test_container_smoke.sh [IMAGE]

set -euo pipefail

IMAGE="${1:-mosaic/trtllm-bench:latest}"
PASS=0
FAIL=0

check() {
    local label="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        echo "  PASS: ${label}"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: ${label}"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Container Smoke Tests ==="
echo "Image: ${IMAGE}"
echo ""

# --- Image exists ---
echo "--- Image availability ---"
check "image exists" docker image inspect "${IMAGE}"

# --- Profiler plugin ---
echo "--- Profiler plugin ---"
check "libnccl-profiler-otel.so exists" \
    docker run --rm "${IMAGE}" \
    "test -f /usr/local/plugins/lib/libnccl-profiler-otel.so"

check "plugin is a shared library" \
    docker run --rm "${IMAGE}" \
    "file /usr/local/plugins/lib/libnccl-profiler-otel.so | grep -q 'ELF.*shared object'"

# --- TensorRT-LLM tools ---
echo "--- TensorRT-LLM ---"
check "trtllm-serve is available" \
    docker run --rm "${IMAGE}" \
    "which trtllm-serve"

check "trtllm-bench is available" \
    docker run --rm "${IMAGE}" \
    "which trtllm-bench"

# --- InferenceX ---
echo "--- InferenceX ---"
check "InferenceX directory exists" \
    docker run --rm "${IMAGE}" \
    "test -d /workspace/InferenceX"

check "benchmark_serving.py exists" \
    docker run --rm "${IMAGE}" \
    "find /workspace/InferenceX -name 'benchmark_serving.py' | grep -q ."

# --- Benchmark scripts ---
echo "--- Benchmark scripts ---"
check "generate_config.py exists" \
    docker run --rm "${IMAGE}" \
    "test -f /workspace/benchmark/generate_config.py"

check "configs.py exists" \
    docker run --rm "${IMAGE}" \
    "test -f /workspace/benchmark/configs.py"

check "parallel_config.yaml.j2 exists" \
    docker run --rm "${IMAGE}" \
    "test -f /workspace/benchmark/parallel_config.yaml.j2"

# --- Python dependencies ---
echo "--- Python dependencies ---"
check "jinja2 importable" \
    docker run --rm "${IMAGE}" \
    "python3 -c 'import jinja2'"

check "yaml importable" \
    docker run --rm "${IMAGE}" \
    "python3 -c 'import yaml'"

check "tabulate importable" \
    docker run --rm "${IMAGE}" \
    "python3 -c 'import tabulate'"

# --- Environment variables ---
echo "--- Environment ---"
check "NCCL_PROFILER_PLUGIN=otel" \
    docker run --rm "${IMAGE}" \
    "bash -c '[[ \$NCCL_PROFILER_PLUGIN == otel ]]'"

check "LD_LIBRARY_PATH includes plugins" \
    docker run --rm "${IMAGE}" \
    "bash -c '[[ \$LD_LIBRARY_PATH == */usr/local/plugins/lib* ]]'"

check "OTEL endpoint configured" \
    docker run --rm "${IMAGE}" \
    "bash -c '[[ -n \$NCCL_PROFILER_OTEL_TELEMETRY_ENDPOINT ]]'"

# --- Config generation works inside container ---
echo "--- Config generation ---"
check "generate_config.py runs" \
    docker run --rm "${IMAGE}" \
    "python3 /workspace/benchmark/generate_config.py --tp 8 --ep 1 --moe-tp 8 --moe-ep 1"

check "configs.py enumerates matrix" \
    docker run --rm "${IMAGE}" \
    "python3 /workspace/benchmark/configs.py --count"

# --- Summary ---
echo ""
echo "════════════════════════════════════════"
echo "Results: ${PASS} passed, ${FAIL} failed"
echo "════════════════════════════════════════"

if [[ ${FAIL} -gt 0 ]]; then
    exit 1
fi
