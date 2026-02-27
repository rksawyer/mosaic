#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

# Static validation of Dockerfile.trtllm.
# Runs without Docker or GPU — checks structure, required stages, and
# optionally runs hadolint if available.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.trtllm"
PASS=0
FAIL=0

assert_contains() {
    local label="$1" file="$2" pattern="$3"
    if grep -q "${pattern}" "${file}"; then
        echo "  PASS: ${label}"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: ${label} — pattern '${pattern}' not found"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Dockerfile structure checks ==="

# File exists
if [[ -f "${DOCKERFILE}" ]]; then
    echo "  PASS: Dockerfile.trtllm exists"
    PASS=$((PASS + 1))
else
    echo "  FAIL: Dockerfile.trtllm not found at ${DOCKERFILE}"
    FAIL=$((FAIL + 1))
    echo "Results: ${PASS} passed, ${FAIL} failed"
    exit 1
fi

# Multi-stage: builder and runner
assert_contains "has builder stage" "${DOCKERFILE}" "AS builder"
assert_contains "has runner stage"  "${DOCKERFILE}" "AS runner"

# Uses TensorRT-LLM base images
assert_contains "builder uses TensorRT-LLM devel" "${DOCKERFILE}" "TRTLLM_DEVEL_IMAGE"
assert_contains "runner uses TensorRT-LLM release" "${DOCKERFILE}" "TRTLLM_RELEASE_IMAGE"

# Builds OpenTelemetry SDK
assert_contains "builds OpenTelemetry" "${DOCKERFILE}" "opentelemetry-cpp"

# Builds NCCL
assert_contains "builds NCCL" "${DOCKERFILE}" "github.com/NVIDIA/nccl"

# Builds profiler plugin
assert_contains "builds profiler plugin" "${DOCKERFILE}" "libnccl-profiler-otel.so"

# Copies plugin to runner
assert_contains "copies plugin to runner" "${DOCKERFILE}" "COPY --from=builder"

# Clones InferenceX
assert_contains "clones InferenceX" "${DOCKERFILE}" "InferenceX"

# Sets NCCL_PROFILER_PLUGIN
assert_contains "sets NCCL_PROFILER_PLUGIN" "${DOCKERFILE}" "NCCL_PROFILER_PLUGIN=otel"

# Sets LD_LIBRARY_PATH with plugins dir
assert_contains "sets LD_LIBRARY_PATH" "${DOCKERFILE}" "/usr/local/plugins/lib"

# Sets OTEL endpoint
assert_contains "sets OTEL endpoint" "${DOCKERFILE}" "NCCL_PROFILER_OTEL_TELEMETRY_ENDPOINT"

# Has OCI labels
assert_contains "has OCI title label" "${DOCKERFILE}" "org.opencontainers.image.title"
assert_contains "has NCCL version label" "${DOCKERFILE}" "ai.openmosaic.image.nccl.version"

# License header
assert_contains "has SPDX license header" "${DOCKERFILE}" "SPDX-License-Identifier"

# ── Optional: hadolint ───────────────────────────────────────────────────────

echo ""
echo "=== Hadolint (optional) ==="
if command -v hadolint > /dev/null 2>&1; then
    if hadolint "${DOCKERFILE}" --ignore DL3008 --ignore DL3013 --ignore DL3003; then
        echo "  PASS: hadolint clean"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: hadolint found issues"
        FAIL=$((FAIL + 1))
    fi
else
    echo "  SKIP: hadolint not installed"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════"
echo "Results: ${PASS} passed, ${FAIL} failed"
echo "════════════════════════════════════════"

if [[ ${FAIL} -gt 0 ]]; then
    exit 1
fi
