#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

# End-to-end single sweep point test.
# Requires: GPUs, model weights downloaded, telemetry stack running.
# Runs one benchmark config point and verifies the full pipeline:
#   trtllm-serve → benchmark_serving.py → JSON result → summarize → correlate
#
# Usage:
#   bash tests/integration/test_single_sweep_point.sh
#
# Prerequisites:
#   make build          # Build the Docker image
#   make telemetry-up   # Start LGTM observability stack
#   Model weights downloaded to ~/.cache/huggingface

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/test-results"
PYTHON="${PYTHON:-${SCRIPT_DIR}/.venv/bin/python}"
PASS=0
FAIL=0

check() {
    local label="$1"
    shift
    if "$@"; then
        echo "  PASS: ${label}"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: ${label}"
        FAIL=$((FAIL + 1))
    fi
}

cleanup() {
    echo "Cleaning up test results..."
    rm -rf "${RESULTS_DIR}"
}
trap cleanup EXIT

echo "=== End-to-End Single Sweep Point Test ==="
echo ""

# --- Run a single sweep point (TP=8, EP=1, smallest config) ---
echo "--- Running sweep --single ---"
"${SCRIPT_DIR}/sweep.sh" \
    --single \
    --tp 8 --ep 1 --conc 4 --isl 1024 --osl 1024 \
    --results-dir "${RESULTS_DIR}"

# --- Verify result file was created ---
echo ""
echo "--- Checking result artifacts ---"
check "result JSON exists" \
    test -f "${RESULTS_DIR}/tp8-4-1024x1024.json"

check "meta JSON exists" \
    test -f "${RESULTS_DIR}/tp8-4-1024x1024.meta.json"

check "config YAML was generated" \
    test -f "${RESULTS_DIR}/configs/tp8-4-1024x1024.yaml"

# --- Verify result JSON is valid and parseable ---
echo ""
echo "--- Validating result content ---"
check "result JSON is valid" \
    "${PYTHON}" -c "
import json, sys
data = json.load(open('${RESULTS_DIR}/tp8-4-1024x1024.json'))
assert 'request_throughput' in data, 'missing request_throughput'
assert 'output_throughput' in data, 'missing output_throughput'
assert data['completed'] > 0, 'no completed requests'
print(f'Result JSON OK: {data[\"completed\"]} completed, {data[\"output_throughput\"]:.1f} tok/s')
"

# --- Run summarize_results.py ---
echo ""
echo "--- Running summarize_results.py ---"
check "summarize produces markdown" \
    "${PYTHON}" "${SCRIPT_DIR}/summarize_results.py" "${RESULTS_DIR}"

check "summarize produces CSV" \
    "${PYTHON}" "${SCRIPT_DIR}/summarize_results.py" "${RESULTS_DIR}" --format csv

# --- Run correlate_telemetry.py ---
echo ""
echo "--- Running correlate_telemetry.py ---"
check "correlate runs (may show N/A if telemetry not available)" \
    "${PYTHON}" "${SCRIPT_DIR}/correlate_telemetry.py" "${RESULTS_DIR}"

# --- Verify meta.json has valid timestamps ---
echo ""
echo "--- Checking telemetry timestamps ---"
check "meta has valid start_ts and end_ts" \
    "${PYTHON}" -c "
import json
meta = json.load(open('${RESULTS_DIR}/tp8-4-1024x1024.meta.json'))
assert isinstance(meta['start_ts'], (int, float)), 'start_ts not numeric'
assert isinstance(meta['end_ts'], (int, float)), 'end_ts not numeric'
assert meta['end_ts'] > meta['start_ts'], 'end_ts not after start_ts'
print(f\"Telemetry window: {meta['end_ts'] - meta['start_ts']}s\")
"

# --- Summary ---
echo ""
echo "════════════════════════════════════════"
echo "Results: ${PASS} passed, ${FAIL} failed"
echo "════════════════════════════════════════"

if [[ ${FAIL} -gt 0 ]]; then
    exit 1
fi
