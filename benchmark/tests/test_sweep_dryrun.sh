#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

# Validates sweep.sh --dry-run output for each mode and single-point mode.
# No Docker or GPU required — purely tests argument parsing and matrix generation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWEEP="${SCRIPT_DIR}/sweep.sh"
PASS=0
FAIL=0

assert_contains() {
    local label="$1" output="$2" pattern="$3"
    if echo "${output}" | grep -q "${pattern}"; then
        echo "  PASS: ${label}"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: ${label} — expected pattern '${pattern}' not found"
        FAIL=$((FAIL + 1))
    fi
}

assert_not_empty() {
    local label="$1" output="$2"
    if [[ -n "${output}" ]]; then
        echo "  PASS: ${label}"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: ${label} — output was empty"
        FAIL=$((FAIL + 1))
    fi
}

# ── Test: --help exits 0 ────────────────────────────────────────────────────

echo "=== Test: --help ==="
if "${SWEEP}" --help >/dev/null 2>&1; then
    echo "  PASS: --help exits 0"
    PASS=$((PASS + 1))
else
    echo "  FAIL: --help exited non-zero"
    FAIL=$((FAIL + 1))
fi

# ── Test: dry-run full sweep ─────────────────────────────────────────────────

echo "=== Test: --dry-run (full sweep) ==="
output=$("${SWEEP}" --dry-run --results-dir /tmp/sweep-test 2>&1)
assert_not_empty "produces output" "${output}"
assert_contains "mentions trtllm-serve" "${output}" "trtllm-serve"
assert_contains "mentions benchmark_serving" "${output}" "benchmark_serving"
assert_contains "mentions generate_config" "${output}" "generate_config"

# ── Test: dry-run tp mode ────────────────────────────────────────────────────

echo "=== Test: --dry-run --mode tp ==="
output=$("${SWEEP}" --dry-run --mode tp --results-dir /tmp/sweep-test 2>&1)
assert_not_empty "produces output" "${output}"
assert_contains "references tp config" "${output}" "tp"

# ── Test: dry-run ep mode ────────────────────────────────────────────────────

echo "=== Test: --dry-run --mode ep ==="
output=$("${SWEEP}" --dry-run --mode ep --results-dir /tmp/sweep-test 2>&1)
assert_not_empty "produces output" "${output}"

# ── Test: dry-run hybrid mode ────────────────────────────────────────────────

echo "=== Test: --dry-run --mode hybrid ==="
output=$("${SWEEP}" --dry-run --mode hybrid --results-dir /tmp/sweep-test 2>&1)
assert_not_empty "produces output" "${output}"

# ── Test: dry-run single mode ────────────────────────────────────────────────

echo "=== Test: --dry-run --single ==="
output=$("${SWEEP}" --dry-run --single --tp 8 --ep 2 --conc 16 --isl 1024 --osl 1024 --results-dir /tmp/sweep-test 2>&1)
assert_not_empty "produces output" "${output}"
assert_contains "single point TP=8" "${output}" "TP=8"
assert_contains "single point EP=2" "${output}" "EP=2"
assert_contains "single point CONC=16" "${output}" "CONC=16"

# ── Test: --single without required args fails ──────────────────────────────

echo "=== Test: --single missing args ==="
if "${SWEEP}" --dry-run --single --tp 8 --results-dir /tmp/sweep-test 2>/dev/null; then
    echo "  FAIL: should have exited non-zero with missing --ep/--conc/--isl/--osl"
    FAIL=$((FAIL + 1))
else
    echo "  PASS: correctly fails with missing args"
    PASS=$((PASS + 1))
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════"
echo "Results: ${PASS} passed, ${FAIL} failed"
echo "════════════════════════════════════════"

if [[ ${FAIL} -gt 0 ]]; then
    exit 1
fi
