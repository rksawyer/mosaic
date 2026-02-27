#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

# Validates sweep.sh --dry-run output for each mode and single-point mode.
# No Docker or GPU required — purely tests argument parsing and matrix generation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWEEP="${SCRIPT_DIR}/sweep.sh"
export PYTHON="${PYTHON:-${SCRIPT_DIR}/.venv/bin/python}"
PASS=0
FAIL=0

TMPOUT=$(mktemp)
trap 'rm -f "${TMPOUT}"' EXIT

assert_file_contains() {
    local label="$1" file="$2" pattern="$3"
    if grep -q "${pattern}" "${file}"; then
        echo "  PASS: ${label}"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: ${label} — expected pattern '${pattern}' not found"
        FAIL=$((FAIL + 1))
    fi
}

assert_file_not_empty() {
    local label="$1" file="$2"
    if [[ -s "${file}" ]]; then
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

echo "=== Test: --dry-run --single (content checks) ==="
"${SWEEP}" --dry-run --single --tp 8 --ep 2 --conc 16 --isl 1024 --osl 1024 \
    --results-dir /tmp/sweep-test > "${TMPOUT}" 2>&1
assert_file_not_empty "produces output" "${TMPOUT}"
assert_file_contains "mentions trtllm-serve" "${TMPOUT}" "trtllm-serve"
assert_file_contains "mentions benchmark_serving" "${TMPOUT}" "benchmark_serving"
assert_file_contains "mentions generate_config" "${TMPOUT}" "generate_config"
assert_file_contains "single point TP=8" "${TMPOUT}" "TP=8"
assert_file_contains "single point EP=2" "${TMPOUT}" "EP=2"
assert_file_contains "single point CONC=16" "${TMPOUT}" "CONC=16"

# ── Test: dry-run tp mode ────────────────────────────────────────────────────

echo "=== Test: --dry-run --mode tp ==="
"${SWEEP}" --dry-run --mode tp --results-dir /tmp/sweep-test > "${TMPOUT}" 2>&1
assert_file_not_empty "produces output" "${TMPOUT}"
assert_file_contains "tp mode references TP=" "${TMPOUT}" "TP="

# ── Test: dry-run ep mode ────────────────────────────────────────────────────

echo "=== Test: --dry-run --mode ep ==="
"${SWEEP}" --dry-run --mode ep --results-dir /tmp/sweep-test > "${TMPOUT}" 2>&1
assert_file_not_empty "produces output" "${TMPOUT}"

# ── Test: dry-run hybrid mode ────────────────────────────────────────────────

echo "=== Test: --dry-run --mode hybrid ==="
"${SWEEP}" --dry-run --mode hybrid --results-dir /tmp/sweep-test > "${TMPOUT}" 2>&1
assert_file_not_empty "produces output" "${TMPOUT}"

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
