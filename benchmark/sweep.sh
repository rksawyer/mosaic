#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

# Parameterised benchmark sweep over TP/EP/CONC/ISL/OSL configurations.
# Reads the sweep matrix from configs.py (JSON lines), generates a
# parallel_config.yaml for each point, launches trtllm-serve inside a
# Docker container, runs InferenceX benchmark_serving.py, and collects results.
#
# Usage:
#   ./sweep.sh --model openai/gpt-oss-120b --results-dir ./results
#   ./sweep.sh --model openai/gpt-oss-120b --results-dir ./results --mode tp
#   ./sweep.sh --dry-run --mode hybrid
#   ./sweep.sh --single --tp 8 --ep 2 --conc 16 --isl 1024 --osl 1024

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-mosaic/trtllm-bench:latest}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
OTEL_ENDPOINT="${OTEL_ENDPOINT:-http://localhost:4318}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
SERVER_PORT=8000
HEALTH_TIMEOUT=600          # seconds to wait for trtllm-serve readiness
HEALTH_INTERVAL=5           # seconds between health checks
CONTAINER_NAME="trtllm-bench-server"
PYTHON="${PYTHON:-python3}"
BENCHMARK_SCRIPT="/workspace/InferenceX/utils/bench_serving/benchmark_serving.py"
NUM_REQUESTS=1000

# ── Argument parsing ─────────────────────────────────────────────────────────

MODE=""
DRY_RUN=false
SINGLE=false
SINGLE_TP="" SINGLE_EP="" SINGLE_CONC="" SINGLE_ISL="" SINGLE_OSL=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --model MODEL          HuggingFace model ID (default: ${MODEL})
  --results-dir DIR      Results output directory (default: ${RESULTS_DIR})
  --mode MODE            Filter: tp, ep, hybrid, dp (default: all)
  --image IMAGE          Docker image (default: ${IMAGE})
  --otel-endpoint URL    OTLP HTTP endpoint (default: ${OTEL_ENDPOINT})
  --dry-run              Print commands without executing
  --single               Run a single config point (requires --tp, --ep, etc.)
  --tp N                 Tensor parallel size (single mode)
  --ep N                 Expert parallel size (single mode)
  --conc N               Concurrency level (single mode)
  --isl N                Input sequence length (single mode)
  --osl N                Output sequence length (single mode)
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2";        shift 2 ;;
        --results-dir)  RESULTS_DIR="$2";  shift 2 ;;
        --mode)         MODE="$2";         shift 2 ;;
        --image)        IMAGE="$2";        shift 2 ;;
        --otel-endpoint) OTEL_ENDPOINT="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true;      shift ;;
        --single)       SINGLE=true;       shift ;;
        --tp)           SINGLE_TP="$2";    shift 2 ;;
        --ep)           SINGLE_EP="$2";    shift 2 ;;
        --conc)         SINGLE_CONC="$2";  shift 2 ;;
        --isl)          SINGLE_ISL="$2";   shift 2 ;;
        --osl)          SINGLE_OSL="$2";   shift 2 ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────

log() { echo "[sweep] $(date '+%H:%M:%S') $*"; }

cleanup_container() {
    if docker ps -q --filter "name=${CONTAINER_NAME}" | grep -q .; then
        log "Stopping container ${CONTAINER_NAME}..."
        docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi
}

wait_for_health() {
    local url="http://localhost:${SERVER_PORT}/health"
    local waited=0
    log "Waiting for server health at ${url} (timeout: ${HEALTH_TIMEOUT}s)..."
    while [[ ${waited} -lt ${HEALTH_TIMEOUT} ]]; do
        if curl -sf "${url}" >/dev/null 2>&1; then
            log "Server healthy after ${waited}s"
            return 0
        fi
        sleep "${HEALTH_INTERVAL}"
        waited=$((waited + HEALTH_INTERVAL))
    done
    log "ERROR: Server failed to become healthy within ${HEALTH_TIMEOUT}s"
    return 1
}

# ── Generate sweep matrix ───────────────────────────────────────────────────

generate_matrix() {
    local mode_arg=""
    if [[ -n "${MODE}" ]]; then
        mode_arg="--mode ${MODE}"
    fi
    # shellcheck disable=SC2086
    "${PYTHON}" "${SCRIPT_DIR}/configs.py" --format jsonl ${mode_arg}
}

generate_single_entry() {
    if [[ -z "${SINGLE_TP}" || -z "${SINGLE_EP}" || -z "${SINGLE_CONC}" \
       || -z "${SINGLE_ISL}" || -z "${SINGLE_OSL}" ]]; then
        echo "ERROR: --single requires --tp, --ep, --conc, --isl, --osl" >&2
        exit 1
    fi
    local moe_ep="${SINGLE_EP}"
    local moe_tp=$(( SINGLE_TP / SINGLE_EP ))
    local dp_attention=false
    local max_model_len=$(( SINGLE_ISL + SINGLE_OSL + 200 ))
    local name="tp${SINGLE_TP}"
    if [[ "${SINGLE_EP}" -gt 1 ]]; then
        name="tp${moe_tp}-ep${SINGLE_EP}"
    fi
    cat <<ENTRY
{"name":"${name}","tp":${SINGLE_TP},"ep":${SINGLE_EP},"moe_tp":${moe_tp},"moe_ep":${moe_ep},"dp_attention":${dp_attention},"isl":${SINGLE_ISL},"osl":${SINGLE_OSL},"conc":${SINGLE_CONC},"max_model_len":${max_model_len},"result_filename":"${name}-${SINGLE_CONC}-${SINGLE_ISL}x${SINGLE_OSL}.json"}
ENTRY
}

# ── Run one benchmark point ─────────────────────────────────────────────────

run_benchmark() {
    local cfg="$1"

    local name tp ep moe_tp moe_ep dp_attention isl osl conc max_model_len result_filename
    name=$(echo "${cfg}"     | "${PYTHON}" -c "import sys,json; print(json.load(sys.stdin)['name'])")
    tp=$(echo "${cfg}"       | "${PYTHON}" -c "import sys,json; print(json.load(sys.stdin)['tp'])")
    ep=$(echo "${cfg}"       | "${PYTHON}" -c "import sys,json; print(json.load(sys.stdin)['ep'])")
    moe_tp=$(echo "${cfg}"   | "${PYTHON}" -c "import sys,json; print(json.load(sys.stdin)['moe_tp'])")
    moe_ep=$(echo "${cfg}"   | "${PYTHON}" -c "import sys,json; print(json.load(sys.stdin)['moe_ep'])")
    dp_attention=$(echo "${cfg}" | "${PYTHON}" -c "import sys,json; d=json.load(sys.stdin)['dp_attention']; print(str(d).lower())")
    isl=$(echo "${cfg}"      | "${PYTHON}" -c "import sys,json; print(json.load(sys.stdin)['isl'])")
    osl=$(echo "${cfg}"      | "${PYTHON}" -c "import sys,json; print(json.load(sys.stdin)['osl'])")
    conc=$(echo "${cfg}"     | "${PYTHON}" -c "import sys,json; print(json.load(sys.stdin)['conc'])")
    max_model_len=$(echo "${cfg}" | "${PYTHON}" -c "import sys,json; print(json.load(sys.stdin)['max_model_len'])")
    result_filename=$(echo "${cfg}" | "${PYTHON}" -c "import sys,json; print(json.load(sys.stdin)['result_filename'])")

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Config: ${name} | TP=${tp} EP=${ep} moe_TP=${moe_tp} moe_EP=${moe_ep}"
    log "         ISL=${isl} OSL=${osl} CONC=${conc} DP-Attn=${dp_attention}"
    log "         Result: ${result_filename}"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Step 1: Generate parallel_config.yaml
    local config_dir="${RESULTS_DIR}/configs"
    mkdir -p "${config_dir}"
    local config_file="${config_dir}/${name}-${conc}-${isl}x${osl}.yaml"

    local dp_flag=""
    if [[ "${dp_attention}" == "true" ]]; then
        dp_flag="--dp-attention"
    fi

    local gen_cmd="${PYTHON} ${SCRIPT_DIR}/generate_config.py \
        --tp ${tp} --ep ${ep} --moe-tp ${moe_tp} --moe-ep ${moe_ep} \
        ${dp_flag} --conc ${conc} --isl ${isl} --osl ${osl} \
        -o ${config_file}"

    if [[ "${DRY_RUN}" == true ]]; then
        log "[dry-run] generate_config: ${gen_cmd}"
        log "[dry-run] trtllm-serve ${MODEL} --extra_llm_api_options ${config_file} --max_seq_len ${max_model_len} --port ${SERVER_PORT}"
        log "[dry-run] benchmark_serving.py --backend openai --host localhost --port ${SERVER_PORT} --model ${MODEL} --dataset-name random --random-input-len ${isl} --random-output-len ${osl} --num-prompts ${NUM_REQUESTS} --request-rate ${conc} --max-concurrency ${conc} --save-result --result-dir /results --result-filename ${result_filename}"
        log "[dry-run] result -> ${RESULTS_DIR}/${result_filename}"
        return 0
    fi

    log "Generating config: ${config_file}"
    eval "${gen_cmd}"

    # Step 2: Start trtllm-serve in a Docker container
    cleanup_container

    local start_ts
    start_ts=$(date +%s)

    log "Starting trtllm-serve (TP=${tp}, EP=${ep})..."
    docker run -d \
        --name "${CONTAINER_NAME}" \
        --gpus all \
        --network host \
        --shm-size=16g \
        -v "${HF_HOME}:/root/.cache/huggingface" \
        -e "HF_HOME=/root/.cache/huggingface" \
        -v "$(realpath "${config_file}"):/workspace/parallel_config.yaml:ro" \
        -v "$(realpath "${RESULTS_DIR}"):/results" \
        -e "NCCL_PROFILER_OTEL_TELEMETRY_ENDPOINT=${OTEL_ENDPOINT}" \
        "${IMAGE}" \
        "trtllm-serve ${MODEL} \
            --extra_llm_api_options /workspace/parallel_config.yaml \
            --max_batch_size ${conc} \
            --max_seq_len ${max_model_len} \
            --port ${SERVER_PORT}"

    # Step 3: Wait for health
    if ! wait_for_health; then
        log "Server failed to start. Dumping logs:"
        docker logs "${CONTAINER_NAME}" 2>&1 | tail -50
        cleanup_container
        return 1
    fi

    # Step 4: Run benchmark_serving.py
    log "Running benchmark (CONC=${conc}, ISL=${isl}, OSL=${osl})..."
    docker exec "${CONTAINER_NAME}" \
        python3 "${BENCHMARK_SCRIPT}" \
            --backend openai \
            --host localhost \
            --port "${SERVER_PORT}" \
            --model "${MODEL}" \
            --dataset-name random \
            --random-input-len "${isl}" \
            --random-output-len "${osl}" \
            --num-prompts "${NUM_REQUESTS}" \
            --request-rate "${conc}" \
            --max-concurrency "${conc}" \
            --save-result \
            --result-dir /results \
            --result-filename "${result_filename}"

    local end_ts
    end_ts=$(date +%s)

    # Step 5: Record timestamp window for telemetry correlation
    local meta_file="${RESULTS_DIR}/${result_filename%.json}.meta.json"
    cat > "${meta_file}" <<META
{
  "start_ts": ${start_ts},
  "end_ts": ${end_ts},
  "config_name": "${name}",
  "tp": ${tp},
  "ep": ${ep},
  "moe_tp": ${moe_tp},
  "moe_ep": ${moe_ep},
  "dp_attention": ${dp_attention},
  "isl": ${isl},
  "osl": ${osl},
  "conc": ${conc},
  "model": "${MODEL}",
  "image": "${IMAGE}",
  "otel_endpoint": "${OTEL_ENDPOINT}"
}
META
    log "Result saved: ${RESULTS_DIR}/${result_filename}"
    log "Metadata saved: ${meta_file}"
    log "Telemetry window: ${start_ts} – ${end_ts} ($(( end_ts - start_ts ))s)"

    # Step 6: Stop server
    cleanup_container
}

# ── Main ─────────────────────────────────────────────────────────────────────

main() {
    mkdir -p "${RESULTS_DIR}"

    local count=0
    local failed=0

    if [[ "${SINGLE}" == true ]]; then
        log "Running single benchmark point..."
        local entry
        entry=$(generate_single_entry)
        run_benchmark "${entry}" || failed=$((failed + 1))
        count=1
    else
        local matrix
        matrix=$(generate_matrix)
        local total
        total=$(echo "${matrix}" | wc -l)
        log "Sweep matrix: ${total} configuration(s)"

        while IFS= read -r cfg; do
            count=$((count + 1))
            log "──── Point ${count}/${total} ────"
            if ! run_benchmark "${cfg}"; then
                log "WARNING: Benchmark failed for config: ${cfg}"
                failed=$((failed + 1))
            fi
        done <<< "${matrix}"
    fi

    log "════════════════════════════════════════════════════════════"
    log "Sweep complete: ${count} points, ${failed} failed"
    log "Results directory: ${RESULTS_DIR}"
    log "════════════════════════════════════════════════════════════"

    if [[ ${failed} -gt 0 ]]; then
        exit 1
    fi
}

trap cleanup_container EXIT
main
