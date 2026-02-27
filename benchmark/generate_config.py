#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

"""Generate TensorRT-LLM parallel_config.yaml from sweep parameters.

Validates parallelism constraints and computes derived values (max_model_len,
max_num_tokens, kv_cache_fraction) before rendering a Jinja2 template.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import jinja2
import yaml

TEMPLATE_PATH = Path(__file__).parent / "parallel_config.yaml.j2"

VALID_TP_VALUES = {1, 2, 4, 8}
VALID_EP_VALUES = {1, 2, 4, 8}
VALID_MOE_BACKENDS = {"TRTLLM", "CUTLASS"}

SEQ_LEN_OVERHEAD = 200


class ConfigError(Exception):
    """Raised when parallelism parameters violate TensorRT-LLM constraints."""


def validate_params(
    *,
    tp: int,
    ep: int,
    moe_tp: int,
    moe_ep: int,
    dp_attention: bool = False,
) -> bool:
    """Validate parallelism parameters against TensorRT-LLM constraints.

    Raises ConfigError on invalid combinations. Returns True on success.
    """
    if tp not in VALID_TP_VALUES:
        raise ConfigError(f"tp must be one of {sorted(VALID_TP_VALUES)}, got {tp}")
    if ep not in VALID_EP_VALUES:
        raise ConfigError(f"ep must be one of {sorted(VALID_EP_VALUES)}, got {ep}")
    if moe_tp < 1:
        raise ConfigError(f"moe_tp must be >= 1, got {moe_tp}")
    if moe_ep < 1:
        raise ConfigError(f"moe_ep must be >= 1, got {moe_ep}")
    if moe_tp * moe_ep != tp:
        raise ConfigError(
            f"moe_tp * moe_ep must equal tp: {moe_tp} * {moe_ep} = "
            f"{moe_tp * moe_ep} != {tp}"
        )
    if ep > 1 and moe_ep != ep:
        raise ConfigError(
            f"When ep > 1, moe_ep must equal ep: moe_ep={moe_ep} != ep={ep}"
        )
    return True


def compute_max_model_len(isl: int, osl: int) -> int:
    """Compute max_model_len from input/output sequence lengths."""
    return isl + osl + SEQ_LEN_OVERHEAD


def compute_max_num_tokens(isl: int, osl: int, conc: int) -> int:
    """Compute max_num_tokens based on sequence lengths and concurrency.

    Heuristic from InferenceX: enough tokens for one batch of concurrent
    requests during prefill, capped at a reasonable upper bound.
    """
    prefill_tokens = isl * conc
    return min(prefill_tokens, 16384)


def compute_kv_cache_fraction(isl: int, osl: int) -> float:
    """Compute KV-cache GPU memory fraction based on sequence lengths.

    Longer sequences need more KV-cache headroom, so the fraction is reduced
    to avoid OOM. Values follow InferenceX tuning guidelines.
    """
    total_seq = isl + osl
    if total_seq <= 2200:
        return 0.95
    if total_seq <= 5000:
        return 0.85
    return 0.70


def generate_config(
    *,
    tp: int,
    ep: int,
    moe_tp: int,
    moe_ep: int,
    dp_attention: bool = False,
    conc: int = 16,
    isl: int = 1024,
    osl: int = 1024,
    moe_backend: str = "TRTLLM",
    template_path: Path | str | None = None,
) -> str:
    """Generate a parallel_config.yaml string from the given parameters.

    Validates all constraints, computes derived values, and renders the
    Jinja2 template.
    """
    validate_params(tp=tp, ep=ep, moe_tp=moe_tp, moe_ep=moe_ep, dp_attention=dp_attention)

    if moe_backend not in VALID_MOE_BACKENDS:
        raise ConfigError(
            f"moe_backend must be one of {sorted(VALID_MOE_BACKENDS)}, got {moe_backend}"
        )

    tpl_path = Path(template_path) if template_path else TEMPLATE_PATH
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(tpl_path.parent)),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    )
    template = env.get_template(tpl_path.name)

    max_model_len = compute_max_model_len(isl, osl)
    max_num_tokens = compute_max_num_tokens(isl, osl, conc)
    kv_cache_fraction = compute_kv_cache_fraction(isl, osl)

    return template.render(
        tp=tp,
        moe_tp=moe_tp,
        moe_ep=moe_ep,
        dp_attention=dp_attention,
        conc=conc,
        isl=isl,
        osl=osl,
        max_model_len=max_model_len,
        max_num_tokens=max_num_tokens,
        kv_cache_fraction=kv_cache_fraction,
        moe_backend=moe_backend,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TensorRT-LLM parallel_config.yaml"
    )
    parser.add_argument("--tp", type=int, required=True, help="tensor_parallel_size")
    parser.add_argument("--ep", type=int, default=1, help="expert_parallel_size")
    parser.add_argument("--moe-tp", type=int, required=True, help="moe_tensor_parallel_size")
    parser.add_argument("--moe-ep", type=int, required=True, help="moe_expert_parallel_size")
    parser.add_argument("--dp-attention", action="store_true", help="Enable DP-attention")
    parser.add_argument("--conc", type=int, default=16, help="Concurrency / max batch size")
    parser.add_argument("--isl", type=int, default=1024, help="Input sequence length")
    parser.add_argument("--osl", type=int, default=1024, help="Output sequence length")
    parser.add_argument(
        "--moe-backend", default="TRTLLM", choices=sorted(VALID_MOE_BACKENDS),
        help="MoE backend (default: TRTLLM)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file path (default: stdout)",
    )

    args = parser.parse_args()

    try:
        output = generate_config(
            tp=args.tp,
            ep=args.ep,
            moe_tp=args.moe_tp,
            moe_ep=args.moe_ep,
            dp_attention=args.dp_attention,
            conc=args.conc,
            isl=args.isl,
            osl=args.osl,
            moe_backend=args.moe_backend,
        )
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Config written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
