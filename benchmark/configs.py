#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

"""Enumerate valid benchmark sweep configurations.

Generates the full parameter matrix from Research.md Section 4.7 and 5.1,
enforcing the constraint moe_tp * moe_ep == tp. Supports mode filtering
(tp-only, ep-only, hybrid) and outputs JSON lines for consumption by sweep.sh.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SweepConfig:
    """A single benchmark sweep configuration point."""

    name: str
    tp: int
    ep: int
    moe_tp: int
    moe_ep: int
    dp_attention: bool
    isl: int
    osl: int
    conc: int

    @property
    def max_model_len(self) -> int:
        return self.isl + self.osl + 200

    @property
    def result_filename(self) -> str:
        dp_suffix = "-dpattn" if self.dp_attention else ""
        return f"{self.name}{dp_suffix}-{self.conc}-{self.isl}x{self.osl}.json"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["max_model_len"] = self.max_model_len
        d["result_filename"] = self.result_filename
        return d


# Parallelism configurations from Research.md Section 4.7
PARALLELISM_CONFIGS: list[dict] = [
    {"name": "tp8",           "tp": 8, "ep": 1, "moe_tp": 8, "moe_ep": 1, "dp_attention": False},
    {"name": "ep8",           "tp": 8, "ep": 8, "moe_tp": 1, "moe_ep": 8, "dp_attention": False},
    {"name": "tp4-ep2",       "tp": 8, "ep": 2, "moe_tp": 4, "moe_ep": 2, "dp_attention": False},
    {"name": "tp2-ep4",       "tp": 8, "ep": 4, "moe_tp": 2, "moe_ep": 4, "dp_attention": False},
    {"name": "tp4",           "tp": 4, "ep": 1, "moe_tp": 4, "moe_ep": 1, "dp_attention": False},
    {"name": "tp2",           "tp": 2, "ep": 1, "moe_tp": 2, "moe_ep": 1, "dp_attention": False},
    {"name": "ep8",           "tp": 8, "ep": 8, "moe_tp": 1, "moe_ep": 8, "dp_attention": True},
    {"name": "tp4-ep2",       "tp": 8, "ep": 2, "moe_tp": 4, "moe_ep": 2, "dp_attention": True},
]

# Sequence-length pairs from Research.md Section 5.1
SEQ_LENGTH_PAIRS: list[tuple[int, int]] = [
    (1024, 1024),
    (1024, 8192),
    (8192, 1024),
]

# Concurrency levels from Research.md Section 5.1
CONCURRENCY_LEVELS: list[int] = [4, 8, 16, 32, 64]

# Mode filters map mode names to a predicate over parallelism config dicts
MODE_FILTERS: dict[str, callable] = {
    "tp":     lambda c: c["moe_ep"] == 1,
    "ep":     lambda c: c["moe_ep"] > 1 and not c["dp_attention"],
    "hybrid": lambda c: c["moe_ep"] > 1 and c["moe_tp"] > 1,
    "dp":     lambda c: c["dp_attention"],
}


def generate_sweep_matrix(
    mode: str | None = None,
    concurrency_levels: list[int] | None = None,
    seq_length_pairs: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """Generate the full sweep matrix, optionally filtered by mode.

    Returns a list of dicts, each representing one sweep configuration point.
    Every config satisfies the constraint moe_tp * moe_ep == tp.
    """
    configs = PARALLELISM_CONFIGS
    if mode and mode in MODE_FILTERS:
        configs = [c for c in configs if MODE_FILTERS[mode](c)]
    elif mode:
        raise ValueError(f"Unknown mode '{mode}'. Valid modes: {sorted(MODE_FILTERS.keys())}")

    conc_levels = concurrency_levels or CONCURRENCY_LEVELS
    seq_pairs = seq_length_pairs or SEQ_LENGTH_PAIRS

    matrix: list[dict] = []
    for cfg in configs:
        assert cfg["moe_tp"] * cfg["moe_ep"] == cfg["tp"], (
            f"Invalid config {cfg['name']}: "
            f"moe_tp({cfg['moe_tp']}) * moe_ep({cfg['moe_ep']}) != tp({cfg['tp']})"
        )
        for isl, osl in seq_pairs:
            for conc in conc_levels:
                sc = SweepConfig(
                    name=cfg["name"],
                    tp=cfg["tp"],
                    ep=cfg["ep"],
                    moe_tp=cfg["moe_tp"],
                    moe_ep=cfg["moe_ep"],
                    dp_attention=cfg["dp_attention"],
                    isl=isl,
                    osl=osl,
                    conc=conc,
                )
                matrix.append(sc.to_dict())

    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Enumerate benchmark sweep configurations")
    parser.add_argument(
        "--mode", choices=sorted(MODE_FILTERS.keys()),
        help="Filter to a specific parallelism mode",
    )
    parser.add_argument(
        "--format", choices=["jsonl", "json"], default="jsonl",
        help="Output format (default: jsonl)",
    )
    parser.add_argument(
        "--count", action="store_true",
        help="Only print the number of configurations",
    )
    args = parser.parse_args()

    matrix = generate_sweep_matrix(mode=args.mode)

    if args.count:
        print(len(matrix))
        return

    if args.format == "json":
        json.dump(matrix, sys.stdout, indent=2)
        print()
    else:
        for entry in matrix:
            print(json.dumps(entry))


if __name__ == "__main__":
    main()
