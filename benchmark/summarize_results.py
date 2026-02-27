#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

"""Aggregate InferenceX JSON benchmark results into summary tables.

Scans a results directory for *.json files (excluding *.meta.json), extracts
throughput and latency metrics from InferenceX's flat output format, pulls
parallelism metadata from the companion .meta.json files, and produces markdown
or CSV output sorted by output throughput.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

from tabulate import tabulate


class ResultParseError(Exception):
    """Raised when a result JSON file is malformed or missing required fields."""


REQUIRED_RESULT_KEYS = {"request_throughput", "output_throughput"}


def _load_meta(result_path: Path) -> dict:
    """Load the companion .meta.json for a result file, if it exists."""
    meta_path = result_path.with_suffix("").with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def parse_result_file(path: str) -> dict:
    """Parse a single InferenceX result JSON file.

    InferenceX produces a flat JSON with keys like request_throughput,
    output_throughput, median_ttft_ms, p99_ttft_ms, etc.  Parallelism
    metadata (tp, ep, concurrency, isl, osl) comes from the companion
    .meta.json written by sweep.sh.

    Returns a flat dict suitable for tabulation.
    Raises ResultParseError on malformed input.
    """
    filepath = Path(path)
    try:
        data = json.loads(filepath.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise ResultParseError(f"Cannot parse {filepath.name}: {e}") from e

    if not isinstance(data, dict):
        raise ResultParseError(f"{filepath.name}: expected a JSON object")

    missing = REQUIRED_RESULT_KEYS - set(data.keys())
    if missing:
        raise ResultParseError(
            f"{filepath.name}: missing required fields: {sorted(missing)}"
        )

    meta = _load_meta(filepath)

    return {
        "file": filepath.name,
        "model_id": data.get("model_id", ""),
        "tp": meta.get("tp", ""),
        "ep": meta.get("ep", ""),
        "conc": meta.get("conc", data.get("max_concurrency", "")),
        "isl": meta.get("isl", ""),
        "osl": meta.get("osl", ""),
        "request_throughput": data["request_throughput"],
        "output_throughput": data["output_throughput"],
        "total_token_throughput": data.get("total_token_throughput", 0.0),
        "ttft_p50_ms": data.get("median_ttft_ms", 0.0),
        "ttft_p99_ms": data.get("p99_ttft_ms", 0.0),
        "tpot_p50_ms": data.get("median_tpot_ms", 0.0),
        "tpot_p99_ms": data.get("p99_tpot_ms", 0.0),
        "e2e_latency_p50_ms": data.get("median_e2el_ms", 0.0),
        "e2e_latency_p99_ms": data.get("p99_e2el_ms", 0.0),
    }


def summarize_directory(
    results_dir: str,
    fmt: str = "markdown",
) -> list[dict] | str:
    """Parse all JSON results in a directory and return sorted rows.

    Args:
        results_dir: Path to directory containing *.json result files.
        fmt: Output format — "rows" returns list of dicts, "markdown" or "csv"
             returns a formatted string.

    Returns:
        Sorted list of result dicts (fmt="rows"), or a formatted string.
    """
    dirpath = Path(results_dir)
    if not dirpath.is_dir():
        return [] if fmt == "rows" else ""

    json_files = sorted(dirpath.glob("*.json"))
    json_files = [f for f in json_files if not f.name.endswith(".meta.json")]

    rows: list[dict] = []
    for filepath in json_files:
        row = parse_result_file(str(filepath))
        rows.append(row)

    rows.sort(key=lambda r: r["output_throughput"], reverse=True)

    if fmt == "rows":
        return rows

    if fmt == "csv":
        return _format_csv(rows)

    return _format_markdown(rows)


DISPLAY_COLUMNS = [
    ("file", "File"),
    ("tp", "TP"),
    ("ep", "EP"),
    ("conc", "Conc"),
    ("isl", "ISL"),
    ("osl", "OSL"),
    ("request_throughput", "Req/s"),
    ("output_throughput", "Tok/s"),
    ("ttft_p50_ms", "TTFT p50"),
    ("ttft_p99_ms", "TTFT p99"),
    ("tpot_p50_ms", "TPOT p50"),
    ("tpot_p99_ms", "TPOT p99"),
    ("e2e_latency_p50_ms", "E2E p50"),
    ("e2e_latency_p99_ms", "E2E p99"),
]


def _format_markdown(rows: list[dict]) -> str:
    if not rows:
        return "No results found."
    headers = [header for _, header in DISPLAY_COLUMNS]
    table = [[row.get(key, "") for key, _ in DISPLAY_COLUMNS] for row in rows]
    return tabulate(table, headers=headers, tablefmt="github", floatfmt=".1f")


def _format_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    keys = [key for key, _ in DISPLAY_COLUMNS]
    writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize InferenceX benchmark results")
    parser.add_argument("results_dir", help="Directory containing result JSON files")
    parser.add_argument(
        "--format", choices=["markdown", "csv"], default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with error code 1 if any result file is malformed",
    )
    args = parser.parse_args()

    try:
        output = summarize_directory(args.results_dir, fmt=args.format)
    except ResultParseError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if isinstance(output, str):
        print(output)
    else:
        for row in output:
            print(row)


if __name__ == "__main__":
    main()
