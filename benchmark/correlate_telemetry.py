#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

"""Correlate InferenceX benchmark results with NCCL profiler telemetry.

Reads sweep metadata (.meta.json) to get timestamp windows, queries
Prometheus for NCCL metrics over those windows, and joins them with
InferenceX JSON results to produce an enriched summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from urllib.parse import urlencode

from tabulate import tabulate

DEFAULT_PROMETHEUS_URL = "http://localhost:9090"
DEFAULT_STEP = "15s"

# Prometheus queries for NCCL metrics (from Research.md Section 6.2 / 6.3)
METRIC_QUERIES = {
    "collective_bytes_per_sec": 'sum(rate(nccl_profiler_collective_bytes_total[1m]))',
    "collective_time_p99_us": 'histogram_quantile(0.99, sum(rate(nccl_profiler_collective_time_microseconds_bucket[1m])) by (le))',
    "rank_bytes_per_sec": 'sum(rate(nccl_profiler_rank_bytes_total[1m]))',
    "rank_rate_avg_mbps": 'avg(rate(nccl_profiler_rank_rate_MB_per_second_sum[1m]) / rate(nccl_profiler_rank_rate_MB_per_second_count[1m]))',
}


def build_prom_query(
    metric_name: str,
    start: int | float,
    end: int | float,
    step: str = DEFAULT_STEP,
) -> dict:
    """Build a Prometheus query_range request parameters dict.

    Args:
        metric_name: Key into METRIC_QUERIES.
        start: Start timestamp (epoch seconds).
        end: End timestamp (epoch seconds).
        step: Query resolution step (default: 15s).

    Returns:
        Dict with keys: query, start, end, step.
    """
    if metric_name not in METRIC_QUERIES:
        raise ValueError(
            f"Unknown metric '{metric_name}'. "
            f"Available: {sorted(METRIC_QUERIES.keys())}"
        )
    return {
        "query": METRIC_QUERIES[metric_name],
        "start": start,
        "end": end,
        "step": step,
    }


def parse_prom_response(response: dict) -> list[float]:
    """Extract numeric values from a Prometheus query_range response.

    Handles both matrix and vector result types. Returns a flat list of
    float values (the second element of each [timestamp, value] pair).
    Returns an empty list for empty or error responses.
    """
    if response.get("status") != "success":
        return []

    data = response.get("data", {})
    results = data.get("result", [])
    if not results:
        return []

    values: list[float] = []
    result_type = data.get("resultType", "")

    if result_type == "matrix":
        for series in results:
            for ts, val in series.get("values", []):
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    continue
    elif result_type == "vector":
        for item in results:
            pair = item.get("value", [])
            if len(pair) == 2:
                try:
                    values.append(float(pair[1]))
                except (ValueError, TypeError):
                    continue

    return values


def query_prometheus(
    prometheus_url: str,
    metric_name: str,
    start: int | float,
    end: int | float,
    step: str = DEFAULT_STEP,
) -> list[float]:
    """Execute a Prometheus query_range and return parsed values.

    Returns an empty list on network errors (non-fatal — telemetry may not
    be running during offline analysis).
    """
    params = build_prom_query(metric_name, start, end, step)
    url = f"{prometheus_url}/api/v1/query_range?{urlencode(params)}"
    req = Request(url, headers={"Accept": "application/json"})

    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return parse_prom_response(data)
    except (URLError, OSError, json.JSONDecodeError):
        return []


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _max_val(values: list[float]) -> float | None:
    return max(values) if values else None


def load_meta_files(results_dir: str) -> list[dict]:
    """Load all .meta.json files from the results directory."""
    dirpath = Path(results_dir)
    if not dirpath.is_dir():
        return []
    meta_files = sorted(dirpath.glob("*.meta.json"))
    entries = []
    for f in meta_files:
        try:
            entries.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return entries


def correlate(
    results_dir: str,
    prometheus_url: str = DEFAULT_PROMETHEUS_URL,
) -> list[dict]:
    """Correlate benchmark results with NCCL telemetry from Prometheus.

    For each .meta.json sweep point, queries Prometheus for NCCL metrics
    over [start_ts, end_ts] and returns enriched rows.
    """
    meta_entries = load_meta_files(results_dir)
    rows = []

    for meta in meta_entries:
        start_ts = meta.get("start_ts")
        end_ts = meta.get("end_ts")
        if start_ts is None or end_ts is None:
            continue

        row = {
            "config": meta.get("config_name", ""),
            "tp": meta.get("tp", 0),
            "ep": meta.get("ep", 0),
            "conc": meta.get("conc", 0),
            "isl": meta.get("isl", 0),
            "osl": meta.get("osl", 0),
            "duration_s": end_ts - start_ts,
        }

        for metric_name in METRIC_QUERIES:
            values = query_prometheus(prometheus_url, metric_name, start_ts, end_ts)
            row[f"{metric_name}_avg"] = _avg(values)
            row[f"{metric_name}_max"] = _max_val(values)

        rows.append(row)

    return rows


DISPLAY_COLUMNS = [
    ("config", "Config"),
    ("tp", "TP"),
    ("ep", "EP"),
    ("conc", "Conc"),
    ("isl", "ISL"),
    ("osl", "OSL"),
    ("duration_s", "Dur(s)"),
    ("collective_bytes_per_sec_avg", "Coll B/s avg"),
    ("collective_time_p99_us_avg", "Coll p99 µs"),
    ("rank_bytes_per_sec_avg", "Rank B/s avg"),
    ("rank_rate_avg_mbps_avg", "Rank MB/s"),
]


def format_correlation_table(rows: list[dict]) -> str:
    """Format correlation results as a markdown table."""
    if not rows:
        return "No telemetry data found."
    headers = [header for _, header in DISPLAY_COLUMNS]
    table = []
    for row in rows:
        table.append([
            row.get(key, "N/A") if row.get(key) is not None else "N/A"
            for key, _ in DISPLAY_COLUMNS
        ])
    return tabulate(table, headers=headers, tablefmt="github", floatfmt=".1f")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlate benchmark results with NCCL telemetry"
    )
    parser.add_argument("results_dir", help="Directory with .meta.json files")
    parser.add_argument(
        "--prometheus-url", default=DEFAULT_PROMETHEUS_URL,
        help=f"Prometheus base URL (default: {DEFAULT_PROMETHEUS_URL})",
    )
    args = parser.parse_args()

    rows = correlate(args.results_dir, args.prometheus_url)
    print(format_correlation_table(rows))


if __name__ == "__main__":
    main()
