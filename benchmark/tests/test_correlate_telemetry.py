# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

"""Tests for correlate_telemetry.py — query building, response parsing,
meta file loading, and correlation logic (no Prometheus required).
"""

from __future__ import annotations

import json

import pytest

from correlate_telemetry import (
    METRIC_QUERIES,
    build_prom_query,
    correlate,
    format_correlation_table,
    load_meta_files,
    parse_prom_response,
)


# ---------------------------------------------------------------------------
# build_prom_query
# ---------------------------------------------------------------------------


class TestBuildPromQuery:
    def test_collective_bytes(self):
        q = build_prom_query("collective_bytes_per_sec", start=1000, end=2000)
        assert "nccl_profiler_collective_bytes" in q["query"]
        assert q["start"] == 1000
        assert q["end"] == 2000
        assert q["step"] == "15s"

    def test_collective_time_p99(self):
        q = build_prom_query("collective_time_p99_us", start=100, end=200, step="5s")
        assert "histogram_quantile" in q["query"]
        assert "nccl_profiler_collective_time" in q["query"]
        assert q["step"] == "5s"

    def test_rank_bytes(self):
        q = build_prom_query("rank_bytes_per_sec", start=0, end=100)
        assert "nccl_profiler_rank_bytes_total" in q["query"]

    def test_rank_rate(self):
        q = build_prom_query("rank_rate_avg_mbps", start=0, end=100)
        assert "nccl_profiler_rank_rate_MB_per_second" in q["query"]

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            build_prom_query("nonexistent_metric", start=0, end=100)

    def test_all_metric_keys_produce_valid_queries(self):
        for metric_name in METRIC_QUERIES:
            q = build_prom_query(metric_name, start=0, end=100)
            assert "query" in q
            assert q["query"]  # non-empty string


# ---------------------------------------------------------------------------
# parse_prom_response
# ---------------------------------------------------------------------------


class TestParsePromResponse:
    def test_matrix_result(self):
        response = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {"values": [[1000, "1234567890"], [1015, "2345678901"]]}
                ],
            },
        }
        values = parse_prom_response(response)
        assert len(values) == 2
        assert values[0] == 1234567890.0
        assert values[1] == 2345678901.0

    def test_matrix_multiple_series(self):
        response = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {"values": [[1000, "100"]]},
                    {"values": [[1000, "200"], [1015, "300"]]},
                ],
            },
        }
        values = parse_prom_response(response)
        assert values == [100.0, 200.0, 300.0]

    def test_vector_result(self):
        response = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"value": [1000, "42.5"]},
                    {"value": [1000, "99.9"]},
                ],
            },
        }
        values = parse_prom_response(response)
        assert values == [42.5, 99.9]

    def test_empty_result(self):
        response = {
            "status": "success",
            "data": {"resultType": "matrix", "result": []},
        }
        assert parse_prom_response(response) == []

    def test_error_status(self):
        response = {"status": "error", "error": "something broke"}
        assert parse_prom_response(response) == []

    def test_missing_data_key(self):
        response = {"status": "success"}
        assert parse_prom_response(response) == []

    def test_non_numeric_values_skipped(self):
        response = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {"values": [[1000, "NaN"], [1015, "100"], [1030, "not_a_number"]]}
                ],
            },
        }
        values = parse_prom_response(response)
        # NaN parses as float("nan"), "not_a_number" raises ValueError
        assert len(values) == 2  # NaN + 100.0 (not_a_number skipped)

    def test_vector_with_empty_value(self):
        response = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [{"value": []}],
            },
        }
        assert parse_prom_response(response) == []


# ---------------------------------------------------------------------------
# load_meta_files
# ---------------------------------------------------------------------------


class TestLoadMetaFiles:
    def test_loads_meta_files(self, tmp_path):
        meta = {"start_ts": 1000, "end_ts": 2000, "config_name": "tp8"}
        (tmp_path / "tp8-4-1024x1024.meta.json").write_text(json.dumps(meta))
        entries = load_meta_files(str(tmp_path))
        assert len(entries) == 1
        assert entries[0]["config_name"] == "tp8"

    def test_skips_malformed(self, tmp_path):
        (tmp_path / "good.meta.json").write_text('{"start_ts": 1, "end_ts": 2}')
        (tmp_path / "bad.meta.json").write_text("{invalid json")
        entries = load_meta_files(str(tmp_path))
        assert len(entries) == 1

    def test_empty_directory(self, tmp_path):
        assert load_meta_files(str(tmp_path)) == []

    def test_nonexistent_directory(self, tmp_path):
        assert load_meta_files(str(tmp_path / "nope")) == []

    def test_ignores_non_meta_json(self, tmp_path):
        (tmp_path / "result.json").write_text('{"not": "meta"}')
        (tmp_path / "data.meta.json").write_text('{"start_ts": 1, "end_ts": 2}')
        entries = load_meta_files(str(tmp_path))
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# correlate (without Prometheus — network errors produce empty metrics)
# ---------------------------------------------------------------------------


class TestCorrelate:
    def test_no_meta_files(self, tmp_path):
        rows = correlate(str(tmp_path), prometheus_url="http://localhost:99999")
        assert rows == []

    def test_meta_without_timestamps_skipped(self, tmp_path):
        (tmp_path / "no_ts.meta.json").write_text('{"config_name": "tp8"}')
        rows = correlate(str(tmp_path), prometheus_url="http://localhost:99999")
        assert rows == []

    def test_produces_rows_with_none_metrics(self, tmp_path):
        meta = {
            "start_ts": 1000, "end_ts": 2000,
            "config_name": "tp8", "tp": 8, "ep": 1,
            "conc": 16, "isl": 1024, "osl": 1024,
        }
        (tmp_path / "tp8.meta.json").write_text(json.dumps(meta))
        rows = correlate(str(tmp_path), prometheus_url="http://localhost:99999")
        assert len(rows) == 1
        assert rows[0]["config"] == "tp8"
        assert rows[0]["tp"] == 8
        assert rows[0]["duration_s"] == 1000
        # Metrics should be None since Prometheus is unreachable
        assert rows[0]["collective_bytes_per_sec_avg"] is None


# ---------------------------------------------------------------------------
# format_correlation_table
# ---------------------------------------------------------------------------


class TestFormatCorrelationTable:
    def test_empty_rows(self):
        output = format_correlation_table([])
        assert "No telemetry data" in output

    def test_formats_rows(self):
        rows = [{
            "config": "tp8", "tp": 8, "ep": 1, "conc": 16,
            "isl": 1024, "osl": 1024, "duration_s": 120,
            "collective_bytes_per_sec_avg": 5e9,
            "collective_time_p99_us_avg": 150.0,
            "p2p_bytes_per_sec_avg": None,
            "rank_rate_avg_mbps_avg": 4500.0,
        }]
        output = format_correlation_table(rows)
        assert "tp8" in output
        assert "N/A" in output  # p2p is None
        assert "|" in output
