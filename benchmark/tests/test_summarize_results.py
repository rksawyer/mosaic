# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

"""Tests for summarize_results.py — JSON parsing, sorting, error handling,
CSV/markdown output, and edge cases.
"""

from __future__ import annotations

import json

import pytest

from summarize_results import (
    ResultParseError,
    parse_result_file,
    summarize_directory,
)

SAMPLE_RESULT = {
    "metadata": {
        "model_id": "openai/gpt-oss-120b",
        "tp": 8,
        "ep": 2,
        "concurrency": 16,
        "input_seq_len": 1024,
        "output_seq_len": 1024,
    },
    "results": {
        "request_throughput": 12.5,
        "output_throughput": 12800.0,
        "total_token_throughput": 25600.0,
        "ttft_p50_ms": 45.2,
        "ttft_p99_ms": 120.3,
        "tpot_p50_ms": 8.1,
        "tpot_p99_ms": 15.6,
        "e2e_latency_p50_ms": 8300.0,
        "e2e_latency_p99_ms": 12400.0,
    },
}


def _write_result(tmp_path, filename, data):
    f = tmp_path / filename
    f.write_text(json.dumps(data))
    return f


# ---------------------------------------------------------------------------
# parse_result_file
# ---------------------------------------------------------------------------


class TestParseResultFile:
    def test_valid_result(self, tmp_path):
        f = _write_result(tmp_path, "tp8-ep2-16-1024x1024.json", SAMPLE_RESULT)
        result = parse_result_file(str(f))
        assert result["tp"] == 8
        assert result["ep"] == 2
        assert result["conc"] == 16
        assert result["output_throughput"] == 12800.0
        assert result["ttft_p50_ms"] == 45.2

    def test_malformed_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{not valid json")
        with pytest.raises(ResultParseError, match="Cannot parse"):
            parse_result_file(str(f))

    def test_missing_metadata_key(self, tmp_path):
        bad = {"results": SAMPLE_RESULT["results"]}
        f = _write_result(tmp_path, "no_meta.json", bad)
        with pytest.raises(ResultParseError, match="missing 'metadata'"):
            parse_result_file(str(f))

    def test_missing_results_key(self, tmp_path):
        bad = {"metadata": SAMPLE_RESULT["metadata"]}
        f = _write_result(tmp_path, "no_results.json", bad)
        with pytest.raises(ResultParseError, match="missing 'results'"):
            parse_result_file(str(f))

    def test_missing_required_metadata_fields(self, tmp_path):
        bad = {
            "metadata": {"model_id": "x"},
            "results": SAMPLE_RESULT["results"],
        }
        f = _write_result(tmp_path, "sparse_meta.json", bad)
        with pytest.raises(ResultParseError, match="missing metadata fields"):
            parse_result_file(str(f))

    def test_missing_required_result_fields(self, tmp_path):
        bad = {
            "metadata": SAMPLE_RESULT["metadata"],
            "results": {"some_other_metric": 42},
        }
        f = _write_result(tmp_path, "sparse_results.json", bad)
        with pytest.raises(ResultParseError, match="missing result fields"):
            parse_result_file(str(f))

    def test_optional_fields_default_to_zero(self, tmp_path):
        minimal = {
            "metadata": SAMPLE_RESULT["metadata"],
            "results": {
                "request_throughput": 10.0,
                "output_throughput": 5000.0,
            },
        }
        f = _write_result(tmp_path, "minimal.json", minimal)
        result = parse_result_file(str(f))
        assert result["ttft_p50_ms"] == 0.0
        assert result["tpot_p99_ms"] == 0.0
        assert result["e2e_latency_p50_ms"] == 0.0

    def test_nonexistent_file(self):
        with pytest.raises(ResultParseError, match="Cannot parse"):
            parse_result_file("/nonexistent/path/result.json")


# ---------------------------------------------------------------------------
# summarize_directory
# ---------------------------------------------------------------------------


class TestSummarizeDirectory:
    def test_empty_directory(self, tmp_path):
        rows = summarize_directory(str(tmp_path), fmt="rows")
        assert rows == []

    def test_nonexistent_directory(self, tmp_path):
        rows = summarize_directory(str(tmp_path / "nope"), fmt="rows")
        assert rows == []

    def test_sorts_by_throughput_descending(self, tmp_path):
        for i, throughput in enumerate([100, 300, 200]):
            data = {
                "metadata": SAMPLE_RESULT["metadata"],
                "results": {
                    **SAMPLE_RESULT["results"],
                    "output_throughput": throughput,
                },
            }
            _write_result(tmp_path, f"result_{i}.json", data)

        rows = summarize_directory(str(tmp_path), fmt="rows")
        throughputs = [r["output_throughput"] for r in rows]
        assert throughputs == [300, 200, 100]

    def test_skips_meta_json_files(self, tmp_path):
        _write_result(tmp_path, "tp8-ep1-4-1024x1024.json", SAMPLE_RESULT)
        meta = {"start_ts": 1000, "end_ts": 2000, "config_name": "tp8"}
        _write_result(tmp_path, "tp8-ep1-4-1024x1024.meta.json", meta)

        rows = summarize_directory(str(tmp_path), fmt="rows")
        assert len(rows) == 1
        assert rows[0]["tp"] == 8

    def test_multiple_results(self, tmp_path):
        for conc in [4, 8, 16]:
            data = {
                "metadata": {**SAMPLE_RESULT["metadata"], "concurrency": conc},
                "results": {
                    **SAMPLE_RESULT["results"],
                    "output_throughput": conc * 1000.0,
                },
            }
            _write_result(tmp_path, f"tp8-ep2-{conc}-1024x1024.json", data)

        rows = summarize_directory(str(tmp_path), fmt="rows")
        assert len(rows) == 3

    def test_malformed_file_raises(self, tmp_path):
        (tmp_path / "bad.json").write_text("{invalid")
        with pytest.raises(ResultParseError):
            summarize_directory(str(tmp_path), fmt="rows")


# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------


class TestOutputFormats:
    def test_markdown_output(self, tmp_path):
        _write_result(tmp_path, "r.json", SAMPLE_RESULT)
        output = summarize_directory(str(tmp_path), fmt="markdown")
        assert isinstance(output, str)
        assert "Tok/s" in output
        assert "12800" in output
        assert "|" in output

    def test_markdown_empty(self, tmp_path):
        output = summarize_directory(str(tmp_path), fmt="markdown")
        assert "No results found" in output

    def test_csv_output(self, tmp_path):
        _write_result(tmp_path, "r.json", SAMPLE_RESULT)
        output = summarize_directory(str(tmp_path), fmt="csv")
        assert isinstance(output, str)
        lines = output.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        assert "output_throughput" in lines[0]
        assert "12800" in lines[1]

    def test_csv_empty(self, tmp_path):
        output = summarize_directory(str(tmp_path), fmt="csv")
        assert output == ""

    def test_csv_multiple_rows(self, tmp_path):
        for i in range(5):
            data = {
                "metadata": {**SAMPLE_RESULT["metadata"], "concurrency": i + 1},
                "results": {
                    **SAMPLE_RESULT["results"],
                    "output_throughput": (i + 1) * 1000.0,
                },
            }
            _write_result(tmp_path, f"r{i}.json", data)

        output = summarize_directory(str(tmp_path), fmt="csv")
        lines = output.strip().split("\n")
        assert len(lines) == 6  # header + 5 data rows
