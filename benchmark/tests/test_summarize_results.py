# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

"""Tests for summarize_results.py — JSON parsing, sorting, error handling,
CSV/markdown output, and edge cases.

InferenceX produces a flat JSON (no nested metadata/results), so test data
mirrors that format.  Parallelism metadata comes from companion .meta.json
files written by sweep.sh.
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
    "date": "20260227-171400",
    "backend": "openai",
    "model_id": "openai/gpt-oss-120b",
    "tokenizer_id": "openai/gpt-oss-120b",
    "best_of": 1,
    "num_prompts": 1000,
    "request_rate": 4.0,
    "burstiness": 1.0,
    "max_concurrency": 4,
    "duration": 806.41,
    "completed": 1000,
    "total_input_tokens": 1024000,
    "total_output_tokens": 1024000,
    "request_throughput": 1.24,
    "output_throughput": 1269.83,
    "total_token_throughput": 2539.65,
    "mean_ttft_ms": 186.34,
    "median_ttft_ms": 156.93,
    "p99_ttft_ms": 1084.29,
    "mean_tpot_ms": 2.97,
    "median_tpot_ms": 2.97,
    "p99_tpot_ms": 3.04,
    "mean_itl_ms": 2.97,
    "median_itl_ms": 2.91,
    "p99_itl_ms": 4.00,
}

SAMPLE_META = {
    "start_ts": 1772211324,
    "end_ts": 1772212448,
    "config_name": "tp8",
    "tp": 8,
    "ep": 1,
    "moe_tp": 8,
    "moe_ep": 1,
    "dp_attention": False,
    "isl": 1024,
    "osl": 1024,
    "conc": 4,
    "model": "openai/gpt-oss-120b",
}


def _write_result(tmp_path, filename, data, meta=None):
    f = tmp_path / filename
    f.write_text(json.dumps(data))
    if meta is not None:
        meta_name = filename.replace(".json", ".meta.json")
        (tmp_path / meta_name).write_text(json.dumps(meta))
    return f


# ---------------------------------------------------------------------------
# parse_result_file
# ---------------------------------------------------------------------------


class TestParseResultFile:
    def test_valid_result_with_meta(self, tmp_path):
        f = _write_result(tmp_path, "tp8-4-1024x1024.json", SAMPLE_RESULT, SAMPLE_META)
        result = parse_result_file(str(f))
        assert result["tp"] == 8
        assert result["ep"] == 1
        assert result["conc"] == 4
        assert result["isl"] == 1024
        assert result["osl"] == 1024
        assert result["output_throughput"] == 1269.83
        assert result["ttft_p50_ms"] == 156.93
        assert result["ttft_p99_ms"] == 1084.29

    def test_valid_result_without_meta(self, tmp_path):
        f = _write_result(tmp_path, "tp8-4-1024x1024.json", SAMPLE_RESULT)
        result = parse_result_file(str(f))
        assert result["tp"] == ""
        assert result["conc"] == 4  # falls back to max_concurrency from result
        assert result["output_throughput"] == 1269.83

    def test_malformed_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{not valid json")
        with pytest.raises(ResultParseError, match="Cannot parse"):
            parse_result_file(str(f))

    def test_missing_required_fields(self, tmp_path):
        bad = {"some_other_metric": 42}
        f = _write_result(tmp_path, "sparse.json", bad)
        with pytest.raises(ResultParseError, match="missing required fields"):
            parse_result_file(str(f))

    def test_optional_fields_default_to_zero(self, tmp_path):
        minimal = {
            "request_throughput": 10.0,
            "output_throughput": 5000.0,
        }
        f = _write_result(tmp_path, "minimal.json", minimal)
        result = parse_result_file(str(f))
        assert result["ttft_p50_ms"] == 0.0
        assert result["tpot_p99_ms"] == 0.0
        assert result["e2e_latency_p50_ms"] == 0.0

    def test_nonexistent_file(self):
        with pytest.raises(ResultParseError, match="Cannot parse"):
            parse_result_file("/nonexistent/path/result.json")

    def test_non_dict_json(self, tmp_path):
        f = tmp_path / "array.json"
        f.write_text("[1, 2, 3]")
        with pytest.raises(ResultParseError, match="expected a JSON object"):
            parse_result_file(str(f))


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
            data = {**SAMPLE_RESULT, "output_throughput": throughput}
            _write_result(tmp_path, f"result_{i}.json", data)

        rows = summarize_directory(str(tmp_path), fmt="rows")
        throughputs = [r["output_throughput"] for r in rows]
        assert throughputs == [300, 200, 100]

    def test_skips_meta_json_files(self, tmp_path):
        _write_result(tmp_path, "tp8-4-1024x1024.json", SAMPLE_RESULT, SAMPLE_META)

        rows = summarize_directory(str(tmp_path), fmt="rows")
        assert len(rows) == 1
        assert rows[0]["tp"] == 8

    def test_multiple_results(self, tmp_path):
        for conc in [4, 8, 16]:
            data = {
                **SAMPLE_RESULT,
                "max_concurrency": conc,
                "output_throughput": conc * 1000.0,
            }
            meta = {**SAMPLE_META, "conc": conc}
            _write_result(tmp_path, f"tp8-{conc}-1024x1024.json", data, meta)

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
        assert "1269" in output
        assert "|" in output

    def test_markdown_empty(self, tmp_path):
        output = summarize_directory(str(tmp_path), fmt="markdown")
        assert "No results found" in output

    def test_csv_output(self, tmp_path):
        _write_result(tmp_path, "r.json", SAMPLE_RESULT)
        output = summarize_directory(str(tmp_path), fmt="csv")
        assert isinstance(output, str)
        lines = output.strip().split("\n")
        assert len(lines) == 2
        assert "output_throughput" in lines[0]
        assert "1269" in lines[1]

    def test_csv_empty(self, tmp_path):
        output = summarize_directory(str(tmp_path), fmt="csv")
        assert output == ""

    def test_csv_multiple_rows(self, tmp_path):
        for i in range(5):
            data = {
                **SAMPLE_RESULT,
                "output_throughput": (i + 1) * 1000.0,
            }
            _write_result(tmp_path, f"r{i}.json", data)

        output = summarize_directory(str(tmp_path), fmt="csv")
        lines = output.strip().split("\n")
        assert len(lines) == 6
