# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

"""Tests for configs.py — sweep matrix enumeration, mode filtering,
constraint enforcement, and unique result filenames.
"""

from __future__ import annotations

import pytest

from configs import (
    CONCURRENCY_LEVELS,
    PARALLELISM_CONFIGS,
    SEQ_LENGTH_PAIRS,
    SweepConfig,
    generate_sweep_matrix,
)


class TestSweepConfig:
    def test_max_model_len_1k1k(self):
        sc = SweepConfig("tp8", 8, 1, 8, 1, False, 1024, 1024, 16)
        assert sc.max_model_len == 2248

    def test_max_model_len_8k1k(self):
        sc = SweepConfig("tp8", 8, 1, 8, 1, False, 8192, 1024, 16)
        assert sc.max_model_len == 9416

    def test_result_filename_no_dp(self):
        sc = SweepConfig("tp8", 8, 1, 8, 1, False, 1024, 1024, 16)
        assert sc.result_filename == "tp8-16-1024x1024.json"

    def test_result_filename_with_dp_attention(self):
        sc = SweepConfig("ep8", 8, 8, 1, 8, True, 1024, 8192, 32)
        assert sc.result_filename == "ep8-dpattn-32-1024x8192.json"

    def test_to_dict_includes_derived(self):
        sc = SweepConfig("tp8", 8, 1, 8, 1, False, 1024, 1024, 16)
        d = sc.to_dict()
        assert "max_model_len" in d
        assert "result_filename" in d
        assert d["max_model_len"] == 2248


class TestGenerateSweepMatrix:
    def test_full_matrix_is_nonempty(self):
        matrix = generate_sweep_matrix()
        assert len(matrix) > 0

    def test_full_matrix_count(self):
        matrix = generate_sweep_matrix()
        expected = len(PARALLELISM_CONFIGS) * len(SEQ_LENGTH_PAIRS) * len(CONCURRENCY_LEVELS)
        assert len(matrix) == expected

    def test_all_configs_satisfy_constraint(self):
        matrix = generate_sweep_matrix()
        for m in matrix:
            assert m["moe_tp"] * m["moe_ep"] == m["tp"], (
                f"Constraint violated: moe_tp={m['moe_tp']} * moe_ep={m['moe_ep']} "
                f"!= tp={m['tp']} in config {m['name']}"
            )

    def test_unique_result_filenames(self):
        matrix = generate_sweep_matrix()
        filenames = [m["result_filename"] for m in matrix]
        assert len(filenames) == len(set(filenames)), (
            f"Duplicate filenames: {[f for f in filenames if filenames.count(f) > 1]}"
        )

    def test_every_entry_has_required_keys(self):
        required = {"name", "tp", "ep", "moe_tp", "moe_ep", "dp_attention",
                     "isl", "osl", "conc", "max_model_len", "result_filename"}
        matrix = generate_sweep_matrix()
        for m in matrix:
            assert required.issubset(m.keys()), f"Missing keys in {m}"


class TestModeFiltering:
    def test_tp_mode_only_ep1(self):
        matrix = generate_sweep_matrix(mode="tp")
        assert len(matrix) > 0
        for m in matrix:
            assert m["moe_ep"] == 1, f"tp mode should have moe_ep=1, got {m}"

    def test_ep_mode_only_expert_parallel(self):
        matrix = generate_sweep_matrix(mode="ep")
        assert len(matrix) > 0
        for m in matrix:
            assert m["moe_ep"] > 1, f"ep mode should have moe_ep > 1, got {m}"
            assert m["dp_attention"] is False

    def test_hybrid_mode(self):
        matrix = generate_sweep_matrix(mode="hybrid")
        assert len(matrix) > 0
        for m in matrix:
            assert m["moe_ep"] > 1 and m["moe_tp"] > 1, (
                f"hybrid mode needs both moe_ep > 1 and moe_tp > 1, got {m}"
            )

    def test_dp_mode(self):
        matrix = generate_sweep_matrix(mode="dp")
        assert len(matrix) > 0
        for m in matrix:
            assert m["dp_attention"] is True

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            generate_sweep_matrix(mode="invalid")


class TestCustomParameters:
    def test_custom_concurrency_levels(self):
        matrix = generate_sweep_matrix(concurrency_levels=[1, 2])
        conc_values = {m["conc"] for m in matrix}
        assert conc_values == {1, 2}

    def test_custom_seq_length_pairs(self):
        matrix = generate_sweep_matrix(seq_length_pairs=[(512, 512)])
        for m in matrix:
            assert m["isl"] == 512
            assert m["osl"] == 512

    def test_single_config_point(self):
        matrix = generate_sweep_matrix(
            concurrency_levels=[16],
            seq_length_pairs=[(1024, 1024)],
        )
        assert len(matrix) == len(PARALLELISM_CONFIGS)
