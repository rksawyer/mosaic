# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

"""Tests for generate_config.py — constraint validation, YAML rendering,
and derived value computation.
"""

from __future__ import annotations

import pytest
import yaml

from generate_config import (
    ConfigError,
    compute_kv_cache_fraction,
    compute_max_model_len,
    compute_max_num_tokens,
    generate_config,
    validate_params,
)


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------


class TestValidateParams:
    def test_tp8_ep1_pure_tensor_parallel(self):
        assert validate_params(tp=8, ep=1, moe_tp=8, moe_ep=1) is True

    def test_ep8_pure_expert_parallel(self):
        assert validate_params(tp=8, ep=8, moe_tp=1, moe_ep=8) is True

    def test_tp4_ep2_hybrid(self):
        assert validate_params(tp=8, ep=2, moe_tp=4, moe_ep=2) is True

    def test_tp2_ep4_hybrid(self):
        assert validate_params(tp=8, ep=4, moe_tp=2, moe_ep=4) is True

    def test_tp4_no_ep(self):
        assert validate_params(tp=4, ep=1, moe_tp=4, moe_ep=1) is True

    def test_tp2_no_ep(self):
        assert validate_params(tp=2, ep=1, moe_tp=2, moe_ep=1) is True

    def test_dp_attention_flag(self):
        assert validate_params(tp=8, ep=8, moe_tp=1, moe_ep=8, dp_attention=True) is True

    def test_invalid_tp_value(self):
        with pytest.raises(ConfigError, match="tp must be one of"):
            validate_params(tp=3, ep=1, moe_tp=3, moe_ep=1)

    def test_invalid_ep_value(self):
        with pytest.raises(ConfigError, match="ep must be one of"):
            validate_params(tp=8, ep=3, moe_tp=8, moe_ep=1)

    def test_moe_tp_times_moe_ep_must_equal_tp(self):
        with pytest.raises(ConfigError, match="moe_tp .* moe_ep must equal tp"):
            validate_params(tp=8, ep=1, moe_tp=4, moe_ep=4)

    def test_moe_ep_must_match_ep_when_ep_gt_1(self):
        with pytest.raises(ConfigError, match="moe_ep must equal ep"):
            validate_params(tp=8, ep=2, moe_tp=2, moe_ep=4)

    def test_moe_tp_less_than_1(self):
        with pytest.raises(ConfigError, match="moe_tp must be >= 1"):
            validate_params(tp=8, ep=1, moe_tp=0, moe_ep=8)

    def test_moe_ep_less_than_1(self):
        with pytest.raises(ConfigError, match="moe_ep must be >= 1"):
            validate_params(tp=8, ep=1, moe_tp=8, moe_ep=0)


# ---------------------------------------------------------------------------
# Derived value computation
# ---------------------------------------------------------------------------


class TestDerivedValues:
    def test_max_model_len_1k1k(self):
        assert compute_max_model_len(1024, 1024) == 2248

    def test_max_model_len_1k8k(self):
        assert compute_max_model_len(1024, 8192) == 9416

    def test_max_model_len_8k1k(self):
        assert compute_max_model_len(8192, 1024) == 9416

    def test_max_num_tokens_low_conc(self):
        result = compute_max_num_tokens(1024, 1024, 4)
        assert result == 4096

    def test_max_num_tokens_capped(self):
        result = compute_max_num_tokens(1024, 1024, 64)
        assert result == 16384  # capped

    def test_kv_cache_fraction_short_seq(self):
        assert compute_kv_cache_fraction(1024, 1024) == 0.95

    def test_kv_cache_fraction_medium_seq(self):
        assert compute_kv_cache_fraction(1024, 8192) == 0.70

    def test_kv_cache_fraction_long_input(self):
        assert compute_kv_cache_fraction(8192, 1024) == 0.70


# ---------------------------------------------------------------------------
# YAML rendering
# ---------------------------------------------------------------------------


class TestGenerateConfig:
    def test_output_is_valid_yaml(self, template_path):
        output = generate_config(
            tp=8, ep=1, moe_tp=8, moe_ep=1,
            conc=16, isl=1024, osl=1024,
            template_path=template_path,
        )
        config = yaml.safe_load(output)
        assert isinstance(config, dict)

    def test_tp8_pure_tensor_parallel(self, template_path):
        output = generate_config(
            tp=8, ep=1, moe_tp=8, moe_ep=1,
            conc=16, isl=1024, osl=1024,
            template_path=template_path,
        )
        config = yaml.safe_load(output)
        assert config["tensor_parallel_size"] == 8
        assert "moe_expert_parallel_size" not in config
        assert config["enable_attention_dp"] is False

    def test_ep8_pure_expert_parallel(self, template_path):
        output = generate_config(
            tp=8, ep=8, moe_tp=1, moe_ep=8,
            conc=16, isl=1024, osl=1024,
            template_path=template_path,
        )
        config = yaml.safe_load(output)
        assert config["tensor_parallel_size"] == 8
        assert config["moe_expert_parallel_size"] == 8
        assert config["moe_tensor_parallel_size"] == 1

    def test_hybrid_tp4_ep2(self, template_path):
        output = generate_config(
            tp=8, ep=2, moe_tp=4, moe_ep=2,
            conc=16, isl=1024, osl=1024,
            template_path=template_path,
        )
        config = yaml.safe_load(output)
        assert config["moe_tensor_parallel_size"] == 4
        assert config["moe_expert_parallel_size"] == 2

    def test_dp_attention_enabled(self, template_path):
        output = generate_config(
            tp=8, ep=8, moe_tp=1, moe_ep=8,
            dp_attention=True, conc=16, isl=1024, osl=1024,
            template_path=template_path,
        )
        config = yaml.safe_load(output)
        assert config["enable_attention_dp"] is True

    def test_kv_cache_config_present(self, template_path):
        output = generate_config(
            tp=8, ep=1, moe_tp=8, moe_ep=1,
            conc=16, isl=1024, osl=1024,
            template_path=template_path,
        )
        config = yaml.safe_load(output)
        assert "kv_cache_config" in config
        assert config["kv_cache_config"]["dtype"] == "fp8"
        assert isinstance(config["kv_cache_config"]["free_gpu_memory_fraction"], float)

    def test_moe_backend_trtllm(self, template_path):
        output = generate_config(
            tp=8, ep=1, moe_tp=8, moe_ep=1,
            moe_backend="TRTLLM",
            template_path=template_path,
        )
        config = yaml.safe_load(output)
        assert config["moe_config"]["backend"] == "TRTLLM"

    def test_moe_backend_cutlass(self, template_path):
        output = generate_config(
            tp=8, ep=1, moe_tp=8, moe_ep=1,
            moe_backend="CUTLASS",
            template_path=template_path,
        )
        config = yaml.safe_load(output)
        assert config["moe_config"]["backend"] == "CUTLASS"

    def test_invalid_moe_backend(self, template_path):
        with pytest.raises(ConfigError, match="moe_backend must be one of"):
            generate_config(
                tp=8, ep=1, moe_tp=8, moe_ep=1,
                moe_backend="INVALID",
                template_path=template_path,
            )

    def test_high_conc_cuda_graph(self, template_path):
        output = generate_config(
            tp=8, ep=1, moe_tp=8, moe_ep=1,
            conc=64, isl=1024, osl=1024,
            template_path=template_path,
        )
        config = yaml.safe_load(output)
        assert config["cuda_graph_config"]["max_batch_size"] == 64

    def test_all_8_research_configs(self, template_path):
        """All 8 configs from Research.md Section 4.7 render without error."""
        configs = [
            {"tp": 8, "ep": 1, "moe_tp": 8, "moe_ep": 1, "dp_attention": False},
            {"tp": 8, "ep": 8, "moe_tp": 1, "moe_ep": 8, "dp_attention": False},
            {"tp": 8, "ep": 2, "moe_tp": 4, "moe_ep": 2, "dp_attention": False},
            {"tp": 8, "ep": 4, "moe_tp": 2, "moe_ep": 4, "dp_attention": False},
            {"tp": 4, "ep": 1, "moe_tp": 4, "moe_ep": 1, "dp_attention": False},
            {"tp": 2, "ep": 1, "moe_tp": 2, "moe_ep": 1, "dp_attention": False},
            {"tp": 8, "ep": 8, "moe_tp": 1, "moe_ep": 8, "dp_attention": True},
            {"tp": 8, "ep": 2, "moe_tp": 4, "moe_ep": 2, "dp_attention": True},
        ]
        for cfg in configs:
            output = generate_config(**cfg, conc=16, isl=1024, osl=1024, template_path=template_path)
            parsed = yaml.safe_load(output)
            assert parsed is not None, f"Failed to parse config: {cfg}"
            assert parsed["tensor_parallel_size"] == cfg["tp"]
