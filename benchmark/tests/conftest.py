# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for benchmark tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure benchmark/ is importable from tests
BENCHMARK_DIR = Path(__file__).resolve().parent.parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))


@pytest.fixture
def template_path() -> Path:
    return BENCHMARK_DIR / "parallel_config.yaml.j2"
