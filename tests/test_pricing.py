"""测试 agents/pricing.py 的成本估算。

价表从 config.toml [pricing.models] 加载（无 env 覆写）。
"""

from __future__ import annotations

import pytest

from agents import pricing


def test_unknown_model_returns_none():
    assert pricing.calculate_cost("totally-fake-model-xyz", 1000, 100) is None


def test_known_model_basic_math():
    # deepseek-v4-flash: 0.14 / 0.28 / 0.0028 per 1M
    cost = pricing.calculate_cost(
        "deepseek-v4-flash", input_tokens=1_000_000, output_tokens=0, cache_read_tokens=0
    )
    assert cost == pytest.approx(0.14, rel=1e-6)


def test_cache_read_uses_cache_price():
    # deepseek-v4-flash: input 0.14, cache_read 0.0028
    # 1M input 全部命中 cache → 应按 0.0028 计价，而非 0.14
    cost = pricing.calculate_cost("deepseek-v4-flash", 1_000_000, 0, cache_read_tokens=1_000_000)
    assert cost == pytest.approx(0.0028, rel=1e-6)


def test_mixed_input_and_cache():
    # 600k 命中 cache (0.0028/M)，400k 未命中 (0.14/M)，输出 100k (0.28/M)
    cost = pricing.calculate_cost("deepseek-v4-flash", 1_000_000, 100_000, 600_000)
    expected = 400_000 * 0.14 / 1e6 + 600_000 * 0.0028 / 1e6 + 100_000 * 0.28 / 1e6
    assert cost == pytest.approx(expected, rel=1e-6)


def test_prefix_match():
    # deepseek-v4-flash-20260101 应当匹配到 deepseek-v4-flash
    cost = pricing.calculate_cost("deepseek-v4-flash-20260101", 1_000_000, 0, 0)
    assert cost == pytest.approx(0.14, rel=1e-6)


def test_empty_model_id_returns_none():
    assert pricing.calculate_cost("", 1000, 100) is None
