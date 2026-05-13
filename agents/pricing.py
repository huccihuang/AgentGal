"""LLM 调用计费。

提供基于模型 ID 的 USD 成本估算。价格从 `config.toml` 的 `[pricing.models]`
区段加载（每百万 token 价格，USD），分别覆盖 cache miss 输入 / 输出 /
cache hit 输入三类计费维度。

匹配规则:
1. 精确匹配 model_id
2. 否则按最长前缀匹配（兼容 deepseek-chat-20250915 这种带后缀的别名）
3. 仍未命中 → 返回 None，UI 显示 "N/A"，不影响主流程

不支持 env 覆写：自定义模型请直接在 config.toml 加表项。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass

from shared.config import PROJECT_ROOT


@dataclass(frozen=True)
class ModelPrice:
    input_per_m: float  # cache miss 输入价
    output_per_m: float
    cache_read_per_m: float | None = None  # None → 按 input_per_m 计价


def _load_pricing_table() -> dict[str, ModelPrice]:
    """从 config.toml [pricing.models] 加载价表，模型 key 归一化为小写。"""
    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    models = cfg.get("pricing", {}).get("models", {})
    table: dict[str, ModelPrice] = {}
    for key, params in models.items():
        if not isinstance(params, dict):
            continue
        try:
            table[key.strip().lower()] = ModelPrice(
                input_per_m=float(params["input_per_m"]),
                output_per_m=float(params["output_per_m"]),
                cache_read_per_m=(
                    float(params["cache_read_per_m"])
                    if params.get("cache_read_per_m") is not None
                    else None
                ),
            )
        except (KeyError, ValueError, TypeError):
            continue
    return table


PRICING_TABLE: dict[str, ModelPrice] = _load_pricing_table()


def _match_model(model_id: str) -> ModelPrice | None:
    """先精确匹配，再前缀匹配（容忍 deepseek-chat-2025xx 这种带后缀的别名）。"""
    if not model_id:
        return None
    key = model_id.strip().lower()
    if key in PRICING_TABLE:
        return PRICING_TABLE[key]
    # 前缀匹配：取最长匹配项
    candidates = [k for k in PRICING_TABLE if key.startswith(k)]
    if not candidates:
        return None
    return PRICING_TABLE[max(candidates, key=len)]


def calculate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> float | None:
    """估算单次调用成本（USD）。未知模型返回 None。

    cache_read_tokens 已包含在 input_tokens 中（API 通常这样报告）。
    本函数会用 cache_read 单价覆盖该部分的 input 单价。
    """
    price = _match_model(model_id)
    if price is None:
        return None

    uncached_input = max(0, input_tokens - cache_read_tokens)
    cost_input = uncached_input * price.input_per_m / 1_000_000
    cache_unit = price.cache_read_per_m if price.cache_read_per_m is not None else price.input_per_m
    cost_cache = cache_read_tokens * cache_unit / 1_000_000
    cost_output = output_tokens * price.output_per_m / 1_000_000
    return cost_input + cost_cache + cost_output
