"""LLM 配置 - 使用 OpenAI-compatible API URL、模型 ID 和 API key。"""

import os

from shared.config import AGENT_TEMPERATURE


_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"


def _get_required_env(key: str) -> str:
    """获取必需的环境变量，不存在则抛出错误"""
    value = _get_optional_env(key)
    if not value:
        raise ValueError(f"{key} not set in environment")
    return value


def _get_optional_env(key: str) -> str | None:
    value = os.getenv(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


# 公共暴露：供 server / pricing 等同步读取，避免 os.getenv 散落
LLM_MODEL_ID: str = _get_optional_env("LLM_MODEL_ID") or ""


def _normalize_api_url(api_url: str) -> str:
    """归一化 OpenAI 兼容 Base URL，避免重复拼接 chat/completions。"""
    normalized = api_url.rstrip("/")
    if normalized.endswith(_CHAT_COMPLETIONS_SUFFIX):
        normalized = normalized[: -len(_CHAT_COMPLETIONS_SUFFIX)]
    return normalized


def get_llm_config(
    model_id: str | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
    temperature: float | None = None,
) -> dict:
    """返回 LLM 配置 dict，供 agent_factory 构建 OpenAI-compatible chat model 使用。

    参数优先级：传入参数 > 环境变量。temperature 不为 None 时覆盖默认值。

    Returns:
        {
            "api_url": str,
            "api_key": str,
            "model_id": str,
            "temperature": float,
            "provider": str,
        }
    """
    provider = _get_optional_env("LLM_PROVIDER") or "openai"
    model_id = _normalize_optional(model_id) or _get_required_env("LLM_MODEL_ID")
    api_key = _normalize_optional(api_key) or _get_required_env("LLM_API_KEY")
    raw_url = _normalize_optional(api_url) or _get_optional_env("LLM_API_URL")
    return {
        "api_url": _normalize_api_url(raw_url) if raw_url else "",
        "api_key": api_key,
        "model_id": model_id,
        "temperature": AGENT_TEMPERATURE if temperature is None else temperature,
        "provider": provider,
    }
