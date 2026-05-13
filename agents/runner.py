"""统一的 Agent 运行层。"""

from __future__ import annotations

import asyncio
import types
import typing
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, TypeVar

from log_config.routing import routing_logger
from shared.config import AGENT_RUN_MAX_ATTEMPTS

T = TypeVar("T")


@dataclass
class UsageAccumulator:
    """单轮对话累计 token usage。runner 每次 agent.run 成功后调用 add()。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    requests: int = 0
    # 每个 phase 桶混存 int（token 计数）和 str（model 标签），用 Any 简化。
    by_phase: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, usage: Any, phase: str = "", model_name: str = "") -> None:
        """从 pydantic-ai RunUsage 实例累加；字段缺失时静默忽略。

        model_name 用于 per-phase 计价：同一会话出现多模型时，按 phase 分别按
        正确单价折算成本（见 server._aggregate_cost）。
        """
        if usage is None:
            return
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out = int(getattr(usage, "output_tokens", 0) or 0)
        cr = int(getattr(usage, "cache_read_tokens", 0) or 0)
        cw = int(getattr(usage, "cache_write_tokens", 0) or 0)
        self.input_tokens += inp
        self.output_tokens += out
        self.cache_read_tokens += cr
        self.cache_write_tokens += cw
        self.requests += int(getattr(usage, "requests", 1) or 1)
        if phase:
            bucket = self.by_phase.setdefault(
                phase, {"input": 0, "output": 0, "cache_read": 0, "model": ""}
            )
            bucket["input"] = int(bucket.get("input", 0)) + inp
            bucket["output"] = int(bucket.get("output", 0)) + out
            bucket["cache_read"] = int(bucket.get("cache_read", 0)) + cr
            if model_name and not bucket.get("model"):
                bucket["model"] = model_name

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# 当前轮的累加器；server.py 在 chat_stream 开始时 set，结束后 reset。
_current_accumulator: ContextVar[UsageAccumulator | None] = ContextVar(
    "agentgal_round_usage_accumulator", default=None
)


def bind_accumulator(acc: UsageAccumulator | None) -> Token:
    """绑定当前 asyncio 上下文的累加器；返回 token 用于 reset。"""
    return _current_accumulator.set(acc)


def reset_accumulator(token: Token) -> None:
    """恢复累加器绑定。

    在 async generator 被外部 aclose() 时，finally 可能在不同的 Context 中执行，
    导致 token 与当前 Context 不匹配抛 ValueError。此时退化为在当前 Context 中
    清空绑定即可（generator 已结束，无需精确还原）。
    """
    try:
        _current_accumulator.reset(token)
    except ValueError:
        _current_accumulator.set(None)


def get_accumulator() -> UsageAccumulator | None:
    return _current_accumulator.get()


def _matches_output_type(value: Any, output_type: Any) -> bool:
    """兼容 UnionType (X | Y) 与参数化泛型 (dict[str, int]) 的 isinstance 检查。

    Python 内建 isinstance 第二参数不接受参数化泛型，会抛 TypeError；
    这里把 union 拆成各 arm 递归，把泛型退化到 origin 后再做常规 isinstance。
    """
    origin = typing.get_origin(output_type)
    if origin is types.UnionType or origin is typing.Union:
        return any(_matches_output_type(value, arg) for arg in typing.get_args(output_type))
    if origin is not None:
        return isinstance(value, origin)
    return isinstance(value, output_type)


def _build_run_metadata(
    workflow_name: str,
    usage_agent: str,
    usage_phase: str,
    model_name: str,
    trace_metadata: dict[str, str] | None,
) -> dict[str, str]:
    metadata = {
        "workflow_name": workflow_name,
        "usage_agent": usage_agent,
        "usage_phase": usage_phase,
        "model_name": model_name,
    }
    if trace_metadata:
        metadata.update(trace_metadata)
    return metadata


async def _run_agent_with_retries(
    *,
    agent,
    user_input: str,
    metadata: dict[str, str],
    timeout_seconds: float,
    label: str,
    on_result,
    max_attempts: int = AGENT_RUN_MAX_ATTEMPTS,
) -> Any:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    for attempt in range(1, max_attempts + 1):
        try:
            result = await asyncio.wait_for(
                agent.run(user_input, metadata=metadata),
                timeout=timeout_seconds,
            )
            acc = get_accumulator()
            if acc is not None:
                try:
                    acc.add(
                        result.usage(),
                        phase=metadata.get("usage_phase", ""),
                        model_name=metadata.get("model_name", ""),
                    )
                except Exception as usage_err:  # noqa: BLE001
                    routing_logger.warning(
                        "[%s] usage 抽取失败（忽略）: %s", label, usage_err
                    )
            return on_result(result)
        except Exception as exc:
            exc_desc = (
                f"超时（>{timeout_seconds}s）"
                if isinstance(exc, asyncio.TimeoutError)
                else f"失败: {exc}"
            )
            if attempt >= max_attempts:
                routing_logger.error(
                    "[%s] LLM 调用第 %s/%s 次%s，停止重试",
                    label,
                    attempt,
                    max_attempts,
                    exc_desc,
                )
                raise
            routing_logger.warning(
                "[%s] LLM 调用第 %s/%s 次%s，准备重试",
                label,
                attempt,
                max_attempts,
                exc_desc,
            )

    raise RuntimeError("unreachable LLM retry state")


async def run_text_agent(
    *,
    agent,
    user_input: str,
    timeout_seconds: float,
    workflow_name: str,
    trace_metadata: dict[str, str] | None,
    usage_agent: str,
    usage_phase: str,
    model_name: str,
    error_label: str | None = None,
    max_attempts: int = AGENT_RUN_MAX_ATTEMPTS,
) -> str:
    """执行文本 Agent，返回原始字符串输出。"""
    label = error_label or usage_agent

    def _extract_text(result) -> str:
        output = result.output
        if not isinstance(output, str):
            routing_logger.error(f"[{label}] 文本输出类型异常: {type(output)!r}")
            raise TypeError(f"{label} expected str output, got {type(output)!r}")
        return output.strip()

    return await _run_agent_with_retries(
        agent=agent,
        user_input=user_input,
        metadata=_build_run_metadata(
            workflow_name, usage_agent, usage_phase, model_name, trace_metadata
        ),
        timeout_seconds=timeout_seconds,
        label=label,
        on_result=_extract_text,
        max_attempts=max_attempts,
    )


async def run_structured_agent(
    *,
    agent,
    user_input: str,
    output_type: type[T],
    timeout_seconds: float,
    workflow_name: str,
    trace_metadata: dict[str, str] | None,
    usage_agent: str,
    usage_phase: str,
    model_name: str,
    error_label: str | None = None,
    max_attempts: int = AGENT_RUN_MAX_ATTEMPTS,
    output_validator: Callable[[T], None] | None = None,
) -> T:
    """执行结构化 Agent，并统一处理超时、用量日志和 typed parse。"""
    label = error_label or usage_agent

    def _extract_structured(result) -> T:
        output = result.output
        if _matches_output_type(output, output_type):
            if output_validator is not None:
                output_validator(output)
            return output

        routing_logger.error(
            f"[{label}] structured output 类型异常: expected={output_type!r}, got={type(output)!r}, raw={result.response!r}"
        )
        raise TypeError(f"{label} expected {output_type!r}, got {type(output)!r}")

    return await _run_agent_with_retries(
        agent=agent,
        user_input=user_input,
        metadata=_build_run_metadata(
            workflow_name, usage_agent, usage_phase, model_name, trace_metadata
        ),
        timeout_seconds=timeout_seconds,
        label=label,
        on_result=_extract_structured,
        max_attempts=max_attempts,
    )
