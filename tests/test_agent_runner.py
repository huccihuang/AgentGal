"""测试 PydanticAI runner wrapper。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import BaseModel

project_root = Path(__file__).parent.parent
os.chdir(project_root)

try:
    import agents.runner as agent_runner_module
except ModuleNotFoundError as exc:
    pytest.skip(f"skip agent runner tests: missing dependency ({exc})", allow_module_level=True)


class _StructuredOutput(BaseModel):
    content: str


class _FakeResult:
    def __init__(self, output, *, usage=None):
        self.output = output
        self.response = "raw-response"
        self._usage = usage

    def usage(self):
        return self._usage


class _FakeUsage:
    """模拟 pydantic-ai RunUsage 的字段子集。"""

    def __init__(self, input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0, requests=1):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_write_tokens = cache_write_tokens
        self.requests = requests


class _FakeAgent:
    def __init__(
        self,
        result: _FakeResult | BaseException | list[_FakeResult | BaseException],
    ):
        self._results = result if isinstance(result, list) else [result]
        self.calls: list[str] = []
        self.metadata_calls: list[dict[str, str]] = []

    async def run(self, user_input: str, *, metadata: dict[str, str] | None = None):
        self.calls.append(user_input)
        self.metadata_calls.append(metadata or {})
        index = min(len(self.calls) - 1, len(self._results) - 1)
        result = self._results[index]
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.mark.asyncio
async def test_run_text_agent_returns_stripped_text():
    agent = _FakeAgent(_FakeResult("  hello  "))
    output = await agent_runner_module.run_text_agent(
        agent=agent,
        user_input="hi",
        timeout_seconds=1,
        workflow_name="wf",
        trace_metadata={"agent_name": "tester"},
        usage_agent="tester",
        usage_phase="agent_run",
        model_name="deepseek-chat",
    )

    assert output == "hello"
    assert agent.calls == ["hi"]
    assert agent.metadata_calls == [
        {
            "workflow_name": "wf",
            "usage_agent": "tester",
            "usage_phase": "agent_run",
            "model_name": "deepseek-chat",
            "agent_name": "tester",
        }
    ]


@pytest.mark.asyncio
async def test_run_text_agent_retries_until_success():
    agent = _FakeAgent([RuntimeError("temporary"), _FakeResult("  hello  ")])

    output = await agent_runner_module.run_text_agent(
        agent=agent,
        user_input="hi",
        timeout_seconds=1,
        workflow_name="wf",
        trace_metadata=None,
        usage_agent="tester",
        usage_phase="agent_run",
        model_name="deepseek-chat",
    )

    assert output == "hello"
    assert agent.calls == ["hi", "hi"]


@pytest.mark.asyncio
async def test_run_structured_agent_returns_typed_output():
    expected = _StructuredOutput(content="ok")
    agent = _FakeAgent(_FakeResult(expected))
    output = await agent_runner_module.run_structured_agent(
        agent=agent,
        user_input="hi",
        output_type=_StructuredOutput,
        timeout_seconds=1,
        workflow_name="wf",
        trace_metadata=None,
        usage_agent="tester",
        usage_phase="agent_run",
        model_name="deepseek-chat",
    )

    assert output == expected
    assert agent.metadata_calls == [
        {
            "workflow_name": "wf",
            "usage_agent": "tester",
            "usage_phase": "agent_run",
            "model_name": "deepseek-chat",
        }
    ]


@pytest.mark.asyncio
async def test_run_structured_agent_retries_until_success():
    expected = _StructuredOutput(content="ok")
    agent = _FakeAgent(
        [
            RuntimeError("temporary"),
            RuntimeError("temporary again"),
            _FakeResult(expected),
        ]
    )

    output = await agent_runner_module.run_structured_agent(
        agent=agent,
        user_input="hi",
        output_type=_StructuredOutput,
        timeout_seconds=1,
        workflow_name="wf",
        trace_metadata=None,
        usage_agent="tester",
        usage_phase="agent_run",
        model_name="deepseek-chat",
    )

    assert output == expected
    assert agent.calls == ["hi", "hi", "hi"]


@pytest.mark.asyncio
async def test_run_structured_agent_retries_output_type_validation():
    expected = _StructuredOutput(content="ok")
    agent = _FakeAgent([_FakeResult("not-json"), _FakeResult(expected)])

    output = await agent_runner_module.run_structured_agent(
        agent=agent,
        user_input="hi",
        output_type=_StructuredOutput,
        timeout_seconds=1,
        workflow_name="wf",
        trace_metadata=None,
        usage_agent="tester",
        usage_phase="agent_run",
        model_name="deepseek-chat",
    )

    assert output == expected
    assert agent.calls == ["hi", "hi"]


@pytest.mark.asyncio
async def test_run_structured_agent_retries_dynamic_output_validation():
    accepted = _StructuredOutput(content="ok")
    agent = _FakeAgent(
        [
            _FakeResult(_StructuredOutput(content="bad")),
            _FakeResult(accepted),
        ]
    )

    def validate_output(output: _StructuredOutput) -> None:
        if output.content != "ok":
            raise ValueError("dynamic output validation failed")

    output = await agent_runner_module.run_structured_agent(
        agent=agent,
        user_input="hi",
        output_type=_StructuredOutput,
        timeout_seconds=1,
        workflow_name="wf",
        trace_metadata=None,
        usage_agent="tester",
        usage_phase="agent_run",
        model_name="deepseek-chat",
        output_validator=validate_output,
    )

    assert output == accepted
    assert agent.calls == ["hi", "hi"]


@pytest.mark.asyncio
async def test_run_structured_agent_stops_after_three_attempts():
    agent = _FakeAgent(RuntimeError("still down"))

    with pytest.raises(RuntimeError):
        await agent_runner_module.run_structured_agent(
            agent=agent,
            user_input="hi",
            output_type=_StructuredOutput,
            timeout_seconds=1,
            workflow_name="wf",
            trace_metadata=None,
            usage_agent="tester",
            usage_phase="agent_run",
            model_name="deepseek-chat",
        )

    assert agent.calls == ["hi", "hi", "hi"]


@pytest.mark.asyncio
async def test_usage_is_captured_when_accumulator_bound():
    """绑定累加器后，runner 应把 result.usage() 累加进去。"""
    usage = _FakeUsage(input_tokens=1000, output_tokens=200, cache_read_tokens=400)
    agent = _FakeAgent(_FakeResult("hello", usage=usage))

    acc = agent_runner_module.UsageAccumulator()
    token = agent_runner_module.bind_accumulator(acc)
    try:
        await agent_runner_module.run_text_agent(
            agent=agent,
            user_input="hi",
            timeout_seconds=1,
            workflow_name="wf",
            trace_metadata=None,
            usage_agent="tester",
            usage_phase="agent_run",
            model_name="deepseek-chat",
        )
    finally:
        agent_runner_module.reset_accumulator(token)

    assert acc.input_tokens == 1000
    assert acc.output_tokens == 200
    assert acc.cache_read_tokens == 400
    assert acc.total_tokens == 1200
    # 同一 phase 应在 by_phase 里聚合
    assert acc.by_phase["agent_run"]["input"] == 1000
    assert acc.by_phase["agent_run"]["output"] == 200
    assert acc.by_phase["agent_run"]["cache_read"] == 400
    # per-phase model 来自 metadata.model_name，用于多模型场景按 phase 计价
    assert acc.by_phase["agent_run"]["model"] == "deepseek-chat"


@pytest.mark.asyncio
async def test_usage_capture_no_op_when_accumulator_not_bound():
    """没有绑定累加器时不应抛错。"""
    usage = _FakeUsage(input_tokens=1000, output_tokens=200)
    agent = _FakeAgent(_FakeResult("hello", usage=usage))

    # 确保没有累加器
    assert agent_runner_module.get_accumulator() is None
    output = await agent_runner_module.run_text_agent(
        agent=agent,
        user_input="hi",
        timeout_seconds=1,
        workflow_name="wf",
        trace_metadata=None,
        usage_agent="tester",
        usage_phase="agent_run",
        model_name="deepseek-chat",
    )
    assert output == "hello"


@pytest.mark.asyncio
async def test_run_structured_agent_raises_on_unexpected_output_type():
    agent = _FakeAgent(_FakeResult("not-json"))

    with pytest.raises(TypeError):
        await agent_runner_module.run_structured_agent(
            agent=agent,
            user_input="hi",
            output_type=_StructuredOutput,
            timeout_seconds=1,
            workflow_name="wf",
            trace_metadata=None,
            usage_agent="tester",
            usage_phase="agent_run",
            model_name="deepseek-chat",
        )

    assert agent.calls == ["hi", "hi", "hi"]
