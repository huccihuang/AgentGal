"""测试 server._build_token_usage_payload：字段、cumulative 推进、持久化触发。"""

from __future__ import annotations

from pathlib import Path

import pytest

# 导入 server 会触发 load_dotenv + FastAPI 实例化；不会执行 startup 事件。
try:
    import server as server_module
except Exception as exc:  # noqa: BLE001  # 服务器实例化失败时跳过整批
    pytest.skip(f"skip token usage payload tests: server import failed ({exc})", allow_module_level=True)

from agents.runner import UsageAccumulator
from storage import token_usage as token_usage_module


@pytest.fixture
def fresh_session(monkeypatch, tmp_path: Path):
    """每个用例都从干净的 session 累计 + 临时持久化路径开始。"""
    monkeypatch.setattr(
        token_usage_module,
        "_TOKEN_USAGE_PATH",
        tmp_path / "narrator" / ".token_usage.jsonl",
    )
    # 重置 server 的会话级状态
    server_module._round_counter = 0
    server_module._session_cumulative.update(
        {"input": 0, "output": 0, "cache_read": 0, "cost": 0.0, "total": 0}
    )
    # 固定模型以便断言 cost；LLM_MODEL_ID 在 server import 时已绑定为模块级常量，
    # env 变更对它无效，必须直接 monkeypatch 模块属性。
    monkeypatch.setattr(server_module, "LLM_MODEL_ID", "deepseek-v4-flash")
    yield


def _acc(input_tokens=1000, output_tokens=200, cache_read=0):
    a = UsageAccumulator()
    a.input_tokens = input_tokens
    a.output_tokens = output_tokens
    a.cache_read_tokens = cache_read
    return a


def test_payload_basic_fields(fresh_session):
    acc = _acc(input_tokens=1000, output_tokens=200, cache_read=400)
    payload = server_module._build_token_usage_payload(acc, round_id=1, is_final=False, turn=0)
    assert payload["input"] == 1000
    assert payload["output"] == 200
    assert payload["cache_read"] == 400
    assert payload["total"] == 1200
    # delta = uncached_input + output = 600 + 200 = 800
    assert payload["delta"] == 800
    assert payload["round_id"] == 1
    assert payload["turn"] == 0
    assert payload["is_final"] is False
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["cost"] is not None
    assert payload["cumulative"] == {
        "input": 0, "output": 0, "cache_read": 0, "total": 0, "cost": 0.0,
    }


def test_intermediate_does_not_advance_cumulative(fresh_session):
    acc = _acc(1000, 200, 0)
    server_module._build_token_usage_payload(acc, round_id=1, is_final=False, turn=5)
    assert server_module._session_cumulative["input"] == 0
    assert server_module._session_cumulative["output"] == 0


def test_final_advances_cumulative(fresh_session):
    acc1 = _acc(1000, 200, 0)
    server_module._build_token_usage_payload(acc1, round_id=1, is_final=True, turn=5)
    acc2 = _acc(500, 50, 0)
    payload = server_module._build_token_usage_payload(acc2, round_id=2, is_final=True, turn=6)
    assert server_module._session_cumulative["input"] == 1500
    assert server_module._session_cumulative["output"] == 250
    assert server_module._session_cumulative["total"] == 1750
    # payload 里的 cumulative 应反映"包含本轮"的累计
    assert payload["cumulative"]["input"] == 1500
    assert payload["cumulative"]["output"] == 250


def test_persists_only_when_final_and_turn_positive(fresh_session):
    acc = _acc(1000, 200, 0)
    # is_final=False → 不写
    server_module._build_token_usage_payload(acc, round_id=1, is_final=False, turn=5)
    assert token_usage_module.read_token_usage() == []
    # is_final=True 但 turn=0 → 不写
    server_module._build_token_usage_payload(acc, round_id=2, is_final=True, turn=0)
    assert token_usage_module.read_token_usage() == []
    # is_final=True 且 turn>0 → 写
    server_module._build_token_usage_payload(acc, round_id=3, is_final=True, turn=7)
    rows = token_usage_module.read_token_usage()
    assert len(rows) == 1
    assert rows[0]["turn"] == 7


def test_unknown_model_cost_is_none(fresh_session, monkeypatch):
    monkeypatch.setattr(server_module, "LLM_MODEL_ID", "totally-fake-zzz")
    payload = server_module._build_token_usage_payload(
        _acc(1000, 200, 0), round_id=1, is_final=True, turn=1
    )
    assert payload["cost"] is None
    # cumulative.cost 也不应被 None 污染（保持累计为 0）
    assert server_module._session_cumulative["cost"] == 0.0


def test_per_phase_pricing_sums_correctly(fresh_session):
    """by_phase 自带 model_name 时，按 phase 独立计价后汇总。"""
    acc = UsageAccumulator()
    # phase A: claude-sonnet-4.6，1M input(全 cache miss)，0 output
    acc.by_phase["narrator"] = {
        "input": 1_000_000, "output": 0, "cache_read": 0, "model": "claude-sonnet-4.6",
    }
    # phase B: deepseek-v4-flash，0 input，1M output
    acc.by_phase["character"] = {
        "input": 0, "output": 1_000_000, "cache_read": 0, "model": "deepseek-v4-flash",
    }
    acc.input_tokens = 1_000_000
    acc.output_tokens = 1_000_000
    payload = server_module._build_token_usage_payload(
        acc, round_id=1, is_final=False, turn=0
    )
    # 3 (sonnet-4.6 input 1M) + 0.28 (v4-flash output 1M) = 3.28
    assert payload["cost"] == pytest.approx(3.28, rel=1e-3)


def test_per_phase_unknown_model_falls_back_to_global(fresh_session, monkeypatch):
    """phase 没带 model_name，使用 global LLM_MODEL_ID 计价。"""
    monkeypatch.setattr(server_module, "LLM_MODEL_ID", "deepseek-v4-flash")
    acc = UsageAccumulator()
    acc.by_phase["narrator"] = {
        "input": 1_000_000, "output": 0, "cache_read": 0, "model": "",
    }
    acc.input_tokens = 1_000_000
    payload = server_module._build_token_usage_payload(
        acc, round_id=1, is_final=False, turn=0
    )
    # deepseek-v4-flash @ 0.14 USD per 1M input
    assert payload["cost"] == pytest.approx(0.14, rel=1e-3)


def test_aggregate_cost_all_unknown_returns_none(fresh_session, monkeypatch):
    """所有 phase 的 model 都未知 → cost = None（与原 unknown fallback 一致）。"""
    monkeypatch.setattr(server_module, "LLM_MODEL_ID", "")
    acc = UsageAccumulator()
    acc.by_phase["narrator"] = {
        "input": 1_000_000, "output": 0, "cache_read": 0, "model": "unknown-foo",
    }
    acc.input_tokens = 1_000_000
    payload = server_module._build_token_usage_payload(
        acc, round_id=1, is_final=False, turn=0
    )
    assert payload["cost"] is None


def test_recompute_token_row_cost_uses_current_pricing(fresh_session, monkeypatch):
    """历史行的 cost=null，read 侧应用当前定价表重算。"""
    legacy_row = {
        "turn": 5, "round_id": 5, "is_final": True,
        "input": 1_000_000, "output": 0, "cache_read": 0,
        "total": 1_000_000, "delta": 1_000_000,
        "cost": None,                 # 旧版本写入时未命中定价表
        "model": "deepseek-v4-flash",
        "cumulative": {"input": 0, "output": 0, "cache_read": 0, "total": 0, "cost": 0.0},
    }
    refreshed = server_module._recompute_token_row_cost(legacy_row)
    # config.toml 里 deepseek-v4-flash: input 0.14 USD per 1M
    assert refreshed["cost"] == pytest.approx(0.14, rel=1e-3)
    # 原行不被改写
    assert legacy_row["cost"] is None


def test_recompute_falls_back_to_global_when_row_model_missing(fresh_session, monkeypatch):
    """行内 model 为空时，fallback 到当前 LLM_MODEL_ID。"""
    monkeypatch.setattr(server_module, "LLM_MODEL_ID", "deepseek-v4-flash")
    legacy_row = {
        "turn": 6, "round_id": 6, "is_final": True,
        "input": 1_000_000, "output": 0, "cache_read": 0,
        "total": 1_000_000, "delta": 1_000_000,
        "cost": None,
        "model": "",  # 旧行缺失 model
        "cumulative": {"input": 0, "output": 0, "cache_read": 0, "total": 0, "cost": 0.0},
    }
    refreshed = server_module._recompute_token_row_cost(legacy_row)
    assert refreshed["cost"] == pytest.approx(0.14, rel=1e-3)
