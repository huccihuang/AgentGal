"""测试 storage/token_usage.py 的 append / read / clear。"""

from __future__ import annotations

from pathlib import Path

import pytest

from storage import token_usage as token_usage_module


@pytest.fixture
def usage_path(tmp_path: Path, monkeypatch):
    """把持久化路径指到临时目录。"""
    path = tmp_path / "narrator" / ".token_usage.jsonl"
    monkeypatch.setattr(token_usage_module, "_TOKEN_USAGE_PATH", path)
    return path


def _make_record(turn: int, total: int = 100, **extra) -> dict:
    rec = {
        "turn": turn,
        "round_id": turn,
        "is_final": True,
        "delta": total,
        "input": 80,
        "output": 20,
        "cache_read": 0,
        "total": total,
        "cost": 0.001,
        "model": "deepseek-chat",
        "cumulative": {"input": 80, "output": 20, "cache_read": 0, "total": total, "cost": 0.001},
    }
    rec.update(extra)
    return rec


def test_read_empty_when_no_file(usage_path):
    assert not usage_path.exists()
    assert token_usage_module.read_token_usage() == []


def test_append_and_read_roundtrip(usage_path):
    token_usage_module.append_token_usage(_make_record(1, total=120))
    token_usage_module.append_token_usage(_make_record(2, total=240))
    rows = token_usage_module.read_token_usage()
    assert [r["turn"] for r in rows] == [1, 2]
    assert rows[0]["total"] == 120
    assert rows[1]["total"] == 240


def test_turn_zero_is_skipped(usage_path):
    token_usage_module.append_token_usage(_make_record(0, total=999))
    assert not usage_path.exists()
    assert token_usage_module.read_token_usage() == []


def test_dedupe_keeps_last_for_same_turn(usage_path):
    token_usage_module.append_token_usage(_make_record(5, total=100))
    token_usage_module.append_token_usage(_make_record(5, total=300))
    rows = token_usage_module.read_token_usage()
    assert len(rows) == 1
    assert rows[0]["total"] == 300


def test_min_max_turn_filters(usage_path):
    for t in [1, 2, 3, 4, 5]:
        token_usage_module.append_token_usage(_make_record(t, total=t * 10))
    rows = token_usage_module.read_token_usage(min_turn=2, max_turn=4)
    assert [r["turn"] for r in rows] == [2, 3, 4]


def test_read_skips_invalid_lines(usage_path):
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    usage_path.write_text(
        '{"turn": 1, "input": 80, "output": 20, "cache_read": 0, "total": 100}\n'
        "not-valid-json\n"
        '{"turn": 2, "input": 160, "output": 40, "cache_read": 0, "total": 200}\n',
        encoding="utf-8",
    )
    rows = token_usage_module.read_token_usage()
    assert [r["turn"] for r in rows] == [1, 2]


def test_append_skips_zero_token_row(usage_path):
    """全 0 行不写盘（防御：边角路径不污染历史）。"""
    rec = _make_record(5, total=0)
    rec.update({"input": 0, "output": 0, "cache_read": 0, "total": 0, "delta": 0})
    token_usage_module.append_token_usage(rec)
    assert not usage_path.exists() or usage_path.stat().st_size == 0
    assert token_usage_module.read_token_usage() == []


def test_read_skips_zero_token_rows(usage_path):
    """读侧也跳过全 0 行（兼容历史脏数据）。"""
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    usage_path.write_text(
        '{"turn": 1, "input": 100, "output": 50, "cache_read": 0, "total": 150}\n'
        '{"turn": 2, "input": 0, "output": 0, "cache_read": 0, "total": 0}\n'
        '{"turn": 3, "input": 0, "output": 0, "cache_read": 0, "total": 0}\n'
        '{"turn": 4, "input": 200, "output": 100, "cache_read": 50, "total": 300}\n',
        encoding="utf-8",
    )
    rows = token_usage_module.read_token_usage()
    assert [r["turn"] for r in rows] == [1, 4]


def test_clear_removes_file(usage_path):
    token_usage_module.append_token_usage(_make_record(1))
    assert usage_path.exists()
    token_usage_module.clear_token_usage()
    assert not usage_path.exists()


def test_clear_is_idempotent_when_missing(usage_path):
    assert not usage_path.exists()
    token_usage_module.clear_token_usage()  # 不应抛
    assert not usage_path.exists()
