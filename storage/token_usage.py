"""每轮 token 用量的持久化。

写入：`/api/chat` 每轮最终 token_usage 事件发出时同步追加一行
读取：`/api/init` / `/api/history` / `/api/load` 把记录还原给前端，刷新页面不丢
清空：reset_game / load_save 调用

文件位置：`data/runtime/characters/narrator/.token_usage.jsonl`
每行：`{"turn": int, "delta": int, "input": int, "output": int,
        "cache_read": int, "total": int, "cost": float|null,
        "model": str, "cumulative": {...}}`

按 turn 去重，重复 turn 写入时保留最后一条（chat 流程内同 turn 不会重写，
但读取侧仍做容错去重）。
"""

from __future__ import annotations

import json
from pathlib import Path

from shared.config import CHARACTERS_DIR


_TOKEN_USAGE_PATH: Path = CHARACTERS_DIR / "narrator" / ".token_usage.jsonl"


def _ensure_parent() -> None:
    _TOKEN_USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)


def append_token_usage(record: dict) -> None:
    """追加一条 token 用量记录。同 turn 重复写入由读取侧去重处理。

    防御性：turn=0 哨兵 或 整轮零 token 行不写入（避免历史脏数据增长）。
    """
    turn = int(record.get("turn") or 0)
    if turn <= 0:
        # 0 是哨兵：narrator 未成功发言，没有有效 turn 锚点，不持久化。
        return
    inp = int(record.get("input") or 0)
    out = int(record.get("output") or 0)
    cr = int(record.get("cache_read") or 0)
    if inp == 0 and out == 0 and cr == 0:
        # 全 0 视为无效：narrator/character 都未真正消耗 token，
        # 通常来自边角失败路径（fake runner / 早期错误），不污染历史。
        return
    _ensure_parent()
    with _TOKEN_USAGE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_token_usage(min_turn: int | None = None, max_turn: int | None = None) -> list[dict]:
    """读取所有 token 用量记录；按 turn 升序、同 turn 保留最后一条。

    min_turn / max_turn 闭区间过滤；None 不限。
    跳过 turn<=0 与整轮零 token 行（兼容历史脏数据）。
    cost 字段保留行内持久化的值；调用方如需用当前定价表实时重算，自行处理。
    """
    if not _TOKEN_USAGE_PATH.exists():
        return []
    latest_by_turn: dict[int, dict] = {}
    with _TOKEN_USAGE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            turn = int(rec.get("turn") or 0)
            if turn <= 0:
                continue
            inp = int(rec.get("input") or 0)
            out = int(rec.get("output") or 0)
            cr = int(rec.get("cache_read") or 0)
            if inp == 0 and out == 0 and cr == 0:
                continue
            if min_turn is not None and turn < min_turn:
                continue
            if max_turn is not None and turn > max_turn:
                continue
            latest_by_turn[turn] = rec
    return [latest_by_turn[t] for t in sorted(latest_by_turn)]


def clear_token_usage() -> None:
    """删除持久化文件；reset / load 时调用。"""
    try:
        _TOKEN_USAGE_PATH.unlink()
    except FileNotFoundError:
        pass
