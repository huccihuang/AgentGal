"""LLM-based memory consolidator."""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from engine.config import character_path
from llm.llm_parser import OpenAICompatibleClient
from log_config.routing import routing_logger
from memory.file_ops import (
    _get_fields_from_file,
    backup_file,
    cleanup_old_backups,
    load_consolidation_state,
    load_growth_for_prompt,
    load_text,
    normalize,
    normalize,
    read_growth_entries,
    read_agent_file,
    safe_write_memory,
    save_consolidation_state,
    split_by_date,
    split_events_raw,
    write_growth_entries,
    write_growth_entries,
)
from memory.vector_store import vector_store

CONSOLIDATION_INTERVAL = int(os.getenv("CONSOLIDATION_INTERVAL", "10"))

_API_KEY = os.getenv("CONSOLIDATION_LLM_API_KEY") or os.getenv("LLM_API_KEY")
_API_URL = (
    os.getenv("CONSOLIDATION_LLM_API_URL")
    or os.getenv("LLM_API_URL")
    or "https://api.deepseek.com/v1"
)
_MODEL_ID = os.getenv("CONSOLIDATION_LLM_MODEL_ID") or os.getenv("LLM_MODEL_ID") or "deepseek-chat"
_TEMPERATURE = float(os.getenv("CONSOLIDATION_TEMPERATURE", "0.0"))
_MAX_TOKENS = int(os.getenv("CONSOLIDATION_MAX_TOKENS", "8192"))

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "consolidation_prompt.txt"
_PLAYER_PROMPT_PATH = (
    Path(__file__).parent.parent / "prompts" / "player_profile_consolidation_prompt.txt"
)

_USER_FIELD_DESCRIPTIONS: dict[str, str] = {
    "基本信息": "最多 5 条基础信息（名字/称呼、身份、核心性格标签等）",
    "观察到的特质": "最多 8 条跨情境的深层理解（角色对玩家的判断）",
    "互动模式": "最多 5 条关系中的行为规律",
    "玩家风格": "最多 5 条玩家在游戏中的行为风格特征",
    "关键选择": "最多 8 条玩家做出的重要选择及其倾向",
    "当前倾向": "最多 5 条玩家当前的行为/情感倾向",
}


def _log(level: str, message: str, **ctx) -> None:
    text = f"[Memory][Consolidator] {message}"
    if ctx:
        text += " " + " ".join(f"{k}={v}" for k, v in ctx.items())
    if level == "error":
        routing_logger.error(text)
    elif level == "warning":
        routing_logger.warning(text)
    else:
        routing_logger.info(text)


def build_fields_definition(agent_name: str) -> str:
    fields = _get_fields_from_file(character_path(agent_name, "user.md")) or [
        "基本信息",
        "观察到的特质",
        "互动模式",
    ]
    return "\n".join(
        f"- 「{field}」：{_USER_FIELD_DESCRIPTIONS.get(field, '')}" for field in fields
    )


@dataclass
class _ConsolidationResult:
    agent_name: str
    days: int = 0
    date_range: str = ""
    original_len: int = 0
    final_len: int = 0
    user_md_before: int = 0
    user_md_after: int = 0
    skipped: bool = False
    skip_reason: str = ""
    errors: list[str] = field(default_factory=list)


class MemoryConsolidator:
    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._client: Optional[OpenAICompatibleClient] = None

    def _get_lock(self, name: str) -> asyncio.Lock:
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]

    async def _call_llm(self, prompt: str) -> dict:
        if not _API_KEY:
            raise ValueError("LLM_API_KEY 或 CONSOLIDATION_LLM_API_KEY 未配置")
        if self._client is None:
            self._client = OpenAICompatibleClient(
                api_url=_API_URL,
                api_key=_API_KEY,
                model=_MODEL_ID,
                temperature=_TEMPERATURE,
                max_tokens=_MAX_TOKENS,
                timeout=120.0,
                max_retries=3,
            )
            await self._client.initialize()

        resp = await self._client.chat(
            messages=[{"role": "user", "content": prompt}],
            enable_thinking=False,
        )
        return {"content": (resp.get("content") or "").strip(), "usage": resp.get("usage") or {}}

    def _resolve_dates(
        self,
        agent_name: str,
        all_dates: list[str],
        last_consolidated: str | None,
    ) -> tuple[list[str], str | None]:
        if not all_dates:
            return [], None
        if not last_consolidated:
            _log("info", "resolve_dates", agent=agent_name, mode="all", days=len(all_dates))
            return all_dates, all_dates[-1]
        if last_consolidated not in all_dates:
            _log("warning", "resolve_dates_missing", agent=agent_name, last_date=last_consolidated)
            return all_dates, all_dates[-1]
        idx = all_dates.index(last_consolidated)
        if idx == len(all_dates) - 1:
            return [last_consolidated], last_consolidated
        return all_dates[idx:], all_dates[-1]

    def _build_consolidation_prompt(
        self,
        agent_name: str,
        sections: OrderedDict[str, str],
        dates: list[str],
    ) -> str:
        return load_text(_PROMPT_PATH).format(
            soul=read_agent_file(agent_name, "soul.md"),
            growth=load_growth_for_prompt(agent_name, default="（尚无）"),
            content="\n\n".join(f"## {date}\n{sections[date]}" for date in dates),
        )

    def _apply_memory_result(
        self,
        agent_name: str,
        sections: OrderedDict[str, str],
        dates: list[str],
        llm_result: str,
        result: _ConsolidationResult,
    ) -> None:
        if len(llm_result.strip()) < 50:
            result.errors.append("LLM返回过短，跳过整理")
            return

        parsed = self._parse_step1_memories(llm_result)
        if not parsed:
            result.errors.append("未能解析第一步:归并整理")
        else:
            for date in dates:
                if date in parsed:
                    sections[date] = parsed[date]
                else:
                    result.errors.append(f"{date} 未在LLM返回中找到")

        updates = self._parse_step2_growth(llm_result)
        if updates:
            _log(
                "info",
                "growth_updates",
                agent=agent_name,
                result=self._apply_growth_updates(agent_name, updates),
            )

    async def consolidate_agent(self, agent_name: str) -> Optional[_ConsolidationResult]:
        lock = self._get_lock(agent_name)
        result = _ConsolidationResult(agent_name=agent_name)
        if lock.locked():
            result.skipped, result.skip_reason = True, "已有整理任务在运行"
            return result

        async with lock:
            path = Path(character_path(agent_name, "memory.md"))
            if not path.exists():
                return None
            original_content = path.read_text(encoding="utf-8")
            if len(original_content.strip()) < 50:
                return None

            sections = split_by_date(normalize(original_content))
            dates_to_consolidate, next_date = self._resolve_dates(
                agent_name,
                list(sections.keys()),
                load_consolidation_state(agent_name),
            )
            if not dates_to_consolidate:
                return None

            result.days = len(dates_to_consolidate)
            result.date_range = f"{dates_to_consolidate[0]}~{dates_to_consolidate[-1]}"
            result.original_len = len(original_content)

            backup_file(path, agent_name, "Memory")
            try:
                llm_result = (
                    await self._call_llm(
                        self._build_consolidation_prompt(
                            agent_name,
                            sections,
                            dates_to_consolidate,
                        )
                    )
                ).get("content", "").strip()
                self._apply_memory_result(
                    agent_name,
                    sections,
                    dates_to_consolidate,
                    llm_result,
                    result,
                )
            except Exception as e:
                result.errors.append(f"整合失败: {e}")
                _log("error", "consolidate_failed", agent=agent_name, op="memory", error=e)

            result.final_len = safe_write_memory(path, sections, agent_name, original_content)
            if result.final_len < 0:
                result.errors.append("并发冲突：检测到中间变更，已放弃写回")
                return result
            if next_date and not result.errors:
                save_consolidation_state(agent_name, next_date)

            result.user_md_before, result.user_md_after = await self._consolidate_player_profile(
                agent_name
            )
            return result

    def _parse_step1_memories(self, llm_result: str) -> OrderedDict[str, str]:
        match = re.search(r"##\s*第一步.*?(?:##\s*第二步|$)", llm_result, re.DOTALL)
        if not match:
            return OrderedDict()

        content = re.sub(r"^##\s*第一步.*\n", "", match.group(0), count=1)
        content = re.sub(r"##\s*第二步.*$", "", content, flags=re.DOTALL)
        sections: OrderedDict[str, str] = OrderedDict()
        for date, event_text in split_events_raw(content.strip()):
            if not date:
                continue
            sections[date] = f"{sections[date]}\n\n{event_text}" if date in sections else event_text
        return sections

    def _parse_step2_growth(self, llm_result: str) -> list[dict]:
        match = re.search(r"<personality_updates>(.*?)</personality_updates>", llm_result, re.DOTALL)
        if not match:
            return []
        updates = []
        for m in re.finditer(r"<update\b([^>]*)\s*(?:/>|>(.*?)</update>)", match.group(1), re.DOTALL):
            attrs = dict(re.findall(r'(\w+)="(.*?)"', m.group(1) or ""))
            up_type = (attrs.get("type") or "").upper()
            up_id = attrs.get("id")
            if up_type and up_id:
                updates.append(
                    {
                        "type": up_type,
                        "id": up_id,
                        "content": m.group(2).strip() if m.group(2) else None,
                    }
                )
        return updates

    def _apply_growth_updates(self, agent_name: str, updates: list[dict]) -> str:
        entries = read_growth_entries(agent_name)
        logs: list[str] = []

        for up in updates:
            up_type = up["type"]
            up_id = up["id"]
            content = up["content"] or ""
            if up_type == "ADD":
                if up_id in entries:
                    logs.append(f"ADD失败:{up_id}已存在")
                else:
                    entries[up_id] = content
                    logs.append(f"ADD {up_id}")
            elif up_type == "UPDATE":
                logs.append(f"UPDATE {up_id}" if up_id in entries else f"UPDATE警告:{up_id}不存在转为ADD")
                entries[up_id] = content
            elif up_type == "DELETE":
                if up_id in entries:
                    del entries[up_id]
                    logs.append(f"DELETE {up_id}")
                else:
                    logs.append(f"DELETE警告:{up_id}不存在")

        write_growth_entries(agent_name, entries)
        return ";".join(logs) if logs else "无更新"

    async def _consolidate_player_profile(self, agent_name: str) -> tuple[int, int]:
        user_path = Path(character_path(agent_name, "user.md"))
        if not user_path.exists():
            return 0, 0
        content = user_path.read_text(encoding="utf-8")
        if len(content.strip()) < 100:
            return 0, 0

        try:
            backup_file(user_path, agent_name, "user")
            prompt = load_text(_PLAYER_PROMPT_PATH).format(
                fields_definition=build_fields_definition(agent_name),
                content=content,
            )
            consolidated = (await self._call_llm(prompt)).get("content", "").strip()
            if len(consolidated) < 20:
                _log("warning", "player_profile_skip", agent=agent_name, reason="too_short")
                return 0, 0

            marker = re.search(r"^.*第二步.*档案.*$", consolidated, re.MULTILINE)
            if marker:
                consolidated = consolidated[marker.end():].lstrip("\n")

            user_path.write_text(consolidated.strip() + "\n", encoding="utf-8")
            return len(content), len(consolidated)
        except Exception as e:
            _log("error", "player_profile_failed", agent=agent_name, error=e)
            return 0, 0

    async def consolidate_player_profile(self, agent_name: str):
        before, after = await self._consolidate_player_profile(agent_name)
        if before > 0:
            _log("info", "player_profile_done", agent=agent_name, before=before, after=after)

    async def consolidate_all(self, agent_names: list[str]):
        t0 = time.monotonic()
        summaries: list[str] = []
        for name in agent_names:
            path = Path(character_path(name, "memory.md"))
            summaries.append(
                f"{name}({len(path.read_text(encoding='utf-8'))}字)" if path.exists() else f"{name}(无文件)"
            )
        _log("info", "start", op="consolidate_all", agents=",".join(summaries))

        raw_results = await asyncio.gather(
            *(self.consolidate_agent(n) for n in agent_names),
            return_exceptions=True,
        )
        for r in raw_results:
            if isinstance(r, Exception):
                _log("error", "agent_exception", op="consolidate_all", error=r)
                continue
            if r is None:
                continue
            if r.skipped:
                _log("info", "agent_skipped", agent=r.agent_name, reason=r.skip_reason)
                continue

            ratio = (1 - r.final_len / r.original_len) * 100 if r.original_len > 0 else 0.0
            _log(
                "info",
                "agent_done",
                agent=r.agent_name,
                days=r.days,
                date_range=r.date_range,
                memory=f"{r.original_len}->{r.final_len}({ratio:+.1f}%)",
                user=(f"{r.user_md_before}->{r.user_md_after}" if r.user_md_before > 0 else "none"),
                errors=(";".join(r.errors) if r.errors else "none"),
            )

        _log("info", "done", op="consolidate_all", elapsed=f"{time.monotonic() - t0:.1f}s")

    async def close(self):
        if self._client:
            await self._client.close()


_cleanup_old_backups = cleanup_old_backups

memory_consolidator = MemoryConsolidator()
