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
from engine.config import character_path
from memory.file_ops import (
    _get_fields_from_file,
    backup_file,
    cleanup_old_backups,
    load_consolidation_state,
    load_growth_for_prompt,
    load_last_memory_size,
    load_text,
    normalize,
    normalize,
    read_growth_entries,
    read_agent_file,
    safe_write_memory,
    save_consolidation_state,
    save_memory_size,
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

_PROMPT_STEP1_PATH = Path(__file__).parent.parent / "prompts" / "consolidation_prompt_step1.txt"
_PROMPT_STEP2_PATH = Path(__file__).parent.parent / "prompts" / "consolidation_prompt_step2.txt"
_PROMPT_STEP3_PATH = Path(__file__).parent.parent / "prompts" / "consolidation_prompt_step3.txt"
_PLAYER_PROMPT_PATH = (
    Path(__file__).parent.parent / "prompts" / "player_profile_consolidation_prompt.txt"
)

# 文件大小变化阈值（字节）：当文件比上次长了 100 字以上，才触发整理
_CONSOLIDATION_SIZE_THRESHOLD = 100

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
    growth_log: str = ""
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
        self, agent_name: str, sections: OrderedDict[str, str], dates: list[str]
    ) -> str:
        """构建记忆整合的 LLM prompt。"""
        soul_content = read_agent_file(agent_name, "soul.md")
        growth_content = load_growth_for_prompt(agent_name, default="（尚无）")

        parts = [f"## {date}\n{sections[date]}" for date in dates]
        combined_text = "\n\n".join(parts)

        template = load_text(_PROMPT_PATH)
        return template.format(
            soul=soul_content,
            growth=growth_content,
            content=combined_text,
        )

    def _apply_memory_result(
        self,
        agent_name: str,
        sections: OrderedDict[str, str],
        dates: list[str],
        llm_result: str,
        result: "_ConsolidationResult",
    ):
        """解析 LLM 结果并应用更新到 sections。"""
        if len(llm_result.strip()) < 50:
            result.errors.append("LLM返回过短，跳过整理")
            return

        # ===== 第一步：更新 memory.md =====
        step1_sections = self._parse_step1_memories(llm_result)
        if step1_sections:
            for date in dates:
                if date in parsed:
                    sections[date] = parsed[date]
                else:
                    result.errors.append(f"{date} 未在LLM返回中找到")
        else:
            result.errors.append("未能解析第一步:归并整理")

        # ===== 第二步：更新 growth.md =====
        step2_updates = self._parse_step2_growth(llm_result)
        if step2_updates:
            growth_log = self._apply_growth_updates(agent_name, step2_updates)
            routing_logger.info(f"[整理器] {agent_name} growth.md: {growth_log}")
        else:
            routing_logger.info(f"[整理器] {agent_name} 无人格沉淀更新")

    async def consolidate_agent(
        self, agent_name: str
    ) -> Optional["_ConsolidationResult"]:
        """整理单个 agent 的记忆，返回结果摘要（供 consolidate_all 汇总日志）"""
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

            # 3. 备份
            backup_file(path, agent_name, "Memory")

            # 4. 构建 prompt 并调用 LLM
            prompt = self._build_consolidation_prompt(
                agent_name, sections, dates_to_consolidate
            )
            try:
                llm_response = await self._call_llm(prompt)
                llm_result = (llm_response.get("content") or "").strip()
                self._apply_memory_result(
                    agent_name, sections, dates_to_consolidate, llm_result, result
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

            # 8. 顺带整理 user.md
            user_before, user_after = await self._consolidate_player_profile(agent_name)
            result.user_md_before = user_before
            result.user_md_after = user_after

            return result

    def _parse_step1_memories(self, llm_result: str) -> OrderedDict[str, str]:
        """
        从 LLM 输出中提取第一步：归并整理后的日记内容。

        新格式（扁平列表，日期在时间字段中）：
        ## 第一步：归并整理
        - **时间**：4月3日 上午
        - **地点**：教室
        - **在场**：莉莉丝、李小明
        - **内容**：事件描述...

        Returns:
            OrderedDict[日期, 该日期的内容]
        """
        # 提取 "## 第一步" 到 "## 第二步" 之间的内容
        step1_pattern = r"##\s*第一步.*?(?:##\s*第二步|$)"
        step1_match = re.search(step1_pattern, llm_result, re.DOTALL)

        if not step1_match:
            return OrderedDict()

        step1_content = step1_match.group(0)
        # 移除第一步标题本身
        step1_content = re.sub(r"^##\s*第一步.*\n", "", step1_content, count=1)
        # 移除第二步标记（如果有）
        step1_content = re.sub(r"##\s*第二步.*$", "", step1_content, flags=re.DOTALL)

        # 解析新格式：从 - **时间**：字段中提取日期
        sections: OrderedDict[str, str] = OrderedDict()
        for date, event_text in split_events_raw(step1_content.strip()):
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
            if path.exists():
                length = len(path.read_text(encoding="utf-8"))
                summaries.append(f"{name}({length}字)")
            else:
                summaries.append(f"{name}(无文件)")

        routing_logger.info(f"[整理器] 开始记忆整理: {', '.join(summaries)}")

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

            if r.original_len > 0:
                ratio = (1 - r.final_len / r.original_len) * 100
                mem_part = f"{r.original_len}→{r.final_len}字({ratio:+.1f}%)"
            else:
                mem_part = "无变化"

            user_part = ""
            if r.user_md_before > 0:
                user_part = f" | user.md {r.user_md_before}→{r.user_md_after}"

            err_part = ""
            if r.errors:
                err_part = f" | 错误: {', '.join(r.errors)}"

            routing_logger.info(
                f"[整理器] {r.agent_name} 完成: "
                f"{r.days}天({r.date_range}) {mem_part}{user_part}{err_part}"
            )

        elapsed = time.monotonic() - t0
        routing_logger.info(f"[整理器] 全部完成 (耗时 {elapsed:.1f}s)")

    async def close(self):
        if self._client:
            await self._client.close()


_cleanup_old_backups = cleanup_old_backups

memory_consolidator = MemoryConsolidator()
