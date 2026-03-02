"""Agent 运行器 - 按需构建 system prompt 并调用 LLM"""

import asyncio
import os
import re
import time
from pathlib import Path

from engine.config import (
    AGENT_RUN_TIMEOUT_SECONDS,
    PROJECT_ROOT,
    character_path,
    get_agent_names,
)
from engine.response_parser import parse_agent_response
from engine.text_utils import clean_response
from llm.llm_parser import OpenAICompatibleClient
from llm.providers import get_llm_config
from log_config.agent_calls import log_agent_call
from log_config.routing import routing_logger
from memory.file_ops import (
    _append_section_file,
    _read_title,
    add_pending_event,
    _update_section_file,
    get_allowed_fields,
    load_growth_for_prompt,
    mark_event_triggered,
    read_agent_file,
    read_file_tail,
)
from memory.vector_store import vector_store


class AgentManager:
    """管理所有角色的 Agent 实例"""

    def __init__(self):
        self.agents: dict[str, Agent] = {}
        self._current_input: str = ""  # 用于传递当前用户输入给 instructions 回调
        self._init_agents()

    def _init_agents(self):
        """初始化所有角色 Agent"""
        for agent_name in get_agent_names():
            self.agents[agent_name] = self._create_agent(agent_name)

    @staticmethod
    def _extract_user_message_from_input(full_input: str) -> str:
        """从拼接的完整输入中提取原始用户消息。

        输入格式:
            最近对话历史:\n\n{history}\n\n---\n\n玩家新消息: {user_input}
        """
        match = re.search(r"玩家新消息:\s*(.+)$", full_input, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return full_input.strip()

    def _create_agent(self, agent_name: str) -> Agent:
        """创建单个 Agent（预创建，复用）"""
        # 加载静态角色设定（soul.md 是只读的）
        soul_content = read_agent_file(agent_name, "soul.md")

        # 定义动态 instructions 函数，每次运行时重新加载记忆文件
        def get_dynamic_instructions(agent: Agent, run_context=None) -> str:
            # 从实例变量获取当前输入，提取原始用户消息用于 RAG
            user_input = self._current_input
            routing_logger.debug(f"[AgentManager] instructions 回调: agent={agent_name}, input_len={len(user_input)}")

            # 同步 RAG 搜索相关记忆
            relevant_memories = self._search_relevant_memories_sync(
                agent_name, user_input
            )

            # 加载 memory.md 最后5行作为 recent_memories
            recent_memories = read_file_tail(
                character_path(agent_name, "memory.md"), lines=5
            )
            if not recent_memories:
                recent_memories = "（尚无记忆）"

            # 加载 status.md
            status_content = read_agent_file(agent_name, "status.md")

            # 加载 user.md
            user_content = read_agent_file(agent_name, "user.md")

            # 加载 growth.md
            growth_content = load_growth_for_prompt(agent_name)

            # 加载并填充 system prompt 模板
            prompt_template = self._load_system_prompt_template(agent_name)
            # 动态获取字段白名单（从文件读取，失败回退到默认值）
            status_fields = "、".join(get_allowed_fields(agent_name, "status"))
            player_fields = "、".join(get_allowed_fields(agent_name, "user"))
            return prompt_template.format(
                agent_name=agent_name,
                soul=soul_content,
                growth=growth_content,
                memory=relevant_memories,
                recent_memories=recent_memories,
                status=status_content if status_content else "（尚无状态记录）",
                user_profile=user_content if user_content else "（尚无玩家认知）",
                status_fields=status_fields,
                player_fields=player_fields,
            )

        return Agent(
            name=agent_name,
            model=get_model(),
            instructions=get_dynamic_instructions,
            markdown=True,
            post_hooks=[log_agent_run],
            # 禁用 Agno 内部历史管理，由应用层通过 jsonl 自行管理
            add_history_to_context=False,
        )

    def _search_relevant_memories_sync(self, agent_name: str, query: str) -> str:
        """同步搜索相关记忆，用于 instructions 函数"""
        try:
            limit_env = int(os.getenv("VECTOR_SEARCH_LIMIT", "5"))
        except ValueError:
            limit_env = 5
        results = vector_store.search(agent_name, query, limit=limit_env, kind="memory")

        if not results:
            return "（无相关记忆）"

        # 格式化召回的记忆
        memories = []
        for r in results:
            content = r["content"].strip()
            if content:
                memories.append(content)

        return "\n\n---\n\n".join(memories) if memories else "（无相关记忆）"


def _build_memory_prefix(agent_name: str, user_input: str) -> str:
    """组装记忆上下文前缀（RAG 召回 + 最近记忆）。"""
    relevant = _search_memories(agent_name, user_input)
    parts = [f"<relevant_memories>\n{relevant}\n</relevant_memories>"]

    if agent_name != "narrator":
        recent = read_file_tail(character_path(agent_name, "memory.md"), lines=5) or "（尚无记忆）"
        parts.append(f"<recent_memories>\n{recent}\n</recent_memories>")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 响应后处理：写回文件
# ---------------------------------------------------------------------------

def _update_memory(agent_name: str, memory_content: str) -> str:
    """追加 memory 内容到 memory.md（带去重）。"""
    if not memory_content or not memory_content.strip():
        return "内容为空，跳过"

    memory_path = character_path(agent_name, "memory.md")
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)
    clean = memory_content.replace("\\n", "\n").strip()

    def _parse_entries(text: str) -> list[str]:
        entries, current = [], []
        for line in text.split("\n"):
            if line.strip().startswith("##") or (line.strip().startswith("-") and "**" in line):
                if current:
                    entries.append("\n".join(current).strip())
                current = [line]
            elif line.strip() or current:
                current.append(line)
        if current:
            entries.append("\n".join(current).strip())
        return entries

    existing = Path(memory_path).read_text(encoding="utf-8") if os.path.exists(memory_path) else ""
    existing_set = set(_parse_entries(existing))
    unique = [e for e in _parse_entries(clean) if e and e not in existing_set]

    if not unique:
        return "所有 entry 已存在，跳过"

    to_append = "\n\n".join(unique)
    if existing.strip():
        with open(memory_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n{to_append}")
    else:
        with open(memory_path, "w", encoding="utf-8") as f:
            f.write(f"# {agent_name} 的长期记忆\n\n{to_append}")

    return f"已追加 {len(unique)} 个新 entry"


def _update_status(agent_name: str, field: str, content: str) -> str:
    """覆盖更新 status.md 的指定字段。"""
    allowed = get_allowed_fields(agent_name, "status")
    if field not in allowed:
        routing_logger.warning(f"[{agent_name}] 不允许的 status 字段: {field}")
        return f"字段 {field} 不在白名单中"
    status_path = character_path(agent_name, "status.md")
    return _update_section_file(status_path, field, content, allowed, _read_title(status_path, "# 我的状态"))


def _update_player(agent_name: str, field: str, content: str) -> str:
    """追加更新 user.md 的指定字段。"""
    allowed = get_allowed_fields(agent_name, "user")
    if field not in allowed:
        routing_logger.warning(f"[{agent_name}] 不允许的 player 字段: {field}")
        return f"字段 {field} 不在白名单中"
    user_path = character_path(agent_name, "user.md")
    return _append_section_file(user_path, field, content, allowed, _read_title(user_path, "# 玩家档案"))


async def _apply_response_updates(agent_name: str, parsed) -> None:
    """将解析后的 XML 更新指令写回对应文件。"""
    results = []

    if parsed.memory:
        try:
            results.append(f"memory: {_update_memory(agent_name, parsed.memory)}")
        except Exception as e:
            routing_logger.error(f"[{agent_name}] 更新 memory 失败: {e}")

    if parsed.status:
        try:
            for field, content in parsed.status.items():
                results.append(f"status[{field}]: {_update_status(agent_name, field, str(content))}")
        except Exception as e:
            routing_logger.error(f"[{agent_name}] 更新 status 失败: {e}")

    if parsed.player:
        try:
            for field, content in parsed.player.items():
                results.append(f"player[{field}]: {_update_player(agent_name, field, str(content))}")
        except Exception as e:
            routing_logger.error(f"[{agent_name}] 更新 player 失败: {e}")

    if parsed.triggered:
        try:
            for event_name in parsed.triggered:
                results.append(f"triggered[{event_name}]: {mark_event_triggered(agent_name, event_name)}")
        except Exception as e:
            routing_logger.error(f"[{agent_name}] 标记触发事件失败: {e}")

    if parsed.add_event:
        try:
            for event_desc in parsed.add_event:
                results.append(f"add_event: {add_pending_event(agent_name, event_desc)}")
        except Exception as e:
            routing_logger.error(f"[{agent_name}] 插入新事件失败: {e}")

    if results:
        routing_logger.info(f"[{agent_name}] 文件更新: {'; '.join(results)}")


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------

async def run_agent(agent_name: str, user_input: str) -> str:
    """运行指定角色的 Agent，返回清理后的响应文本。"""
    start = time.time()

    pure_input = _extract_user_message(user_input)
    memory_prefix = _build_memory_prefix(agent_name, pure_input)
    full_input = f"{memory_prefix}\n\n---\n\n{user_input}" if memory_prefix else user_input

    soul_content = read_agent_file(agent_name, "soul.md")
    system_prompt = _build_system_prompt(agent_name, soul_content)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_input},
    ]

    config = get_llm_config()
    try:
        async with OpenAICompatibleClient(**config) as client:
            response = await asyncio.wait_for(
                client.chat(messages),
                timeout=AGENT_RUN_TIMEOUT_SECONDS,
            )
        routing_logger.info(f"{agent_name} 运行完成，耗时 {time.time() - start:.1f}秒")
        log_agent_call(agent_name, config["model"], messages, response)

        parsed = parse_agent_response(response["content"], agent_name)
        await _apply_response_updates(agent_name, parsed)
        return clean_response(parsed.content)

    except asyncio.TimeoutError:
        routing_logger.error(f"{agent_name} 运行超时（{time.time() - start:.1f}秒），强制终止")
        return f"[{agent_name} 回应超时，请稍后再试]"
