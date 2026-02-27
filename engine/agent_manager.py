"""Agent 管理器 - 初始化 Agent 并处理运行"""

import asyncio
import os
import re
from pathlib import Path

from agno.agent import Agent

from engine.config import (
    AGENT_RUN_TIMEOUT_SECONDS,
    PROJECT_ROOT,
    character_path,
    get_agent_names,
)
from engine.response_parser import parse_agent_response
from engine.text_utils import clean_response
from llm.providers import get_model
from log_config.agent_calls import log_agent_run
from log_config.routing import routing_logger
from memory.file_ops import (
    _append_section_file,
    _read_title,
    _update_section_file,
    get_allowed_fields,
    load_growth_for_prompt,
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
        # 默认走 all：当天 round + 今天之前 memory（由 VectorStore 内部规则控制）
        results = vector_store.search(agent_name, query, limit=limit_env)

        if not results:
            return "（无相关记忆）"

        # 格式化召回的记忆
        memories = []
        for r in results:
            content = r["content"].strip()
            if content:
                memories.append(content)

        return "\n\n---\n\n".join(memories) if memories else "（无相关记忆）"

    def _load_system_prompt_template(self, agent_name: str) -> str:
        """加载 system prompt 模板"""
        # narrator 使用专用模板
        if agent_name == "narrator":
            template_path = PROJECT_ROOT / "prompts" / "narrator_prompt.txt"
        else:
            template_path = PROJECT_ROOT / "prompts" / "character_prompt.txt"

        return Path(template_path).read_text(encoding="utf-8")

    async def run_agent(self, agent_name: str, user_input: str) -> str:
        import time

        start = time.time()

        # 获取预创建的 agent
        agent = self.agents.get(agent_name)
        if not agent:
            routing_logger.error(f"[{agent_name}] Agent 未初始化")
            return f"[{agent_name} 系统错误]"

        # 保存当前输入（提取纯玩家消息），供 instructions 回调使用
        self._current_input = self._extract_user_message_from_input(user_input)

        try:
            response = await asyncio.wait_for(
                agent.arun(user_input),
                timeout=AGENT_RUN_TIMEOUT_SECONDS,
            )
            elapsed = time.time() - start
            routing_logger.info(f"{agent_name} 运行完成，耗时 {elapsed:.1f}秒")

            # 解析响应中的 XML 更新指令
            raw_content = response.content
            parsed = parse_agent_response(raw_content, agent_name)

            # 应用更新到文件
            await self._apply_response_updates(agent_name, parsed)

            # 清理响应内容
            return clean_response(parsed.content)
        except asyncio.TimeoutError:
            elapsed = time.time() - start
            routing_logger.error(f"{agent_name} 运行超时（{elapsed:.1f}秒），强制终止")
            return f"[{agent_name} 回应超时，请稍后再试]"

    async def _apply_response_updates(self, agent_name: str, parsed) -> None:
        """
        应用解析后的更新到对应文件。

        Args:
            agent_name: 角色名称
            parsed: ParsedResponse 对象
        """
        results = []

        # --- memory: 追加到 memory.md（带去重） ---
        if parsed.memory:
            try:
                result = self._update_memory(agent_name, parsed.memory)
                results.append(f"memory: {result}")
            except Exception as e:
                routing_logger.error(f"[{agent_name}] 更新 memory 失败: {e}")
                results.append("memory: 失败")

        # --- status: 覆盖更新到 status.md ---
        if parsed.status:
            try:
                for field, content in parsed.status.items():
                    result = self._update_status(agent_name, field, str(content))
                    results.append(f"status[{field}]: {result}")
            except Exception as e:
                routing_logger.error(f"[{agent_name}] 更新 status 失败: {e}")
                results.append("status: 失败")

        # --- player: 追加更新到 user.md ---
        if parsed.player:
            try:
                for field, content in parsed.player.items():
                    result = self._update_player(agent_name, field, str(content))
                    results.append(f"player[{field}]: {result}")
            except Exception as e:
                routing_logger.error(f"[{agent_name}] 更新 player 失败: {e}")
                results.append("player: 失败")

        if results:
            routing_logger.info(f"[{agent_name}] 文件更新: {'; '.join(results)}")

    def _update_memory(self, agent_name: str, memory_content: str) -> str:
        """追加 memory 内容到 memory.md（带去重）"""
        if not memory_content or not memory_content.strip():
            return "内容为空，跳过"

        memory_path = character_path(agent_name, "memory.md")
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)

        clean = memory_content.replace("\\n", "\n").strip()

        # 解析 entries
        def _parse_entries(text: str) -> list[str]:
            entries = []
            current_entry = []
            for line in text.split("\n"):
                if line.strip().startswith("##") or (
                    line.strip().startswith("-") and "**" in line
                ):
                    if current_entry:
                        entries.append("\n".join(current_entry).strip())
                    current_entry = [line]
                elif line.strip() or current_entry:
                    current_entry.append(line)
            if current_entry:
                entries.append("\n".join(current_entry).strip())
            return entries

        # 读取现有内容
        existing = ""
        if os.path.exists(memory_path):
            with open(memory_path, "r", encoding="utf-8") as f:
                existing = f.read()

        new_entries = _parse_entries(clean)
        existing_entries = _parse_entries(existing)
        existing_set = set(existing_entries)

        # 去重
        unique_entries = [e for e in new_entries if e and e not in existing_set]

        if not unique_entries:
            return "所有 entry 已存在，跳过"

        # 写入
        to_append = "\n\n".join(unique_entries)
        if existing.strip():
            with open(memory_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n{to_append}")
        else:
            with open(memory_path, "w", encoding="utf-8") as f:
                f.write(f"# {agent_name} 的长期记忆\n\n{to_append}")

        msg = f"已追加 {len(unique_entries)} 个新 entry"
        routing_logger.info(f"[{agent_name}] {msg}")
        return msg

    def _update_status(self, agent_name: str, field: str, content: str) -> str:
        """覆盖更新 status.md 的指定字段"""
        allowed = get_allowed_fields(agent_name, "status")
        if field not in allowed:
            routing_logger.warning(
                f"[{agent_name}] 不允许的 status 字段: {field}, "
                f"允许的: {', '.join(allowed)}"
            )
            return f"字段 {field} 不在白名单中"

        status_path = character_path(agent_name, "status.md")
        title = _read_title(status_path, "# 我的状态")

        result = _update_section_file(status_path, field, content, allowed, title)
        return result

    def _update_player(self, agent_name: str, field: str, content: str) -> str:
        """追加更新 user.md 的指定字段"""
        allowed = get_allowed_fields(agent_name, "user")
        if field not in allowed:
            routing_logger.warning(
                f"[{agent_name}] 不允许的 player 字段: {field}, "
                f"允许的: {', '.join(allowed)}"
            )
            return f"字段 {field} 不在白名单中"

        user_path = character_path(agent_name, "user.md")
        title = _read_title(user_path, "# 玩家档案")

        result = _append_section_file(user_path, field, content, allowed, title)
        return result


# 全局实例
agent_manager = AgentManager()
