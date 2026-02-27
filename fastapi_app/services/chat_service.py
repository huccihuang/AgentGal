"""聊天业务服务：对齐 app.py 的 narrator 路由 + 多角色顺序回应 + 命令分发"""

import asyncio
import json
import re
from uuid import uuid4

from engine.agent_manager import agent_manager
from engine.config import (
    HISTORY_LIMIT_DEFAULT,
    HISTORY_LIMIT_NARRATOR,
    get_agent_names,
    get_valid_response_agents,
)
from engine.message_router import message_router
from engine.text_utils import (
    clean_response,
    is_valid_response,
    process_character_response,
)
from game.save_manager import export_save_archive, reset_game
from memory.consolidator import CONSOLIDATION_INTERVAL, memory_consolidator
from memory.vector_store import vector_store
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi_app.repositories.conversation_repository import (
    create_conversation,
    get_conversation_by_id,
)
from fastapi_app.repositories.message_repository import create_message
from fastapi_app.repositories.user_repository import create_user, get_user_by_id
from fastapi_app.schemas.chat import AgentReply, ChatResponse


DEFAULT_USER_ID = "fastapi_user"


def _parse_narrator_response(content: str) -> tuple[list[str], str]:
    valid_agents = get_valid_response_agents()

    targets_pattern = re.compile(
        r"TARGETS\s*:?\s*\[?([^\]\n]*)\]?", re.IGNORECASE
    )
    all_matches = list(targets_pattern.finditer(content))

    if not all_matches:
        return [], content

    targets_match = all_matches[-1]
    targets_str = targets_match.group(1)
    targets = [
        t.strip().lower()
        for t in targets_str.split(",")
        if t.strip() and t.strip().lower() in valid_agents
    ]

    scene_description = content[targets_match.end() :].strip()
    return targets, scene_description


def _build_agent_input(history: str, user_input: str) -> str:
    parts = []
    if history:
        parts.append(f"最近对话历史:\n\n{history}")
        if f"玩家: {user_input}" not in history:
            parts.append(f"玩家新消息: {user_input}")
    else:
        parts.append(f"玩家新消息: {user_input}")
    return "\n\n---\n\n".join(parts)


async def _run_one_agent(
    agent_name: str, targets: list[str], user_input: str
) -> AgentReply:
    try:
        history = message_router.load_recent_history(
            agent_name, limit=HISTORY_LIMIT_DEFAULT
        )
        full_input = _build_agent_input(history, user_input)

        response = await agent_manager.run_agent(agent_name, full_input)
        response = process_character_response(response)

        valid = is_valid_response(response, agent_name)
        if valid:
            await message_router.broadcast_agent_response(agent_name, targets, response)

        return AgentReply(agent=agent_name, content=response, valid=valid)
    except Exception as exc:  # pragma: no cover
        return AgentReply(agent=agent_name, content=f"[错误: {exc}]", valid=False)


def _format_round_content(
    user_input: str,
    narrator_text: str | None,
    replies: list["AgentReply"],
) -> str:
    """将一轮对话格式化为可存入向量数据库的纯文本。"""
    parts = [f"玩家: {user_input}"]
    if narrator_text:
        parts.append(f"旁白: {narrator_text}")
    for reply in replies:
        if reply.valid:
            parts.append(f"{reply.agent}: {reply.content}")
    return "\n".join(parts)


def _extract_game_date(scene_description: str | None) -> str | None:
    if not scene_description:
        return None
    import re
    m = re.search(r"\*\*时间\*\*：\s*(\d{1,2}月\d{1,2}日)", scene_description)
    return m.group(1) if m else None


def _build_visible_to(targets: list[str]) -> str:
    visible = list(targets)
    if "narrator" not in visible:
        visible.append("narrator")
    return json.dumps(visible, ensure_ascii=False)


async def _ensure_user_and_conversation(
    db: AsyncSession,
    user_id: str,
    conversation_id: str,
) -> int:
    dialogue_count = 0
    user = await get_user_by_id(db, user_id)
    if user is None:
        await create_user(db=db, username=user_id, user_id=user_id)

    conversation = await get_conversation_by_id(db, conversation_id)
    if conversation is None:
        conversation = await create_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            name="默认会话",
            status="active",
            is_deleted=False,
            dialogue_count=0,
        )
    if conversation.dialogue_count is not None:
        dialogue_count = conversation.dialogue_count
    return dialogue_count


async def _apply_command(
    command: str,
    db: AsyncSession,
    conversation_id: str,
) -> ChatResponse | None:
    if command == "/save":
        save_path = await export_save_archive()
        if save_path:
            text = (
                f"✅ 存档已导出: `{save_path}`\n\n包含所有角色的记忆、对话历史和状态。"
            )
        else:
            text = "❌ 存档导出失败，请检查日志。"
        return ChatResponse(narrator=text, targets=[], replies=[])

    if command == "/reset":
        opening = await reset_game()
        conversation = await get_conversation_by_id(db, conversation_id)
        if conversation is not None:
            conversation.dialogue_count = 0
            await db.commit()
        return ChatResponse(
            narrator=f"✅ 游戏已重置，开始新故事...\n\n{opening}",
            targets=[],
            replies=[],
        )
    return None


async def _increase_dialogue_count(
    db: AsyncSession,
    conversation_id: str,
    current_count: int,
) -> int:
    next_count = current_count + 1
    conversation = await get_conversation_by_id(db, conversation_id)
    if conversation is not None:
        conversation.dialogue_count = next_count
        await db.commit()
    return next_count


def _trigger_consolidation_if_needed(message_counter: int) -> None:
    if message_counter % CONSOLIDATION_INTERVAL != 0:
        return
    all_agents = get_agent_names()
    asyncio.create_task(memory_consolidator.consolidate_all(all_agents))


async def run_chat(
    user_input: str,
    db: AsyncSession,
    user_id: str | None = None,
    conversation_id: str | None = None,
) -> ChatResponse:
    resolved_user_id = (user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
    resolved_conversation_id = (
        conversation_id or f"{resolved_user_id}:default"
    ).strip() or f"{resolved_user_id}:default"
    current_count = await _ensure_user_and_conversation(
        db, resolved_user_id, resolved_conversation_id
    )

    command_result = await _apply_command(
        user_input.strip(),
        db=db,
        conversation_id=resolved_conversation_id,
    )
    if command_result is not None:
        return command_result

    message_counter = await _increase_dialogue_count(
        db=db,
        conversation_id=resolved_conversation_id,
        current_count=current_count,
    )

    narrator_history = message_router.load_recent_history(
        "narrator", limit=HISTORY_LIMIT_NARRATOR
    )
    narrator_input = (
        f"最近对话历史:\n\n{narrator_history}\n\n---\n\n玩家新消息: {user_input}"
        if narrator_history
        else user_input
    )

    narrator_response = await agent_manager.run_agent("narrator", narrator_input)
    narrator_content = clean_response(narrator_response)
    targets, scene_description = _parse_narrator_response(narrator_content)
    narrator_valid = is_valid_response(narrator_content, "narrator")

    await message_router.broadcast_player_message(targets, user_input)
    await create_message(
        db=db,
        message_id=uuid4().hex,
        conversation_id=resolved_conversation_id,
        user_id=resolved_user_id,
        role="player",
        content=user_input,
        visible_to=_build_visible_to(targets),
    )

    narrator_text: str | None = None
    if scene_description and narrator_valid:
        await message_router.broadcast_agent_response(
            "narrator", targets, scene_description
        )
        narrator_text = scene_description
        await create_message(
            db=db,
            message_id=uuid4().hex,
            conversation_id=resolved_conversation_id,
            user_id=resolved_user_id,
            role="narrator",
            content=scene_description,
            visible_to=_build_visible_to(targets),
        )

    if not targets:
        _trigger_consolidation_if_needed(message_counter)
        return ChatResponse(narrator=narrator_text, targets=[], replies=[])

    replies: list[AgentReply] = []
    for agent_name in targets:
        replies.append(await _run_one_agent(agent_name, targets, user_input))
    for reply in replies:
        if not reply.valid:
            continue
        await create_message(
            db=db,
            message_id=uuid4().hex,
            conversation_id=resolved_conversation_id,
            user_id=resolved_user_id,
            role=reply.agent,
            content=reply.content,
            visible_to=_build_visible_to(targets),
        )

    if targets:
        # narrator 也在可见列表中（上帝视角）
        visible_to = targets if "narrator" in targets else [*targets, "narrator"]
        round_id = f"{resolved_conversation_id}_{message_counter}"
        previous_game_date = vector_store._conv_game_date.get(resolved_conversation_id)
        game_date = _extract_game_date(narrator_text)
        vector_store.add_round(
            visible_to,
            round_id,
            _format_round_content(user_input, narrator_text, replies),
            game_date=game_date,
        )
        if previous_game_date and game_date and game_date != previous_game_date:
            for agent in visible_to:
                asyncio.create_task(vector_store.add_memory(agent, previous_game_date))

    _trigger_consolidation_if_needed(message_counter)
    return ChatResponse(narrator=narrator_text, targets=targets, replies=replies)
