"""FastAPI 入口"""

# load_dotenv 必须在所有内部模块 import 之前执行，
# 否则 llm/embedding.py 等模块级常量会在 .env 加载前就固化为空值
from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import re
import traceback
from functools import lru_cache
from pathlib import Path

from typing import Literal

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.factory import initialize_conversation_agents, reload_conversation_agent
from consolidation.flow import memory_consolidation_flow
from engine.character import narrator, reset_entities
from engine.conversation_flow import (
    bootstrap_new_characters,
    generate_choices,
    run_agent_in_scene,
)
from memory.parser import (
    EpisodeMemory,
    Understanding,
    extract_status_field,
    read_memory_jsonl,
    read_understandings,
)
from storage.save_manager import (
    delete_save_leaf,
    delete_save_game,
    export_save_archive_with_detail,
    get_current_save_context,
    has_existing_save,
    import_save_archive,
    list_save_archives,
    list_save_worldlines,
    load_story_file,
    reset_game,
)
from log_config.routing import routing_logger
from log_config.logfire import setup_logfire
from shared.config import CHARACTERS_DIR, get_agent_names
from shared.narrator_output import extract_narrator_output
from shared.text_utils import get_display_name
from storage.agent_files import read_agent_file
from storage.history import (
    extract_game_date_anchors,
    load_conversation_history,
    load_history_before,
    search_history,
)
from storage.message_router import message_router

app = FastAPI(title="AgentGal")

STATIC_DIR = Path(__file__).parent / "static"
_LAST_CHOICES_FILE = CHARACTERS_DIR / "last_choices.json"
_pending_state_update_task: asyncio.Task[None] | None = None
_pending_state_update_requested = False
_pending_choices_task: asyncio.Task[list[str]] | None = None
_choices_generation_token = 0
_RECENT_HISTORY_LIMIT = 12
_MEMORY_GRAPH_LABEL_LIMIT = 42
_MEMORY_GRAPH_DETAIL_LIMIT = 260
_MEMORY_GRAPH_RAW_LIMIT = 12000


# =============================================================================
# 工具函数
# =============================================================================


def _save_last_choices(choices: list[str]) -> None:
    _LAST_CHOICES_FILE.write_text(json.dumps(choices, ensure_ascii=False), encoding="utf-8")


def _load_last_choices() -> list[str]:
    try:
        return json.loads(_LAST_CHOICES_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _clear_last_choices() -> None:
    _LAST_CHOICES_FILE.unlink(missing_ok=True)


def _consume_cancelled_choices_task(task: asyncio.Task[list[str]]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        routing_logger.info("[choices] 选项生成任务已取消")
    except Exception as exc:
        routing_logger.warning("[choices] 选项生成任务结束异常: %s", exc)


def _invalidate_pending_choices(*, clear_saved: bool = False) -> int:
    """让旧选项生成失效；返回当前请求应使用的 generation token。"""
    global _pending_choices_task, _choices_generation_token
    _choices_generation_token += 1
    task = _pending_choices_task
    if task is not None and not task.done():
        task.cancel()
        task.add_done_callback(_consume_cancelled_choices_task)
    _pending_choices_task = None
    if clear_saved:
        _clear_last_choices()
    return _choices_generation_token


@lru_cache(maxsize=64)
def _get_agent_display_name(agent_name: str) -> str:
    if agent_name == "narrator":
        return "旁白"
    if agent_name == "player":
        return "你"

    soul_content = read_agent_file(agent_name, "soul.md")
    return get_display_name(agent_name, soul_content) if soul_content else agent_name


def _format_history_message(msg: dict) -> dict | None:
    role = msg.get("role", "")
    payload = extract_narrator_output(msg)
    content = (
        str(payload.get("scene_description") or "").strip()
        if payload
        else str(msg.get("content") or "").strip()
    )
    if not content and not payload:
        return None
    formatted = {
        "role": role,
        "author": _get_agent_display_name(role) if role else "",
        "content": content,
        "turn": int(msg.get("turn") or 0),
    }
    if payload:
        formatted["payload"] = payload
    return formatted


def _clip_text(value: str, limit: int, *, normalize_whitespace: bool = True) -> str:
    text = " ".join((value or "").split()) if normalize_whitespace else (value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}…"


def _current_scene_status() -> dict[str, object]:
    try:
        status = read_agent_file("narrator", "status.md")
    except Exception:
        status = ""
    return {
        "time": extract_status_field(status, "当前时间").strip(),
        "scene": extract_status_field(status, "场景").strip(),
        "focus": extract_status_field(status, "叙事焦点").strip(),
    }


def _format_raw_dialogue_preview(raw_dialogue: str, limit: int) -> str:
    text = (raw_dialogue or "").strip()
    if not text:
        return ""
    text = text[: limit * 3]

    entries: list[str] = []
    for segment in re.split(r"\s*(?=\[turn=\d+\]\s*)", text):
        cleaned = re.sub(r"^\[turn=\d+\]\s*", "", segment.strip())
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            continue

        speaker, sep, body = cleaned.partition(":")
        if sep and speaker and len(speaker) <= 18:
            speaker_name = speaker.strip()
            if speaker_name.lower() in {"旁白", "narrator"}:
                continue
            entries.append(f"{speaker_name}：{body.strip()}")
        else:
            entries.append(cleaned)

    return _clip_text("\n\n".join(entries), limit, normalize_whitespace=False)



def _memory_graph_agents() -> list[dict]:
    agents: list[dict] = []
    for agent_name in get_agent_names(include_narrator=False):
        episodes = read_memory_jsonl(agent_name)
        understandings = read_understandings(agent_name)
        edge_count = sum(len(u.linked_episodes) for u in understandings.values())
        agents.append(
            {
                "name": agent_name,
                "display_name": _get_agent_display_name(agent_name),
                "episode_count": len(episodes),
                "understanding_count": len(understandings),
                "edge_count": edge_count,
            }
        )
    return agents


def _episode_node(agent_name: str, episode: EpisodeMemory, index: int) -> dict:
    key = episode.id or f"row-{index}"
    label = episode.title or episode.content or key
    full_label = f"{episode.date} · {label}" if episode.date else label
    return {
        "id": f"episode:{key}",
        "label": _clip_text(full_label, _MEMORY_GRAPH_LABEL_LIMIT),
        "group": "episode",
        "value": max(2, episode.importance),
        "meta": {
            "id": key,
            "agent": agent_name,
            "type": "episode",
            "type_label": "Episode",
            "title": episode.title or "未命名记忆",
            "date": episode.date,
            "time": episode.time,
            "location": episode.location,
            "participants": episode.participants,
            "keywords": episode.keywords,
            "importance": episode.importance,
            "content": episode.content,
            "content_preview": _clip_text(episode.content, _MEMORY_GRAPH_DETAIL_LIMIT),
            "raw_dialogue_preview": _format_raw_dialogue_preview(
                episode.raw_dialogue, _MEMORY_GRAPH_RAW_LIMIT
            ),
        },
    }


def _understanding_node(agent_name: str, understanding: Understanding, index: int) -> dict:
    key = understanding.id or f"row-{index}"
    label = understanding.subject or understanding.content or key
    return {
        "id": f"understanding:{key}",
        "label": _clip_text(label, _MEMORY_GRAPH_LABEL_LIMIT),
        "group": "understanding",
        "value": max(3, len(understanding.linked_episodes)),
        "meta": {
            "id": key,
            "agent": agent_name,
            "type": "understanding",
            "type_label": "Understanding",
            "title": understanding.subject or "未命名理解",
            "keywords": understanding.keywords,
            "linked_episodes": understanding.linked_episodes,
            "content": understanding.content,
            "content_preview": _clip_text(understanding.content, _MEMORY_GRAPH_DETAIL_LIMIT),
            "history": [entry.model_dump() for entry in understanding.history],
        },
    }


def _missing_episode_node(agent_name: str, episode_id: str) -> dict:
    return {
        "id": f"episode:{episode_id}",
        "label": f"缺失 · {_clip_text(episode_id, 10)}",
        "group": "missing_episode",
        "value": 1,
        "meta": {
            "id": episode_id,
            "agent": agent_name,
            "type": "missing_episode",
            "type_label": "Missing Episode",
            "title": "缺失 Episode",
            "keywords": [],
            "content": f"understanding.jsonl 引用了这个 episode id，但当前 memory.jsonl 中没有对应记录：{episode_id}",
            "content_preview": "",
        },
    }


def _build_memory_graph(agent_name: str, display_name: str) -> dict:
    episodes = read_memory_jsonl(agent_name)
    understandings = list(read_understandings(agent_name).values())

    nodes: list[dict] = []
    edges: list[dict] = []
    episode_ids: set[str] = set()
    missing_ids: set[str] = set()

    for i, ep in enumerate(episodes):
        node = _episode_node(agent_name, ep, i)
        nodes.append(node)
        episode_ids.add(node["id"])

    for i, u in enumerate(understandings):
        u_node = _understanding_node(agent_name, u, i)
        nodes.append(u_node)
        for j, ep_id in enumerate(u.linked_episodes):
            if not ep_id:
                continue
            ep_node_id = f"episode:{ep_id}"
            if ep_node_id not in episode_ids and ep_node_id not in missing_ids:
                nodes.append(_missing_episode_node(agent_name, ep_id))
                missing_ids.add(ep_node_id)
            edges.append({
                "id": f"{ep_node_id}->{u_node['id']}:{j}",
                "from": ep_node_id,
                "to": u_node["id"],
                "meta": {
                    "type": "edge",
                    "episode_id": ep_id,
                    "understanding_id": u_node["meta"]["id"],
                },
            })

    return {
        "selected_agent": agent_name,
        "selected_display_name": display_name,
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "episode_count": len(episodes),
            "understanding_count": len(understandings),
            "edge_count": len(edges),
            "missing_episode_count": len(missing_ids),
        },
    }


def _sse_event(event_type: str, data: dict) -> str:
    """格式化单条 SSE 事件。"""
    payload = json.dumps({"type": event_type, **data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


async def _settle_pending_state_update(*, cancel: bool = False) -> None:
    """结清后台 state_updater 任务。cancel=True 时先取消（用于重置/读档）。"""
    global _pending_state_update_task, _pending_state_update_requested
    task = _pending_state_update_task
    if task is None:
        return
    if cancel and not task.done():
        _pending_state_update_requested = False
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        routing_logger.info("[state_updater] 后台任务已取消")
    finally:
        if _pending_state_update_task is task:
            if not task.done():
                # _settle 自身被外部取消（如 wait_for 超时），而非 cancel=True 主动取消：
                # 内层 _run_state_update_loop 仍在运行，必须显式取消，否则成为孤儿任务。
                task.cancel()
            _pending_state_update_task = None


async def _run_state_update_loop() -> None:
    """串行执行 state_updater；运行期间的新请求合并为最后一次补跑。"""
    global _pending_state_update_task, _pending_state_update_requested
    try:
        while True:
            _pending_state_update_requested = False
            await narrator.update_state()
            if not _pending_state_update_requested:
                break
    finally:
        # 只由本 task 自身清理全局引用：若任务因超时成为孤儿后另一个新任务已注册，
        # 此处若无条件赋 None 会覆盖新任务的引用，导致 _settle 下次直接 return 跳过等待。
        if _pending_state_update_task is asyncio.current_task():
            _pending_state_update_task = None
            _pending_state_update_requested = False


def _start_state_update() -> None:
    global _pending_state_update_task, _pending_state_update_requested
    if _pending_state_update_task is not None and not _pending_state_update_task.done():
        _pending_state_update_requested = True
        return
    _pending_state_update_requested = False
    _pending_state_update_task = asyncio.create_task(_run_state_update_loop())


# =============================================================================
# 启动初始化
# =============================================================================


@app.on_event("startup")
async def startup() -> None:
    setup_logfire()
    initialize_conversation_agents()


# =============================================================================
# 静态文件 & 页面
# =============================================================================

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# =============================================================================
# /api/init
# =============================================================================


@app.get("/api/init")
async def api_init() -> JSONResponse:
    """返回初始状态：是否有存档、最近历史、最近选项。"""
    has_save = has_existing_save()
    recent: list[dict] = []
    last_choices: list[str] = []

    if has_save:
        raw = load_conversation_history(limit=_RECENT_HISTORY_LIMIT)
        for msg in raw:
            formatted = _format_history_message(msg)
            if formatted:
                recent.append(formatted)
        last_choices = _load_last_choices()

    return JSONResponse(
        {
            "has_save": has_save,
            "recent": recent,
            "last_choices": last_choices,
            "scene_status": _current_scene_status(),
            "character_count": len(get_agent_names(include_narrator=False)),
        }
    )


# =============================================================================
# /api/history
# =============================================================================


@app.get("/api/history")
async def api_history(before_turn: int, limit: int = 30) -> JSONResponse:
    limit = max(1, min(limit, 200))
    raw = load_history_before(before_turn=before_turn, limit=limit)
    messages = [m for m in (_format_history_message(item) for item in raw) if m]
    return JSONResponse({"messages": messages})


@app.get("/api/history/dates")
async def api_history_dates() -> JSONResponse:
    return JSONResponse({"anchors": extract_game_date_anchors()})


@app.get("/api/history/search")
async def api_history_search(q: str, limit: int = 50) -> JSONResponse:
    limit = max(1, min(limit, 200))
    raw = search_history(q, limit=limit)
    messages = [m for m in (_format_history_message(item) for item in raw) if m]
    return JSONResponse({"messages": messages})


# =============================================================================
# /api/stories
# =============================================================================


@app.get("/api/stories")
async def api_stories() -> JSONResponse:
    """列出可选故事模板。"""
    stories = [
        {
            "id": "school",
            "label": "私立城川中学",
            "title": "私立城川中学",
            "tagline": "青春校园",
            "summary": "夏季午后的教室、走廊与操场里，关系会在一次次对话和试探中慢慢偏移。",
        },
        {
            "id": "modern",
            "label": "不期而遇",
            "title": "不期而遇",
            "tagline": "现代都市",
            "summary": "在城市日常与偶然重逢之间，让暧昧、克制和误差一点点累积成新的局面。",
        },
    ]
    return JSONResponse({"stories": stories})


# =============================================================================
# /api/memory-graph
# =============================================================================


@app.get("/api/memory-graph")
async def api_memory_graph(agent: str | None = None) -> JSONResponse:
    """返回指定角色的 understanding ↔ episode 可视化数据。"""
    agents = _memory_graph_agents()
    agent_names = {item["name"] for item in agents}
    if not agents:
        return JSONResponse(
            {
                "agents": [],
                "selected_agent": "",
                "selected_display_name": "",
                "nodes": [],
                "edges": [],
                "stats": {
                    "episode_count": 0,
                    "understanding_count": 0,
                    "edge_count": 0,
                    "missing_episode_count": 0,
                },
            }
        )

    selected_agent = agent or agents[0]["name"]
    if selected_agent not in agent_names:
        return JSONResponse({"detail": "角色不存在。"}, status_code=404)

    selected_display_name = next(item["display_name"] for item in agents if item["name"] == selected_agent)
    graph = _build_memory_graph(selected_agent, selected_display_name)
    return JSONResponse({"agents": agents, **graph})


# =============================================================================
# /api/new_game
# =============================================================================


class NewGameRequest(BaseModel):
    story_id: str = "school"


@app.post("/api/new_game")
async def api_new_game(req: NewGameRequest) -> JSONResponse:
    """重置并开始新游戏，返回开场内容。"""
    if memory_consolidation_flow.is_running:
        return JSONResponse(
            {"ok": False, "detail": "记忆整理进行中，请稍后再开始新故事。"},
            status_code=409,
        )
    _invalidate_pending_choices(clear_saved=True)
    await _settle_pending_state_update(cancel=True)
    intro_text, opening_text = await reset_game(req.story_id)
    reset_entities()
    for name in get_agent_names(include_narrator=True):
        reload_conversation_agent(name)

    opening_choices_text = load_story_file(req.story_id, "opening_choices.txt")
    choices = [line.strip() for line in opening_choices_text.splitlines() if line.strip()]
    if choices:
        _save_last_choices(choices)

    return JSONResponse(
        {
            "intro": intro_text,
            "opening": opening_text,
            "choices": choices,
            "scene_status": _current_scene_status(),
            "character_count": len(get_agent_names(include_narrator=False)),
        }
    )


# =============================================================================
# /api/chat  (SSE 流式)
# =============================================================================


class ChatRequest(BaseModel):
    message: str
    mode: Literal["participate", "observe"] = "participate"


async def _chat_stream(user_input: str, mode: Literal["participate", "observe"] = "participate"):
    """核心游戏循环，通过 SSE 逐步推送结果。"""
    global _pending_choices_task
    observation_mode = mode == "observe"
    choices_token = _invalidate_pending_choices(clear_saved=True)
    # 0 = 哨兵：本轮还没有 narrator 成功发言，不触发 consolidation
    current_turn = 0

    async def _settle_state_update_with_timeout() -> None:
        """等待 state_updater 完成；卡住时不阻塞 done 事件。"""
        try:
            await asyncio.wait_for(_settle_pending_state_update(), timeout=60.0)
        except asyncio.TimeoutError:
            routing_logger.warning("state_updater 超时（>60s），done 事件提前发出")

    # 1. narrator 路由
    narrator_output, is_narrator_valid = await narrator.route(
        user_input,
        observation_mode=observation_mode,
    )
    targets = narrator_output.targets if narrator_output is not None else []
    new_character_specs = narrator_output.new_characters if narrator_output is not None else []

    # 1.5 处理 narrator 请求的新角色（孵化成功后加入 targets）
    targets, created_new_characters = await bootstrap_new_characters(
        new_character_specs, targets
    )
    if narrator_output is not None:
        narrator_output = narrator_output.model_copy(update={"targets": targets})
    is_narrator_valid = is_narrator_valid and narrator_output is not None and bool(targets)
    if created_new_characters:
        for character in created_new_characters:
            yield _sse_event(
                "system",
                {
                    "title": "角色已创建",
                    "name": character.display_name,
                    "identity": character.identity,
                    "character_id": character.character_id,
                },
            )

    # 2. 广播玩家消息与旁白
    # 旁白失败时不把玩家消息写进 raw，避免下一轮上下文里残留没人回应的玩家话语。
    narrator_dump = narrator_output.model_dump() if narrator_output is not None else None
    if is_narrator_valid:
        if not observation_mode:
            await message_router.broadcast_player_message(targets, user_input)
        current_turn = await message_router.broadcast_narrator_output(targets, narrator_dump)
    if narrator_dump is not None:
        yield _sse_event(
            "narrator",
            {
                "content": narrator_output.scene_description,
                "author": "旁白",
                "payload": narrator_dump,
            },
        )

    if not targets:
        routing_logger.info("[导演] 无角色需要回应")
        if narrator_output is not None:
            _start_state_update()
            await _settle_state_update_with_timeout()
        yield _sse_event("done", {"consolidating": memory_consolidation_flow.is_running})
        return

    # 4. 顺序处理角色，每完成一个立即推送
    agent_responses: list[tuple[str, str]] = []
    for agent_name in targets:
        try:
            response = await run_agent_in_scene(
                agent_name,
                targets,
                user_input,
                observation_mode=observation_mode,
                narrator_output=narrator_output,
            )
            if response:
                agent_responses.append((agent_name, response))
                yield _sse_event(
                    "agent",
                    {"content": response, "author": _get_agent_display_name(agent_name)},
                )
        except Exception as e:
            routing_logger.error(f"Agent {agent_name} 运行失败: {e}")
            yield _sse_event(
                "agent",
                {"content": f"（{_get_agent_display_name(agent_name)}暂时无法回应，请稍后再试）", "author": _get_agent_display_name(agent_name)},
            )

    choices_task: asyncio.Task[list[str]] | None = None
    if agent_responses and not observation_mode and choices_token == _choices_generation_token:
        choices_task = asyncio.create_task(generate_choices(narrator_output, agent_responses))
        _pending_choices_task = choices_task

    _start_state_update()
    if current_turn > 0:
        # 返回的 task 故意丢弃：MemoryConsolidationFlow._scheduled_task 已是强引用，
        # 任务不会被 GC；新 turn 在已运行任务内 coalesce 成 _pending_turn 补跑。
        memory_consolidation_flow.schedule_detect_and_consolidate(current_turn)
    yield _sse_event("response_done", {"consolidating": memory_consolidation_flow.is_running})

    # 5. 选项是辅助建议：与后台维护并行生成，且新一轮输入会让旧结果失效。
    if choices_task is not None:
        try:
            choices = await choices_task
        except asyncio.CancelledError:
            return
        finally:
            if _pending_choices_task is choices_task:
                _pending_choices_task = None

        if choices and choices_token == _choices_generation_token:
            _save_last_choices(choices)
            yield _sse_event("choices", {"choices": choices})

    # choices 已经 await 完了，这里等 state_updater 写完叙事焦点/最近世界事件，
    # 前端 fetchCharacters 才能读到最新的 narrator/status.md。
    await _settle_state_update_with_timeout()
    yield _sse_event("done", {"consolidating": memory_consolidation_flow.is_running})


@app.post("/api/chat")
async def api_chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _chat_stream(req.message, req.mode),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# =============================================================================
# /api/status
# =============================================================================


@app.get("/api/status")
async def api_status() -> JSONResponse:
    """轻量状态查询。前端在收到 done 后若 consolidating=true 会轮询此接口。"""
    running = memory_consolidation_flow.is_running
    payload: dict = {"consolidating": running, "scene_status": _current_scene_status()}
    if not running and memory_consolidation_flow.last_created_episodes:
        episodes, memory_consolidation_flow.last_created_episodes = (
            memory_consolidation_flow.last_created_episodes,
            [],
        )
        payload["new_episodes"] = [
            {
                "display_name": _get_agent_display_name(ep.agent_name),
                "title": ep.title,
            }
            for ep in episodes
        ]
    return JSONResponse(payload)


# =============================================================================
# /api/saves
# =============================================================================


@app.get("/api/saves")
async def api_list_saves() -> JSONResponse:
    """列出所有存档，并返回按故事世界观拼出的临时世界线树。"""
    saves = list_save_archives()
    return JSONResponse(
        {
            "saves": saves,
            "worlds": list_save_worldlines(saves),
            **get_current_save_context(),
        }
    )


# =============================================================================
# /api/save
# =============================================================================


class SaveRequest(BaseModel):
    filename: str | None = None


@app.post("/api/save")
async def api_save(req: SaveRequest | None = None) -> JSONResponse:
    """导出新的不可变世界线节点。"""
    if memory_consolidation_flow.is_running:
        return JSONResponse(
            {"ok": False, "detail": "记忆整理进行中，请稍后再存档。"},
            status_code=409,
        )
    try:
        await _settle_pending_state_update()
        request = req or SaveRequest()
        if request.filename:
            return JSONResponse(
                {
                    "ok": False,
                    "detail": "世界线存档不支持覆盖，请新建一个存档节点。",
                },
                status_code=400,
            )
        save_path, error_detail, is_duplicate = await export_save_archive_with_detail()
        if save_path:
            return JSONResponse(
                {"ok": True, "path": save_path, "filename": Path(save_path).name, "duplicate": is_duplicate}
            )
        detail = error_detail or "存档导出失败。"
        routing_logger.error("[save] /api/save 失败: %s", detail)
        return JSONResponse({"ok": False, "detail": detail}, status_code=500)
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        routing_logger.error("[save] /api/save 未捕获异常: %s\n%s", detail, traceback.format_exc())
        return JSONResponse({"ok": False, "detail": detail}, status_code=500)


# =============================================================================
# /api/load
# =============================================================================


class LoadRequest(BaseModel):
    filename: str


@app.post("/api/load")
async def api_load(req: LoadRequest) -> JSONResponse:
    """加载存档。"""
    if memory_consolidation_flow.is_running:
        return JSONResponse(
            {"ok": False, "detail": "记忆整理进行中，请稍后再读档。"},
            status_code=409,
        )
    _invalidate_pending_choices(clear_saved=True)
    await _settle_pending_state_update(cancel=True)
    success = await import_save_archive(req.filename)
    if success:
        reset_entities()
        for name in get_agent_names(include_narrator=True):
            reload_conversation_agent(name)
        raw = load_conversation_history(limit=_RECENT_HISTORY_LIMIT)
        recent = [m for m in (_format_history_message(item) for item in raw) if m]
        last_choices = _load_last_choices()
        return JSONResponse(
            {
                "ok": True,
                "recent": recent,
                "last_choices": last_choices,
                "scene_status": _current_scene_status(),
                "character_count": len(get_agent_names(include_narrator=False)),
            }
        )
    return JSONResponse({"ok": False}, status_code=500)


# =============================================================================
# /api/save-node/{filename} DELETE
# =============================================================================


@app.delete("/api/save-node/{filename}")
async def api_delete_save_node(filename: str) -> JSONResponse:
    """删除没有子分支的单个存档节点。"""
    deleted, reason = delete_save_leaf(filename)
    if deleted:
        return JSONResponse({"ok": True, "deleted": [deleted]})

    if reason == "has_children":
        return JSONResponse(
            {"ok": False, "detail": "这个存档已有子分支，不能单独删除。"},
            status_code=409,
        )
    return JSONResponse({"ok": False, "detail": "存档不存在或文件名非法。"}, status_code=404)


# =============================================================================
# /api/save/{filename} DELETE
# =============================================================================


@app.delete("/api/save/{filename}")
async def api_delete_save(filename: str) -> JSONResponse:
    """删除以指定存档为根的整棵 Game 世界线。"""
    deleted = delete_save_game(filename)
    if deleted is not None:
        return JSONResponse({"ok": True, "deleted": deleted})
    return JSONResponse({"ok": False, "detail": "Game 不存在或文件名非法。"}, status_code=404)


# =============================================================================
# /api/reset
# =============================================================================


class ResetRequest(BaseModel):
    story_id: str = "school"


@app.post("/api/reset")
async def api_reset(req: ResetRequest) -> JSONResponse:
    """重置游戏。"""
    if memory_consolidation_flow.is_running:
        return JSONResponse(
            {"ok": False, "detail": "记忆整理进行中，请稍后再重置。"},
            status_code=409,
        )
    _invalidate_pending_choices(clear_saved=True)
    await _settle_pending_state_update(cancel=True)
    intro_text, opening_text = await reset_game(req.story_id)
    reset_entities()
    for name in get_agent_names(include_narrator=True):
        reload_conversation_agent(name)

    opening_choices_text = load_story_file(req.story_id, "opening_choices.txt")
    choices = [line.strip() for line in opening_choices_text.splitlines() if line.strip()]
    if choices:
        _save_last_choices(choices)

    return JSONResponse(
        {
            "intro": intro_text,
            "opening": opening_text,
            "choices": choices,
            "scene_status": _current_scene_status(),
            "character_count": len(get_agent_names(include_narrator=False)),
        }
    )


# =============================================================================
# /api/characters
# =============================================================================


@app.get("/api/characters")
async def api_characters() -> JSONResponse:
    """返回当前所有角色的身份信息与位置信息。"""
    try:
        narrator_status = read_agent_file("narrator", "status.md")
    except Exception:
        narrator_status = ""

    # 从 narrator status 的 `角色位置` 字段解析各角色位置
    location_text = extract_status_field(narrator_status, "角色位置")
    if narrator_status and not location_text:
        routing_logger.warning("/api/characters: narrator/status.md 中未找到「角色位置」字段，位置信息将为空")
    locations: dict[str, str] = {}
    for line in location_text.split("\n"):
        line = line.strip()
        if not line.startswith("- "):
            continue
        rest = line[2:].strip()
        # 取最先出现的分隔符，避免位置值中包含另一种冒号时分割位置错误
        idx_cn = rest.find("：")
        idx_en = rest.find(":")
        candidates = [i for i in (idx_cn, idx_en) if i != -1]
        if candidates:
            split_at = min(candidates)
            name_part, loc = rest[:split_at], rest[split_at + 1:]
            locations[name_part.strip()] = loc.strip()

    def _resolve_location(agent_name: str, display_name: str) -> str:
        """按优先级匹配角色位置：精确 display_name > 精确 agent_name > 模糊包含。"""
        if display_name in locations:
            return locations[display_name]
        if agent_name in locations:
            return locations[agent_name]
        if not display_name:
            return ""
        # 模糊匹配：仅在命中唯一条目时返回，防止子串重叠时选错
        fuzzy = [v for k, v in locations.items() if display_name in k or k in display_name]
        return fuzzy[0] if len(fuzzy) == 1 else ""

    characters: list[dict[str, str]] = []
    for agent_name in get_agent_names(include_narrator=False):
        # display_name 和 identity 分开捕获：任一失败都不应让角色整体消失
        try:
            display_name = _get_agent_display_name(agent_name)
        except Exception as e:
            routing_logger.warning("[%s] /api/characters 获取显示名失败，降级为 agent_name: %s", agent_name, e)
            display_name = agent_name

        try:
            status_text = read_agent_file(agent_name, "status.md")
            identity = extract_status_field(status_text, "身份")
        except Exception as e:
            routing_logger.warning("[%s] /api/characters 读取 status.md 失败: %s", agent_name, e)
            identity = ""

        characters.append(
            {
                "name": agent_name,
                "display_name": display_name,
                "identity": identity,
                "location": _resolve_location(agent_name, display_name),
            }
        )

    return JSONResponse({"characters": characters})
