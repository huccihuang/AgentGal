"""测试 server 层 state_updater 后台任务协调。"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

project_root = Path(__file__).parent.parent
os.chdir(project_root)

try:
    from agents.schema import NarratorOutput
    import server as server_module
except ModuleNotFoundError as exc:
    pytest.skip(f"skip server tests: missing dependency ({exc})", allow_module_level=True)


@pytest.fixture(autouse=True)
def isolate_last_choices_file(monkeypatch, tmp_path):
    monkeypatch.setattr(server_module, "_LAST_CHOICES_FILE", tmp_path / "last_choices.json")
    server_module._pending_choices_task = None
    server_module._choices_generation_token = 0
    yield
    task = server_module._pending_choices_task
    if task is not None and not task.done():
        task.cancel()
    server_module._pending_choices_task = None
    server_module._choices_generation_token = 0


def _parse_sse_chunk(chunk: str) -> dict:
    return json.loads(chunk.removeprefix("data: ").strip())


def _narrator_output(**overrides) -> NarratorOutput:
    data = {
        "targets": ["alice"],
        "date": "4月3日 星期三",
        "time": "16:10",
        "location": "走廊",
        "present_characters": {"北原悠": "门口", "Alice": "窗边"},
        "scene_description": "场景推进",
        "new_characters": [],
    }
    data.update(overrides)
    return NarratorOutput(**data)



@pytest.mark.asyncio
async def test_settle_pending_state_update_waits_for_background_task(monkeypatch):
    started = False

    async def fake_update_state(_self, *_args):
        nonlocal started
        started = True

    monkeypatch.setattr(server_module.narrator.__class__, "update_state", fake_update_state)
    server_module._pending_state_update_task = None

    server_module._start_state_update()
    assert server_module._pending_state_update_task is not None

    await server_module._settle_pending_state_update()

    assert started is True
    assert server_module._pending_state_update_task is None


@pytest.mark.asyncio
async def test_settle_pending_state_update_cancels_background_task(monkeypatch):
    release = False

    async def fake_update_state(_self, *_args):
        while not release:
            await server_module.asyncio.sleep(0)

    monkeypatch.setattr(server_module.narrator.__class__, "update_state", fake_update_state)
    server_module._pending_state_update_task = None

    server_module._start_state_update()
    task = server_module._pending_state_update_task
    assert task is not None

    await server_module._settle_pending_state_update(cancel=True)

    assert task.cancelled()
    assert server_module._pending_state_update_task is None


@pytest.mark.asyncio
async def test_start_state_update_coalesces_while_running(monkeypatch):
    started = server_module.asyncio.Event()
    release = server_module.asyncio.Event()
    calls = 0

    async def fake_update_state(_self, *_args):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()

    monkeypatch.setattr(server_module.narrator.__class__, "update_state", fake_update_state)
    server_module._pending_state_update_task = None
    server_module._pending_state_update_requested = False

    server_module._start_state_update()
    task = server_module._pending_state_update_task
    assert task is not None
    await started.wait()

    server_module._start_state_update()
    server_module._start_state_update()

    assert server_module._pending_state_update_task is task
    assert server_module._pending_state_update_requested is True

    release.set()
    await server_module._settle_pending_state_update()

    assert calls == 2
    assert server_module._pending_state_update_task is None
    assert server_module._pending_state_update_requested is False


@pytest.mark.asyncio
async def test_chat_stream_does_not_wait_for_pending_state_update(monkeypatch):
    release = server_module.asyncio.Event()

    async def blocking_task():
        await release.wait()

    async def fake_route(_self, _user_input, *, observation_mode=False):
        return None, False

    task = server_module.asyncio.create_task(blocking_task())
    server_module._pending_state_update_task = task
    server_module._pending_state_update_requested = False
    monkeypatch.setattr(server_module.narrator.__class__, "route", fake_route)

    try:
        chunks = await server_module.asyncio.wait_for(
            _collect_stream(server_module._chat_stream("继续")),
            timeout=0.1,
        )
    finally:
        release.set()
        await task
        server_module._pending_state_update_task = None
        server_module._pending_state_update_requested = False

    assert chunks == ['data: {"type": "done", "consolidating": false}\n\n']


@pytest.mark.asyncio
async def test_chat_stream_yields_response_done_before_choices(monkeypatch):
    release_choices = server_module.asyncio.Event()
    choices_started = server_module.asyncio.Event()
    maintenance_calls: list[object] = []

    async def fake_route(_self, _user_input, *, observation_mode=False):
        return _narrator_output(), True

    async def fake_broadcast_player_message(_targets, _user_input):
        return None

    async def fake_broadcast_narrator_output(_targets, _output):
        return 7

    async def fake_run_agent_in_scene(*_args, **_kwargs):
        return "角色回应"

    async def fake_generate_choices(_scene_description, _agent_responses):
        choices_started.set()
        await release_choices.wait()
        return ["继续追问"]

    def fake_start_state_update():
        maintenance_calls.append("state")

    def fake_schedule_detect_and_consolidate(turn):
        maintenance_calls.append(("memory", turn))

    monkeypatch.setattr(server_module.narrator.__class__, "route", fake_route)
    monkeypatch.setattr(
        server_module.message_router,
        "broadcast_player_message",
        fake_broadcast_player_message,
    )
    monkeypatch.setattr(
        server_module.message_router,
        "broadcast_narrator_output",
        fake_broadcast_narrator_output,
    )
    monkeypatch.setattr(server_module, "run_agent_in_scene", fake_run_agent_in_scene)
    monkeypatch.setattr(server_module, "generate_choices", fake_generate_choices)
    monkeypatch.setattr(server_module, "_get_agent_display_name", lambda name: name)
    monkeypatch.setattr(server_module, "_start_state_update", fake_start_state_update)
    monkeypatch.setattr(
        server_module.memory_consolidation_flow,
        "schedule_detect_and_consolidate",
        fake_schedule_detect_and_consolidate,
    )

    stream = server_module._chat_stream("继续")
    try:
        chunks = [
            await server_module.asyncio.wait_for(anext(stream), timeout=0.1),
            await server_module.asyncio.wait_for(anext(stream), timeout=0.1),
            await server_module.asyncio.wait_for(anext(stream), timeout=0.1),
        ]

        assert [_parse_sse_chunk(chunk)["type"] for chunk in chunks] == [
            "narrator",
            "agent",
            "response_done",
        ]
        await server_module.asyncio.wait_for(choices_started.wait(), timeout=0.1)
        assert maintenance_calls == ["state", ("memory", 7)]

        release_choices.set()
        choices_chunk = await server_module.asyncio.wait_for(anext(stream), timeout=0.1)
        done_chunk = await server_module.asyncio.wait_for(anext(stream), timeout=0.1)

        assert _parse_sse_chunk(choices_chunk) == {
            "type": "choices",
            "choices": ["继续追问"],
        }
        assert _parse_sse_chunk(done_chunk)["type"] == "done"
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_new_chat_cancels_pending_choices(monkeypatch):
    choices_started = server_module.asyncio.Event()
    choices_cancelled = server_module.asyncio.Event()

    async def fake_route(_self, user_input, *, observation_mode=False):
        if user_input == "第一轮":
            return _narrator_output(scene_description="第一轮场景"), True
        return None, False

    async def fake_broadcast_player_message(_targets, _user_input):
        return None

    async def fake_broadcast_narrator_output(_targets, _output):
        return 8

    async def fake_run_agent_in_scene(*_args, **_kwargs):
        return "第一轮回应"

    async def fake_generate_choices(_scene_description, _agent_responses):
        choices_started.set()
        try:
            await server_module.asyncio.sleep(3600)
        except server_module.asyncio.CancelledError:
            choices_cancelled.set()
            raise
        return ["过期选项"]

    monkeypatch.setattr(server_module.narrator.__class__, "route", fake_route)
    monkeypatch.setattr(
        server_module.message_router,
        "broadcast_player_message",
        fake_broadcast_player_message,
    )
    monkeypatch.setattr(
        server_module.message_router,
        "broadcast_narrator_output",
        fake_broadcast_narrator_output,
    )
    monkeypatch.setattr(server_module, "run_agent_in_scene", fake_run_agent_in_scene)
    monkeypatch.setattr(server_module, "generate_choices", fake_generate_choices)
    monkeypatch.setattr(server_module, "_get_agent_display_name", lambda name: name)
    monkeypatch.setattr(server_module, "_start_state_update", lambda: None)
    monkeypatch.setattr(
        server_module.memory_consolidation_flow,
        "schedule_detect_and_consolidate",
        lambda _turn: None,
    )

    first_stream = server_module._chat_stream("第一轮")
    try:
        first_chunks = [
            await server_module.asyncio.wait_for(anext(first_stream), timeout=0.1),
            await server_module.asyncio.wait_for(anext(first_stream), timeout=0.1),
            await server_module.asyncio.wait_for(anext(first_stream), timeout=0.1),
        ]
        assert _parse_sse_chunk(first_chunks[-1])["type"] == "response_done"

        pending_first_choice = server_module.asyncio.create_task(anext(first_stream))
        await server_module.asyncio.wait_for(choices_started.wait(), timeout=0.1)

        second_chunks = [
            chunk async for chunk in server_module._chat_stream("第二轮")
        ]

        await server_module.asyncio.wait_for(choices_cancelled.wait(), timeout=0.1)
        with pytest.raises(StopAsyncIteration):
            await server_module.asyncio.wait_for(pending_first_choice, timeout=0.1)

        assert [_parse_sse_chunk(chunk)["type"] for chunk in second_chunks] == ["done"]
        assert server_module._load_last_choices() == []
    finally:
        await first_stream.aclose()


async def _collect_stream(stream):
    return [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_api_save_returns_error_detail(monkeypatch):
    async def fake_export_save_archive_with_detail():
        return None, "sqlite 已锁定", False

    logged: list[str] = []

    def fake_log_error(message, *args):
        logged.append(message % args if args else message)

    monkeypatch.setattr(
        server_module,
        "export_save_archive_with_detail",
        fake_export_save_archive_with_detail,
    )
    monkeypatch.setattr(server_module.routing_logger, "error", fake_log_error)
    server_module._pending_state_update_task = None

    response = await server_module.api_save()

    assert response.status_code == 500
    assert json.loads(response.body) == {"ok": False, "detail": "sqlite 已锁定"}
    assert logged == ["[save] /api/save 失败: sqlite 已锁定"]


@pytest.mark.asyncio
async def test_api_save_rejects_target_filename(monkeypatch):
    called = False

    async def fake_export_save_archive_with_detail():
        nonlocal called
        called = True
        return "/tmp/school_slot.zip", None, False

    monkeypatch.setattr(
        server_module,
        "export_save_archive_with_detail",
        fake_export_save_archive_with_detail,
    )
    server_module._pending_state_update_task = None

    response = await server_module.api_save(
        server_module.SaveRequest(filename="school_slot.zip")
    )

    assert response.status_code == 400
    assert json.loads(response.body) == {
        "ok": False,
        "detail": "世界线存档不支持覆盖，请新建一个存档节点。",
    }
    assert called is False


@pytest.mark.asyncio
async def test_api_save_catches_unhandled_exception(monkeypatch):
    async def fake_settle_pending_state_update(*, cancel=False):
        raise RuntimeError("pending task 爆了")

    logged: list[str] = []

    def fake_log_error(message, *args):
        logged.append(message % args if args else message)

    monkeypatch.setattr(
        server_module,
        "_settle_pending_state_update",
        fake_settle_pending_state_update,
    )
    monkeypatch.setattr(server_module.routing_logger, "error", fake_log_error)

    response = await server_module.api_save()

    assert response.status_code == 500
    assert json.loads(response.body) == {
        "ok": False,
        "detail": "RuntimeError: pending task 爆了",
    }
    assert len(logged) == 1
    assert logged[0].startswith("[save] /api/save 未捕获异常: RuntimeError: pending task 爆了")


@pytest.mark.asyncio
async def test_api_delete_save_deletes_game_tree(monkeypatch):
    monkeypatch.setattr(
        server_module,
        "delete_save_game",
        lambda filename: ["root.zip", "child.zip"] if filename == "root.zip" else None,
    )

    response = await server_module.api_delete_save("root.zip")

    assert response.status_code == 200
    assert json.loads(response.body) == {
        "ok": True,
        "deleted": ["root.zip", "child.zip"],
    }


@pytest.mark.asyncio
async def test_api_delete_save_node_rejects_parent(monkeypatch):
    monkeypatch.setattr(
        server_module,
        "delete_save_leaf",
        lambda filename: (None, "has_children"),
    )

    response = await server_module.api_delete_save_node("root.zip")

    assert response.status_code == 409
    assert json.loads(response.body) == {
        "ok": False,
        "detail": "这个存档已有子分支，不能单独删除。",
    }


@pytest.mark.asyncio
async def test_api_delete_save_node_deletes_leaf(monkeypatch):
    monkeypatch.setattr(
        server_module,
        "delete_save_leaf",
        lambda filename: (filename, None),
    )

    response = await server_module.api_delete_save_node("leaf.zip")

    assert response.status_code == 200
    assert json.loads(response.body) == {"ok": True, "deleted": ["leaf.zip"]}


@pytest.mark.asyncio
async def test_api_memory_graph_links_understandings_to_episodes(monkeypatch):
    monkeypatch.setattr(
        server_module,
        "get_agent_names",
        lambda include_narrator=True: ["alice"] if not include_narrator else ["alice", "narrator"],
    )
    monkeypatch.setattr(
        server_module,
        "_get_agent_display_name",
        lambda agent_name: "Alice" if agent_name == "alice" else agent_name,
    )
    monkeypatch.setattr(
        server_module,
        "read_memory_jsonl",
        lambda agent_name: [
            server_module.EpisodeMemory(
                id="e1",
                date="4月3日",
                time="4月3日 16:10",
                title="旧阅览室",
                content="她请求玩家保密。",
                importance=4,
                keywords=["保密"],
                raw_dialogue="[turn=12] 玩家: 我会保密 [turn=12] 旁白: **时间**：4月3日 16:10",
            )
        ],
    )
    monkeypatch.setattr(
        server_module,
        "read_understandings",
        lambda agent_name: {
            "u1": server_module.Understanding(
                id="u1",
                subject="玩家值得信任",
                content="玩家会认真履行约定。",
                keywords=["信任"],
                linked_episodes=["e1", "missing-e2"],
                history=[
                    {
                        "episode_id": "e1",
                        "date": "4月3日",
                        "title": "旧阅览室",
                        "content": "玩家会认真履行约定。",
                    }
                ],
            )
        },
    )

    response = await server_module.api_memory_graph("alice")
    body = json.loads(response.body)

    assert response.status_code == 200
    assert body["agents"] == [
        {
            "name": "alice",
            "display_name": "Alice",
            "episode_count": 1,
            "understanding_count": 1,
            "edge_count": 2,
        }
    ]
    assert body["selected_agent"] == "alice"
    assert body["stats"] == {
        "episode_count": 1,
        "understanding_count": 1,
        "edge_count": 2,
        "missing_episode_count": 1,
    }
    assert {node["id"] for node in body["nodes"]} == {
        "episode:e1",
        "understanding:u1",
        "episode:missing-e2",
    }
    episode = next(node for node in body["nodes"] if node["id"] == "episode:e1")
    understanding = next(node for node in body["nodes"] if node["id"] == "understanding:u1")
    assert understanding["meta"]["history"] == [
        {
            "episode_id": "e1",
            "date": "4月3日",
            "title": "旧阅览室",
            "content": "玩家会认真履行约定。",
        }
    ]
    assert episode["meta"]["date"] == "4月3日"
    assert episode["meta"]["time"] == "4月3日 16:10"
    assert "[turn=" not in episode["meta"]["raw_dialogue_preview"]
    assert "旁白" not in episode["meta"]["raw_dialogue_preview"]
    assert episode["meta"]["raw_dialogue_preview"] == "玩家：我会保密"
    assert {edge["from"] for edge in body["edges"]} == {"episode:e1", "episode:missing-e2"}
    assert {edge["to"] for edge in body["edges"]} == {"understanding:u1"}
    assert next(edge for edge in body["edges"] if edge["from"] == "episode:e1")["meta"] == {
        "type": "edge",
        "episode_id": "e1",
        "understanding_id": "u1",
    }


@pytest.mark.asyncio
async def test_api_memory_graph_rejects_unknown_agent(monkeypatch):
    monkeypatch.setattr(
        server_module,
        "get_agent_names",
        lambda include_narrator=True: ["alice"] if not include_narrator else ["alice", "narrator"],
    )
    monkeypatch.setattr(server_module, "read_memory_jsonl", lambda agent_name: [])
    monkeypatch.setattr(server_module, "read_understandings", lambda agent_name: {})

    response = await server_module.api_memory_graph("bob")

    assert response.status_code == 404
    assert json.loads(response.body) == {"detail": "角色不存在。"}


@pytest.mark.asyncio
async def test_chat_stream_emits_created_character_identity(monkeypatch):
    async def fake_settle_pending_state_update(*, cancel=False):
        return None

    async def fake_route(_self, _user_input, *, observation_mode=False):
        return _narrator_output(
            targets=[],
            new_characters=[
                {
                    "name_hint": "桥本志津",
                    "background_hint": "Alice 的家人，在附近工作，偶尔来探望。",
                }
            ],
        ), True

    async def fake_bootstrap_new_characters(_specs, _targets):
        return [], [
            SimpleNamespace(
                character_id="mitsukimom",
                display_name="桥本志津",
                identity="美月的妈妈，来学校接她放学的家长。",
            )
        ]

    async def fake_broadcast_player_message(_targets, _user_input):
        return None

    monkeypatch.setattr(
        server_module,
        "_settle_pending_state_update",
        fake_settle_pending_state_update,
    )
    monkeypatch.setattr(server_module.narrator.__class__, "route", fake_route)
    monkeypatch.setattr(
        server_module,
        "bootstrap_new_characters",
        fake_bootstrap_new_characters,
    )
    monkeypatch.setattr(
        server_module.message_router,
        "broadcast_player_message",
        fake_broadcast_player_message,
    )

    chunks = [chunk async for chunk in server_module._chat_stream("来个新角色")]

    assert len(chunks) == 3
    created_event = json.loads(chunks[0].removeprefix("data: ").strip())
    assert created_event == {
        "type": "system",
        "title": "角色已创建",
        "name": "桥本志津",
        "identity": "美月的妈妈，来学校接她放学的家长。",
        "character_id": "mitsukimom",
    }
    narrator_event = json.loads(chunks[1].removeprefix("data: ").strip())
    assert narrator_event["type"] == "narrator"
    assert narrator_event["payload"]["scene_description"] == "场景推进"
    done_event = json.loads(chunks[2].removeprefix("data: ").strip())
    assert done_event == {"type": "done", "consolidating": False}
