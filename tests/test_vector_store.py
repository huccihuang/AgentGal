"""VectorStore tests (layered round/memory behavior)."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio

project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv

    load_dotenv(project_root / ".env")
except ImportError:
    pass

try:
    import memory.vector_store as vector_store_module
    from memory.vector_store import EMBED_API_KEY, EMBED_API_URL, vector_store
except ModuleNotFoundError as exc:
    pytest.skip(f"skip vector_store tests: missing dependency ({exc})", allow_module_level=True)


pytestmark = pytest.mark.skipif(
    not EMBED_API_URL or not EMBED_API_KEY,
    reason="EMBEDDING_API_URL 或 EMBEDDING_API_KEY 未配置，跳过测试",
)

test_db_path = str(project_root / "data" / "test_vectors.sqlite")


@pytest_asyncio.fixture
async def clean_store(monkeypatch):
    monkeypatch.setattr(vector_store_module, "DB_PATH", test_db_path)

    store = vector_store
    if store._db is not None:
        await store._db.close()
    store._db = None
    store._conv_game_date.clear()

    os.makedirs(os.path.dirname(test_db_path), exist_ok=True)
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    await store.init_tables()
    yield store

    if store._background_tasks:
        await asyncio.gather(*list(store._background_tasks), return_exceptions=True)
    if store._db is not None:
        await store._db.close()
        store._db = None
    if os.path.exists(test_db_path):
        os.remove(test_db_path)


async def wait_for_search(
    store,
    agent_name: str,
    query: str,
    kind: str = "round",
    timeout: float = 10.0,
):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        res = store.search(agent_name, query, kind=kind)
        if res:
            return res
        await asyncio.sleep(0.1)
    raise TimeoutError(f"等待搜索结果超时: agent={agent_name}, query={query}, kind={kind}")


def make_character_path(tmp_path):
    def _path(name, subpath=None):
        base = tmp_path / name
        if subpath:
            return str(base / subpath)
        return str(base)

    return _path


def write_memory(tmp_path, agent_name: str, content: str):
    agent_dir = tmp_path / agent_name
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / "memory.md"
    path.write_text(content, encoding="utf-8")
    return path


class TestRoundLayer:
    @pytest.mark.asyncio
    async def test_round_id_with_session_prefix_appends_instead_of_overwrite(self, clean_store):
        store = clean_store
        store.add_round(
            ["lilith", "narrator"],
            "chainlit_sess_a_1",
            "玩家: A\n旁白: **时间**：4月3日 08:00\nlilith: A1",
        )
        store.add_round(
            ["lilith", "narrator"],
            "chainlit_sess_b_1",
            "玩家: B\n旁白: **时间**：4月3日 08:01\nlilith: B1",
        )
        await wait_for_search(store, "lilith", "A1", kind="round")
        await wait_for_search(store, "lilith", "B1", kind="round")
        db = await store._get_db()
        count = (await (await db.execute("SELECT COUNT(*) FROM chunks WHERE round_id LIKE 'chainlit_sess_%_1'")).fetchone())[0]
        assert count == 2

    @pytest.mark.asyncio
    async def test_add_round_and_visibility(self, clean_store):
        store = clean_store
        store.add_round(
            ["lilith", "narrator"],
            "round_1",
            "玩家: 你好\n旁白: **时间**：4月3日 08:00\nlilith: 早上好",
        )
        store.add_round(
            ["mitsuki", "narrator"],
            "round_2",
            "玩家: 去篮球场吗\n旁白: **时间**：4月3日 08:10\nmitsuki: 一起吧",
        )

        res_lilith = await wait_for_search(store, "lilith", "早上好", kind="round")
        res_mitsuki = await wait_for_search(store, "mitsuki", "篮球场", kind="round")
        assert len(res_lilith) >= 1
        assert len(res_mitsuki) >= 1

    @pytest.mark.asyncio
    async def test_duplicate_round_id_overwrite(self, clean_store):
        store = clean_store
        store.add_round(
            ["lilith", "narrator"],
            "same_id",
            "玩家: 第一次\n旁白: **时间**：4月3日 08:00\nlilith: 你好",
        )
        await wait_for_search(store, "lilith", "你好", kind="round")

        store.add_round(
            ["lilith", "narrator"],
            "same_id",
            "玩家: 第二次\n旁白: **时间**：4月3日 08:10\nlilith: 重复",
        )
        await wait_for_search(store, "lilith", "重复", kind="round")
        if store._background_tasks:
            await asyncio.gather(*list(store._background_tasks), return_exceptions=True)

        db = await store._get_db()
        count = (await (await db.execute("SELECT COUNT(*) FROM chunks WHERE round_id = ?", ("same_id",))).fetchone())[0]
        assert count == 1
        content = (await (await db.execute("SELECT content FROM chunks WHERE round_id = ?", ("same_id",))).fetchone())[0]
        assert "重复" in content

    @pytest.mark.asyncio
    async def test_chunk_fields(self, clean_store):
        store = clean_store
        store.add_round(
            ["lilith", "narrator"],
            "round_fields",
            "玩家: 你好\n旁白: **时间**：4月3日 08:00\nlilith: 早上好",
        )
        await wait_for_search(store, "lilith", "早上好", kind="round")
        db = await store._get_db()
        row = await (await db.execute("SELECT round_id, date, created_at, visible_to FROM chunks WHERE round_id = 'round_fields'")).fetchone()
        assert row is not None
        _, date, created_at, vis = row
        assert date == "4月3日"
        assert created_at is not None
        vis_list = json.loads(vis)
        assert "lilith" in vis_list and "narrator" in vis_list


class TestMemoryLayer:
    @pytest.mark.asyncio
    async def test_add_memory_only_target_date(self, clean_store, tmp_path, monkeypatch):
        store = clean_store
        write_memory(
            tmp_path,
            "mitsuki",
            """
# mitsuki 的长期记忆

## 4月3日
- **时间**：4月3日 上午
- **内容**：今天的事让我在意。

## 4月4日
- **时间**：4月4日 上午
- **内容**：我独自待了一会儿。
""".strip(),
        )
        monkeypatch.setattr(store, "character_path", make_character_path(tmp_path))
        await store.add_memory("mitsuki", "4月3日")

        hit = await wait_for_search(store, "mitsuki", "让我在意", kind="memory")
        miss = store.search("mitsuki", "独自待了一会儿", kind="memory")
        assert len(hit) >= 1
        assert not any("独自待了一会儿" in r["content"] for r in miss)

    @pytest.mark.asyncio
    async def test_date_switch_app_style_triggers_add_memory(self, clean_store, tmp_path, monkeypatch):
        store = clean_store
        write_memory(
            tmp_path,
            "lilith",
            """
# lilith 的长期记忆

## 4月3日
- **时间**：4月3日 晚上
- **内容**：这是昨天锚点。
""".strip(),
        )
        monkeypatch.setattr(store, "character_path", make_character_path(tmp_path))

        store.add_round(
            ["lilith", "narrator"],
            "switch_1",
            "玩家: 今天聊聊\n旁白: **时间**：4月3日 09:00\nlilith: 今天内容",
            game_date="4月3日",
        )
        await wait_for_search(store, "lilith", "今天内容", kind="round")

        store.add_round(
            ["lilith", "narrator"],
            "switch_2",
            "玩家: 明天聊聊\n旁白: **时间**：4月4日 09:00\nlilith: 明天内容",
            game_date="4月4日",
        )
        await wait_for_search(store, "lilith", "明天内容", kind="round")
        await store.add_memory("lilith", "4月3日")

        res_mem = await wait_for_search(store, "lilith", "昨天锚点", kind="memory")
        assert any("昨天锚点" in r["content"] for r in res_mem)

    @pytest.mark.asyncio
    async def test_search_all_only_today_round_and_previous_memory(self, clean_store, tmp_path, monkeypatch):
        store = clean_store
        write_memory(
            tmp_path,
            "lilith",
            """
# lilith 的长期记忆

## 4月3日
- **时间**：4月3日 晚上
- **内容**：旧日记忆。

## 4月4日
- **时间**：4月4日 下午
- **内容**：当日记忆。
""".strip(),
        )
        monkeypatch.setattr(store, "character_path", make_character_path(tmp_path))

        store.add_round(
            ["lilith", "narrator"],
            "all_scope_old_round",
            "玩家: 昨天\n旁白: **时间**：4月3日 09:00\nlilith: 旧日对话",
            game_date="4月3日",
        )
        store.add_round(
            ["lilith", "narrator"],
            "all_scope_today_round",
            "玩家: 今天\n旁白: **时间**：4月4日 09:00\nlilith: 当日对话",
            game_date="4月4日",
        )
        await wait_for_search(store, "lilith", "当日对话", kind="round")
        await store.add_memory("lilith", "4月3日")
        await store.add_memory("lilith", "4月4日")

        old_mem_hits = store.search("lilith", "旧日记忆")
        assert any("旧日记忆" in item["content"] for item in old_mem_hits)

        today_mem_hits = store.search("lilith", "当日记忆")
        assert not any("当日记忆" in item["content"] for item in today_mem_hits)

        today_round_hits = store.search("lilith", "当日对话")
        assert any("当日对话" in item["content"] for item in today_round_hits)

        old_round_hits = store.search("lilith", "旧日对话")
        assert not any("旧日对话" in item["content"] for item in old_round_hits)


class TestMaintenance:
    @pytest.mark.asyncio
    async def test_rebuild_from_jsonl(self, clean_store, tmp_path, monkeypatch):
        store = clean_store
        raw_dir = tmp_path / "narrator" / "raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "2024-01-01.jsonl").write_text(
            """\
{"role": "player", "content": "你好", "visible_to": ["lilith", "narrator"]}
{"role": "narrator", "content": "**时间**：4月3日 08:00\\n早上好", "visible_to": ["lilith", "narrator"]}
""",
            encoding="utf-8",
        )
        monkeypatch.setattr(store, "character_path", make_character_path(tmp_path))
        await store.rebuild("narrator")
        res = store.search("lilith", "早上好", kind="round")
        assert len(res) >= 1

    @pytest.mark.asyncio
    async def test_delete_all_agents_partial_and_full(self, clean_store, monkeypatch):
        store = clean_store
        store.add_round(
            ["lilith", "narrator"],
            "partial_1",
            "玩家: hi\n旁白: **时间**：4月3日 10:00\nlilith: only lilith",
        )
        store.add_round(
            ["mitsuki", "narrator"],
            "partial_2",
            "玩家: hi\n旁白: **时间**：4月3日 10:01\nmitsuki: only mitsuki",
        )
        await wait_for_search(store, "lilith", "only lilith", kind="round")
        await wait_for_search(store, "mitsuki", "only mitsuki", kind="round")

        partial = await store.delete_all_agents(["lilith"])
        assert partial == {"lilith": True}
        assert store.search("lilith", "only lilith", kind="round") == []
        assert len(store.search("mitsuki", "only mitsuki", kind="round")) >= 1

        monkeypatch.setattr(vector_store_module, "get_agent_names", lambda: ["lilith", "mitsuki", "narrator"])
        full = await store.delete_all_agents(["lilith", "mitsuki", "narrator"])
        assert full == {"lilith": True, "mitsuki": True, "narrator": True}
        db = await store._get_db()
        chunk_count = (await (await db.execute("SELECT COUNT(*) FROM chunks")).fetchone())[0]
        vec_count = (await (await db.execute("SELECT COUNT(*) FROM vec_chunks")).fetchone())[0]
        assert chunk_count == 0
        assert vec_count == 0
