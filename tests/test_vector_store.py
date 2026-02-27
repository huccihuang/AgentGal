"""
向量库 pytest 测试

测试 sqlite-vec 向量库的增/查/删/重建功能
使用真实 embedding API 请求（不 mock）
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio

# 设置项目根目录，确保相对路径与模块导入一致
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# 必须在导入 vector_store 之前加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass

# 现在导入 vector_store，此时环境变量已加载
try:
    import memory.vector_store as vector_store_module
    from memory.vector_store import vector_store, EMBED_DIM, EMBED_API_URL, EMBED_API_KEY
except ModuleNotFoundError as exc:
    pytest.skip(f"skip vector_store tests: missing dependency ({exc})", allow_module_level=True)


# 检查 embedding 配置是否可用
pytestmark = pytest.mark.skipif(
    not EMBED_API_URL or not EMBED_API_KEY,
    reason="EMBEDDING_API_URL 或 EMBEDDING_API_KEY 未配置，跳过测试"
)


# 使用测试数据库路径，避免污染真实数据
test_db_path = str(project_root / "data" / "test_vectors.sqlite")


@pytest_asyncio.fixture
async def clean_store(monkeypatch):
    """提供清理后的 VectorStore，每个测试隔离。"""
    # 使用 monkeypatch 修改 DB_PATH，避免全局污染
    monkeypatch.setattr(vector_store_module, "DB_PATH", test_db_path)

    store = vector_store

    # 重置连接状态，避免跨用例共享缓存
    if store._db is not None:
        await store._db.close()
    store._db = None
    store._conv_game_date.clear()

    # 确保目录存在
    os.makedirs(os.path.dirname(test_db_path), exist_ok=True)

    # 删除旧测试数据库
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    await store.init_tables()

    yield store

    # 清理：关闭数据库连接
    if store._db is not None:
        await store._db.close()
        store._db = None

    # 删除测试数据库文件
    if os.path.exists(test_db_path):
        os.remove(test_db_path)


async def wait_for_search(store, agent_name: str, query: str, kind: str = "round", timeout: float = 10.0):
    """轮询等待向量检索可命中（超时抛出异常）。"""
    deadline = asyncio.get_event_loop().time() + timeout
    last_error = None

    while asyncio.get_event_loop().time() < deadline:
        try:
            res = store.search(agent_name, query, kind=kind)
            if res:
                return res
        except Exception as e:
            last_error = e
        await asyncio.sleep(0.1)

    error_msg = f"等待搜索结果超时: agent={agent_name}, query={query}, kind={kind}"
    if last_error:
        error_msg += f", last_error={last_error}"
    raise TimeoutError(error_msg)


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


class TestVectorStoreBasic:
    """基础增查删测试"""

    @pytest.mark.asyncio
    async def test_add_round_and_search(self, clean_store):
        """测试写入轮次并搜索"""
        store = clean_store

        # 写入两轮：分别对不同角色可见
        store.add_round(
            ["lilith", "narrator"],
            "testcase_1",
            "玩家: 你好\n旁白: **时间**：4月3日 08:00\nlilith: 早上好",
        )
        store.add_round(
            ["mitsuki", "narrator"],
            "testcase_2",
            "玩家: 去篮球场吗\n旁白: **时间**：4月3日 08:10\nmitsuki: 一起吧",
        )

        # 等待异步写入完成并验证结果
        res_lilith = await wait_for_search(store, "lilith", "早上好", kind="round")
        res_mitsuki = await wait_for_search(store, "mitsuki", "篮球场", kind="round")

        assert len(res_lilith) >= 1, "lilith 应该能找到自己的记录"
        assert len(res_mitsuki) >= 1, "mitsuki 应该能找到自己的记录"

    @pytest.mark.asyncio
    async def test_visibility_isolation(self, clean_store):
        """测试可见性隔离：相同内容，角色只能看到自己的记录"""
        store = clean_store

        # 写入两轮，内容相同但对不同角色可见
        store.add_round(
            ["lilith", "narrator"],
            "testcase_1",
            "玩家: 你好\n旁白: **时间**：4月3日 08:00\nlilith: 早上好",
        )
        store.add_round(
            ["mitsuki", "narrator"],
            "testcase_2",
            "玩家: 你好\n旁白: **时间**：4月3日 08:00\nlilith: 早上好",
        )

        # 等待两组数据都写入完成并验证结果
        res_lilith = await wait_for_search(store, "lilith", "早上好", kind="round")
        res_mitsuki = await wait_for_search(store, "mitsuki", "早上好", kind="round")

        assert len(res_lilith) == 1, f"lilith 应该只能看到1条记录，实际看到{len(res_lilith)}条"
        assert len(res_mitsuki) == 1, f"mitsuki 应该只能看到1条记录，实际看到{len(res_mitsuki)}条"

    @pytest.mark.asyncio
    async def test_chunk_fields(self, clean_store):
        """验证 chunk 字段完整性：date/created_at/visible_to"""
        store = clean_store

        store.add_round(
            ["lilith", "narrator"],
            "testcase_1",
            "玩家: 你好\n旁白: **时间**：4月3日 08:00\nlilith: 早上好",
        )
        await wait_for_search(store, "lilith", "早上好", kind="round")

        db = await store._get_db()
        cur = await db.execute(
            "SELECT round_id, date, created_at, visible_to FROM chunks ORDER BY id ASC LIMIT 1"
        )
        row = await cur.fetchone()

        assert row is not None, "应该有数据写入"
        rid, date, created_at, vis = row

        assert rid == "testcase_1"
        assert date == "4月3日", f"日期解析错误: {date}"
        assert created_at is not None, "created_at 不应为空"
        # 验证 visible_to 是合法 JSON
        vis_list = json.loads(vis)
        assert isinstance(vis_list, list)
        assert "lilith" in vis_list
        assert "narrator" in vis_list

    @pytest.mark.asyncio
    async def test_delete(self, clean_store):
        """测试删除指定角色的可见记录"""
        store = clean_store

        # 写入数据：只对 lilith 可见
        store.add_round(
            ["lilith", "narrator"],
            "testcase_1",
            "玩家: 你好\n旁白: **时间**：4月3日 08:00\nlilith: 早上好",
        )
        # 确认有数据
        res_before = await wait_for_search(store, "lilith", "早上好", kind="round")
        assert len(res_before) >= 1, "删除前应该有数据"

        # 删除 lilith 的可见记录
        ok = await store.delete("lilith")
        assert ok is True, "删除应该成功"

        # 验证删除后查询为空
        res_after = store.search("lilith", "早上好", kind="round")
        assert len(res_after) == 0, "删除后应该没有数据"


class TestVectorStoreRebuild:
    """重建功能测试"""

    @pytest.mark.asyncio
    async def test_rebuild_from_jsonl(self, clean_store, tmp_path, monkeypatch):
        """测试从 narrator/raw/*.jsonl 回放索引"""
        store = clean_store

        # 创建模拟的 jsonl 数据目录
        raw_dir = tmp_path / "narrator" / "raw"
        raw_dir.mkdir(parents=True)

        # 创建模拟的 jsonl 文件：两轮对话，分别可见给不同角色
        jsonl_content = """\
{"role": "player", "content": "你好", "visible_to": ["lilith", "narrator"]}
{"role": "narrator", "content": "**时间**：4月3日 08:00\\n早上好", "visible_to": ["lilith", "narrator"]}
{"role": "player", "content": "去篮球场吗", "visible_to": ["mitsuki", "narrator"]}
{"role": "narrator", "content": "**时间**：4月3日 08:10\\n一起吧", "visible_to": ["mitsuki", "narrator"]}
"""
        (raw_dir / "2024-01-01.jsonl").write_text(jsonl_content, encoding="utf-8")

        # mock store 实例的 character_path 方法
        monkeypatch.setattr(store, "character_path", make_character_path(tmp_path))

        # 执行重建：jsonl -> 轮次 -> 向量库
        await store.rebuild("narrator")

        # 验证重建后可以搜索到数据
        res = store.search("lilith", "早上好", kind="round")
        assert len(res) >= 1, "重建后应该能搜索到 lilith 的记录"


class TestVectorStoreEdgeCases:
    """边界情况测试"""

    @pytest.mark.asyncio
    async def test_empty_search(self, clean_store):
        """测试空查询返回空结果"""
        store = clean_store

        res = store.search("lilith", "", kind="round")
        assert res == [], "空查询应该返回空列表"

        res = store.search("lilith", "   ", kind="round")
        assert res == [], "空白查询应该返回空列表"

    @pytest.mark.asyncio
    async def test_search_nonexistent_agent(self, clean_store):
        """测试搜索不存在的角色"""
        store = clean_store

        # 不存在的角色应不会命中任何 chunk
        res = store.search("nonexistent", "查询", kind="round")
        assert res == [], "不存在的角色应该返回空列表"

    @pytest.mark.asyncio
    async def test_duplicate_round_id(self, clean_store):
        """测试重复 round_id 会覆盖旧数据"""
        store = clean_store

        # 写入相同 round_id 两次：第二次应覆盖
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
        # 等待后台任务完成，避免写入未落盘时读取数据库
        if store._background_tasks:
            await asyncio.gather(*list(store._background_tasks), return_exceptions=True)

        # 查询数据库确认只写入了一条
        db = await store._get_db()
        cur = await db.execute("SELECT COUNT(*) FROM chunks WHERE round_id = ?", ("same_id",))
        count = (await cur.fetchone())[0]
        assert count == 1, f"重复 round_id 应该只保留一条，实际有{count}条"

        # 验证内容已更新为第二次的
        cur = await db.execute("SELECT content FROM chunks WHERE round_id = ?", ("same_id",))
        content = (await cur.fetchone())[0]
        assert "重复" in content, "内容应该被更新为第二次的"

    @pytest.mark.asyncio
    async def test_multiple_conv_id_isolation(self, clean_store):
        """测试不同 conversation_id 的数据隔离（通过 round_id 前缀区分）"""
        store = clean_store

        store.add_round(["lilith", "narrator"], "conv1_1", "玩家: 你好\n旁白: **时间**：4月3日\nlilith: 早上好")
        store.add_round(["lilith", "narrator"], "conv2_1", "玩家: 再见\n旁白: **时间**：4月3日\nlilith: 晚安")

        # 等待两组数据都写入完成并验证
        res1 = await wait_for_search(store, "lilith", "你好", kind="round")
        res2 = await wait_for_search(store, "lilith", "再见", kind="round")

        assert len(res1) >= 1, "应该能搜索到第一个对话"
        assert len(res2) >= 1, "应该能搜索到第二个对话"

    @pytest.mark.asyncio
    async def test_layered_search_memory_and_round(self, clean_store, tmp_path, monkeypatch):
        """测试分层检索：memory 与 round 互不混淆。"""
        store = clean_store

        # 使用临时目录承载角色 memory.md
        write_memory(
            tmp_path,
            "lilith",
            (
                "# lilith 的长期记忆\n\n"
                "## 4月2日\n"
                "- **时间**：4月2日 晚上\n"
                "- **地点**：天台\n"
                "- **在场**：莉莉丝、玩家\n"
                "- **内容**：这是昨天的长期记忆锚点。\n\n"
                "## 4月3日\n"
                "- **时间**：4月3日 上午\n"
                "- **地点**：教室\n"
                "- **在场**：莉莉丝、玩家\n"
                "- **内容**：这是今天的长期记忆，不应进 memory 层。"
            ),
        )

        monkeypatch.setattr(store, "character_path", make_character_path(tmp_path))

        # 触发一次 round 写入（"今天"语义），round 层仍可检索
        store.add_round(
            ["lilith", "narrator"],
            "layer_case_1",
            "玩家: 现在聊今天\n旁白: **时间**：4月3日 09:00\nlilith: 今天轮次内容",
            game_date="4月3日",
        )
        # 等待 round 层写入完成
        await wait_for_search(store, "lilith", "今天轮次内容", kind="round")

        # 索引长期记忆层（只索引 4月2日）
        await store.add_memory("lilith", "4月2日")

        # 查询"昨天锚点"应命中长期记忆层
        res_old = await wait_for_search(store, "lilith", "昨天的长期记忆锚点", kind="memory")
        assert any("昨天的长期记忆锚点" in r["content"] for r in res_old), "应该能搜索到昨天的记忆"

        # 查询"今天长期记忆"不应从 memory 层命中
        res_today_mem = store.search("lilith", "今天的长期记忆", kind="memory")
        assert not any("今天的长期记忆" in r["content"] for r in res_today_mem), "4月3日不应在 memory 层"

        # 查询"今天轮次内容"应命中 round 层
        res_today_round = store.search("lilith", "今天轮次内容", kind="round")
        assert any("今天轮次内容" in r["content"] for r in res_today_round), "应该能搜索到今天的轮次"

    @pytest.mark.asyncio
    async def test_delete_all_agents_partial(self, clean_store):
        """测试 delete_all_agents 的逐角色删除路径。"""
        store = clean_store

        store.add_round(
            ["lilith", "narrator"],
            "partial_1",
            "玩家: 你好\n旁白: **时间**：4月3日 10:00\nlilith: 仅 lilith 可见",
        )
        store.add_round(
            ["mitsuki", "narrator"],
            "partial_2",
            "玩家: 你好\n旁白: **时间**：4月3日 10:01\nmitsuki: 仅 mitsuki 可见",
        )

        await wait_for_search(store, "lilith", "仅 lilith 可见", kind="round")
        await wait_for_search(store, "mitsuki", "仅 mitsuki 可见", kind="round")

        # 删除 lilith 的数据
        result = await store.delete_all_agents(["lilith"])
        assert result == {"lilith": True}

        res_lilith = store.search("lilith", "仅 lilith 可见", kind="round")
        res_mitsuki = store.search("mitsuki", "仅 mitsuki 可见", kind="round")
        assert len(res_lilith) == 0, "lilith 的数据应该被删除"
        assert len(res_mitsuki) >= 1, "mitsuki 的数据应该保留"

    @pytest.mark.asyncio
    async def test_delete_all_agents_full_clear(self, clean_store, tmp_path, monkeypatch):
        """测试 delete_all_agents 命中全角色时走全量清理。"""
        store = clean_store

        # 先写入 round 层
        store.add_round(
            ["lilith", "narrator"],
            "full_1",
            "玩家: 你好\n旁白: **时间**：4月3日 11:00\nlilith: round 数据",
        )
        await wait_for_search(store, "lilith", "round 数据", kind="round")

        # 再写入 memory 层，确保全量清理会清除两层
        write_memory(
            tmp_path,
            "lilith",
            (
                "# lilith 的长期记忆\n\n"
                "## 4月2日\n"
                "- **时间**：4月2日 晚上\n"
                "- **地点**：天台\n"
                "- **在场**：莉莉丝、玩家\n"
                "- **内容**：这是昨天的长期记忆锚点。"
            ),
        )

        monkeypatch.setattr(store, "character_path", make_character_path(tmp_path))
        await store.add_memory("lilith", "4月2日")

        # 等待 memory 层写入完成
        await wait_for_search(store, "lilith", "昨天的长期记忆锚点", kind="memory")

        # monkeypatch 全角色集合，强制进入全量清理分支
        monkeypatch.setattr(vector_store_module, "get_agent_names", lambda: ["lilith", "mitsuki", "narrator"])
        result = await store.delete_all_agents(["lilith", "mitsuki", "narrator"])
        assert result == {"lilith": True, "mitsuki": True, "narrator": True}

        db = await store._get_db()
        chunk_count = (await (await db.execute("SELECT COUNT(*) FROM chunks")).fetchone())[0]
        vec_count = (await (await db.execute("SELECT COUNT(*) FROM vec_chunks")).fetchone())[0]
        assert chunk_count == 0, f"chunks 表应该为空，实际有{chunk_count}条"
        assert vec_count == 0, f"vec_chunks 表应该为空，实际有{vec_count}条"


class TestVectorStoreMemoryIndexing:
    """长期记忆索引测试"""

    @pytest.mark.asyncio
    async def test_search_kind_memory_excludes_dialogue(self, clean_store):
        """只检索 memory 时，不应返回仅有的对话数据。"""
        store = clean_store

        store.add(
            ["mitsuki"],
            "dialogue_1",
            "玩家: 你好\n旁白: **时间**：4月3日 08:00\nmitsuki: 早上好",
            kind="dialogue",
        )
        # 等待对话写入 round 层完成
        await wait_for_search(store, "mitsuki", "早上好", kind="round")

        # 查询 memory 层应该为空
        res_mem = store.search("mitsuki", "早上好", kind="memory")
        assert res_mem == [], "仅有对话时，kind=memory 应该返回空"

    @pytest.mark.asyncio
    async def test_add_memory_splits_events(self, clean_store, tmp_path, monkeypatch):
        """索引指定日期的 memory.md，按事件块拆分并可检索。"""
        store = clean_store

        write_memory(
            tmp_path,
            "mitsuki",
            """
# mitsuki 的长期记忆

## 4月3日
- **时间**：4月3日 上午 08:18 - 08:52
- **地点**：教室
- **在场**：桥本美月、玩家（小明）、莉莉丝（转学生）
- **内容**：转学生莉莉丝到来，她的眼神让我在意。

- **时间**：4月3日 中午 12:00 - 12:30
- **地点**：走廊
- **在场**：桥本美月、玩家（小明）
- **内容**：我找他确认下午的安排。
""".strip(),
        )

        monkeypatch.setattr(store, "character_path", make_character_path(tmp_path))
        await store.add_memory("mitsuki", "4月3日")

        # 等待索引完成并验证结果
        res1 = await wait_for_search(store, "mitsuki", "眼神让我在意", kind="memory")
        res2 = await wait_for_search(store, "mitsuki", "下午的安排", kind="memory")
        assert len(res1) >= 1, "应命中第一个事件块"
        assert len(res2) >= 1, "应命中第二个事件块"

        res_other = store.search("lilith", "眼神让我在意", kind="memory")
        assert res_other == [], "他人不可见的 memory 不应返回"

    @pytest.mark.asyncio
    async def test_add_memory_only_targets_date(self, clean_store, tmp_path, monkeypatch):
        """只索引指定日期，其它日期不应被写入。"""
        store = clean_store

        write_memory(
            tmp_path,
            "mitsuki",
            """
# mitsuki 的长期记忆

## 4月3日
- **时间**：4月3日 上午 08:18 - 08:52
- **地点**：教室
- **在场**：桥本美月、玩家（小明）、莉莉丝（转学生）
- **内容**：今天的事让我在意。

## 4月4日
- **时间**：4月4日 上午 09:00 - 09:30
- **地点**：操场
- **在场**：桥本美月
- **内容**：我独自待了一会儿。
""".strip(),
        )

        monkeypatch.setattr(store, "character_path", make_character_path(tmp_path))
        await store.add_memory("mitsuki", "4月3日")

        # 等待索引完成并验证结果
        res_hit = await wait_for_search(store, "mitsuki", "让我在意", kind="memory")
        res_miss = store.search("mitsuki", "独自待了一会儿", kind="memory")
        assert len(res_hit) >= 1, "应命中索引的日期"
        assert not any("独自待了一会儿" in r["content"] for r in res_miss), "未索引日期不应返回"
