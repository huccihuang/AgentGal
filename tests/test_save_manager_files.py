"""测试 storage.save_manager._get_agent_save_files 覆盖到的文件清单。"""

from __future__ import annotations

import json
import hashlib
import zipfile
from pathlib import Path

import pytest

from storage import agent_files as agent_files_module
from storage import save_manager
from storage.agent_files import PLAYER_NAME_FILENAME


@pytest.fixture
def character_dir(tmp_path: Path, monkeypatch):
    """把 CHARACTERS_DIR 指到临时目录，避免污染仓库数据。"""
    from shared import config as shared_config

    monkeypatch.setattr(shared_config, "CHARACTERS_DIR", tmp_path)
    monkeypatch.setattr(save_manager, "CHARACTERS_DIR", tmp_path)
    return tmp_path


def _seed_character(root: Path, name: str) -> Path:
    agent_dir = root / name
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "soul.md").write_text("# soul\n", encoding="utf-8")
    (agent_dir / "status.md").write_text("## 当前位置\n教室\n", encoding="utf-8")
    (agent_dir / "memory.md").write_text("# memory\n", encoding="utf-8")
    (agent_dir / "understanding.jsonl").write_text(
        '{"id":"u1","content":"理解。"}\n', encoding="utf-8"
    )
    return agent_dir


def test_restore_player_name_from_raw_history(tmp_path: Path, monkeypatch):
    from shared import config as shared_config

    raw_dir = tmp_path / "narrator" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "2026-05-05.jsonl").write_text(
        '{"role":"narrator","content":"你叫什么？","turn":1}\n'
        '{"role":"player","content":"我叫北原悠，是个男生","turn":1}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(shared_config, "CHARACTERS_DIR", tmp_path)
    monkeypatch.setattr(agent_files_module, "CHARACTERS_DIR", tmp_path)

    save_manager._restore_player_name_from_raw_history()

    assert (tmp_path / PLAYER_NAME_FILENAME).read_text(encoding="utf-8") == "北原悠"


def test_character_save_files_omit_legacy_recall_sidecar(character_dir: Path):
    agent_dir = _seed_character(character_dir, "mitsuki")
    (agent_dir / ".memory_recall_state.json").write_text("{}", encoding="utf-8")

    files = save_manager._get_agent_save_files("mitsuki")
    basenames = {Path(f).name for f in files}

    assert ".memory_recall_state.json" not in basenames


def test_narrator_save_files_include_world_schedule_when_present(character_dir: Path):
    narrator_dir = character_dir / "narrator"
    narrator_dir.mkdir(parents=True)
    (narrator_dir / "soul.md").write_text("# narrator\n", encoding="utf-8")
    (narrator_dir / "status.md").write_text("## 当前时间\n4月5日\n", encoding="utf-8")
    (narrator_dir / "world_schedule.json").write_text(
        '{"events":[{"name":"体育祭报名","status":"pending"}]}',
        encoding="utf-8",
    )

    files = save_manager._get_agent_save_files("narrator")
    basenames = {Path(f).name for f in files}

    assert "world_schedule.json" in basenames
    assert "soul.md" in basenames


def test_narrator_save_files_omit_world_schedule_when_missing(character_dir: Path):
    narrator_dir = character_dir / "narrator"
    narrator_dir.mkdir(parents=True)
    (narrator_dir / "soul.md").write_text("# narrator\n", encoding="utf-8")
    (narrator_dir / "status.md").write_text("## 当前时间\n4月5日\n", encoding="utf-8")

    files = save_manager._get_agent_save_files("narrator")
    basenames = {Path(f).name for f in files}

    assert "world_schedule.json" not in basenames
    assert "soul.md" in basenames


def test_narrator_does_not_include_character_only_sidecars(character_dir: Path):
    narrator_dir = character_dir / "narrator"
    narrator_dir.mkdir(parents=True, exist_ok=True)
    (narrator_dir / "soul.md").write_text("# narrator\n", encoding="utf-8")
    (narrator_dir / "status.md").write_text("## 当前时间\n4月5日\n", encoding="utf-8")
    # 即便误放了这些文件，narrator 也不应把它们当作存档内容
    (narrator_dir / "schedule.json").write_text("{}", encoding="utf-8")
    (narrator_dir / ".memory_recall_state.json").write_text("{}", encoding="utf-8")
    (narrator_dir / ".world_event_state.json").write_text("{}", encoding="utf-8")

    files = save_manager._get_agent_save_files("narrator")
    basenames = {Path(f).name for f in files}

    assert "schedule.json" not in basenames
    assert ".memory_recall_state.json" not in basenames
    assert ".world_event_state.json" not in basenames
    assert "soul.md" in basenames


def test_memory_jsonl_archive_payload_merges_db_recall_state(
    character_dir: Path,
    monkeypatch,
):
    from memory.parser import EpisodeMemory, parse_jsonl_line, serialize_episode

    def _character_path(agent_name: str, *parts: str) -> str:
        return str(character_dir / agent_name / Path(*parts))

    monkeypatch.setattr(save_manager, "character_path", _character_path)

    agent_dir = character_dir / "mitsuki"
    agent_dir.mkdir(parents=True, exist_ok=True)
    episode = EpisodeMemory(
        date="4月3日",
        content="需要合并最新召回日期的记忆。",
        memory_owner="mitsuki",
    )
    (agent_dir / "memory.jsonl").write_text(
        serialize_episode(episode) + "\n",
        encoding="utf-8",
    )
    content_hash = hashlib.sha1(episode.content.encode("utf-8")).hexdigest()

    payload = save_manager._memory_jsonl_archive_payload(
        "mitsuki",
        {
            "1": {
                "date": "4月3日",
                "content_hash": content_hash,
                "last_recalled_at": "4月8日",
            }
        },
    )

    assert payload is not None
    archived = parse_jsonl_line(payload)
    assert archived is not None
    assert archived.last_recalled_at == "4月8日"
    original = parse_jsonl_line((agent_dir / "memory.jsonl").read_text(encoding="utf-8"))
    assert original is not None
    assert original.last_recalled_at == "4月3日"


@pytest.mark.asyncio
async def test_export_new_save_uses_fresh_slot_filename(tmp_path: Path, monkeypatch):
    characters_dir = tmp_path / "data" / "runtime" / "characters"
    narrator_dir = characters_dir / "narrator"
    narrator_dir.mkdir(parents=True)
    (characters_dir / ".story_id").write_text("school", encoding="utf-8")
    (characters_dir / ".turn_counter.json").write_text('{"turn": 3}', encoding="utf-8")
    (characters_dir / PLAYER_NAME_FILENAME).write_text("北原悠", encoding="utf-8")
    (narrator_dir / "soul.md").write_text("# narrator\n", encoding="utf-8")
    (narrator_dir / "status.md").write_text("## 叙事焦点\n屋顶\n", encoding="utf-8")

    monkeypatch.setattr(save_manager, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(save_manager, "CHARACTERS_DIR", characters_dir)
    monkeypatch.setattr(agent_files_module, "CHARACTERS_DIR", characters_dir)
    monkeypatch.setattr(save_manager, "get_agent_names", lambda: ["narrator"])
    monkeypatch.setattr(
        save_manager,
        "character_path",
        lambda agent_name, *parts: str(characters_dir / agent_name / Path(*parts)),
    )
    monkeypatch.setattr(save_manager, "_read_narrator_focus", lambda: "屋顶")

    save_path, error, is_duplicate = await save_manager.export_save_archive_with_detail()

    assert error is None
    assert save_path is not None
    assert not is_duplicate
    archive_path = Path(save_path)
    assert archive_path.name.startswith("school_")
    assert archive_path.name.endswith(".zip")

    with zipfile.ZipFile(archive_path) as zf:
        metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
        assert metadata["filename"] == archive_path.name
        assert metadata["save_id"] == archive_path.stem.rsplit("_", 1)[-1]
        assert metadata["parent_save_id"] is None
        assert metadata["story_id"] == "school"
        assert metadata["turn"] == 3
        assert metadata["title"] == "屋顶"
        assert zf.read(".save_id").decode("utf-8") == metadata["save_id"]
        assert zf.read(PLAYER_NAME_FILENAME).decode("utf-8") == "北原悠"

    assert (characters_dir / ".save_id").read_text(encoding="utf-8") == metadata[
        "save_id"
    ]

    second_path, second_error, second_duplicate = await save_manager.export_save_archive_with_detail()

    assert second_error is None
    assert second_path is not None
    assert second_duplicate
    # 去重后返回的是已有存档，不会再创建新文件
    assert second_path == save_path

def test_list_save_worldlines_groups_by_story_and_parent(tmp_path: Path, monkeypatch):
    save_dir = tmp_path / "saves"
    save_dir.mkdir()

    def write_archive(filename: str, metadata: dict) -> None:
        with zipfile.ZipFile(save_dir / filename, "w") as zf:
            zf.writestr(
                "metadata.json",
                json.dumps(metadata, ensure_ascii=False),
            )

    write_archive(
        "school_a.zip",
        {
            "save_id": "a",
            "story_id": "school",
            "created_at": "2026-05-09T10:00:00",
            "title": "根节点",
        },
    )
    write_archive(
        "school_b.zip",
        {
            "save_id": "b",
            "parent_save_id": "a",
            "story_id": "school",
            "created_at": "2026-05-09T10:05:00",
            "title": "同一分支",
        },
    )
    write_archive(
        "school_c.zip",
        {
            "save_id": "c",
            "parent_save_id": "missing",
            "story_id": "school",
            "created_at": "2026-05-09T10:07:00",
            "title": "父节点缺失",
        },
    )
    write_archive(
        "school_d.zip",
        {
            "save_id": "d",
            "story_id": "school",
            "created_at": "2026-05-09T09:00:00",
            "title": "较早根节点",
        },
    )
    write_archive(
        "school_e.zip",
        {
            "save_id": "e",
            "parent_save_id": "d",
            "story_id": "school",
            "created_at": "2026-05-09T10:10:00",
            "title": "最新子节点",
        },
    )
    write_archive(
        "modern_x.zip",
        {
            "save_id": "x",
            "story_id": "modern",
            "created_at": "2026-05-09T11:00:00",
            "title": "现代根节点",
        },
    )

    monkeypatch.setattr(save_manager, "PROJECT_ROOT", tmp_path)

    worlds = save_manager.list_save_worldlines()
    by_story = {world["story_id"]: world for world in worlds}

    assert set(by_story) == {"modern", "school"}
    assert by_story["school"]["save_count"] == 5
    assert by_story["school"]["root_count"] == 2
    assert by_story["school"]["orphan_count"] == 1
    assert by_story["school"]["roots"][0]["save_id"] == "d"
    assert by_story["school"]["roots"][0]["latest_created_at"] == "2026-05-09T10:10:00"
    assert by_story["school"]["roots"][0]["children"][0]["save_id"] == "e"
    assert by_story["school"]["roots"][1]["save_id"] == "a"
    assert by_story["school"]["roots"][1]["children"][0]["save_id"] == "b"
    assert by_story["school"]["orphans"][0]["save_id"] == "c"
    assert by_story["school"]["orphans"][0]["parent_missing"] is True
    assert by_story["modern"]["roots"][0]["save_id"] == "x"


def test_delete_save_game_deletes_root_descendants_and_clears_current(
    tmp_path: Path,
    monkeypatch,
):
    save_dir = tmp_path / "saves"
    save_dir.mkdir()
    characters_dir = tmp_path / "data" / "runtime" / "characters"
    characters_dir.mkdir(parents=True)
    (characters_dir / ".save_id").write_text("b", encoding="utf-8")

    def write_archive(filename: str, metadata: dict) -> None:
        with zipfile.ZipFile(save_dir / filename, "w") as zf:
            zf.writestr(
                "metadata.json",
                json.dumps(metadata, ensure_ascii=False),
            )

    write_archive(
        "school_a.zip",
        {"save_id": "a", "story_id": "school", "created_at": "2026-05-09T10:00:00"},
    )
    write_archive(
        "school_b.zip",
        {
            "save_id": "b",
            "parent_save_id": "a",
            "story_id": "school",
            "created_at": "2026-05-09T10:05:00",
        },
    )
    write_archive(
        "school_c.zip",
        {"save_id": "c", "story_id": "school", "created_at": "2026-05-09T11:00:00"},
    )

    monkeypatch.setattr(save_manager, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(save_manager, "CHARACTERS_DIR", characters_dir)

    deleted = save_manager.delete_save_game("school_a.zip")

    assert set(deleted or []) == {"school_a.zip", "school_b.zip"}
    assert not (save_dir / "school_a.zip").exists()
    assert not (save_dir / "school_b.zip").exists()
    assert (save_dir / "school_c.zip").exists()
    assert not (characters_dir / ".save_id").exists()


def test_delete_save_leaf_only_allows_non_parent_nodes(
    tmp_path: Path,
    monkeypatch,
):
    save_dir = tmp_path / "saves"
    save_dir.mkdir()
    characters_dir = tmp_path / "data" / "runtime" / "characters"
    characters_dir.mkdir(parents=True)
    (characters_dir / ".save_id").write_text("b", encoding="utf-8")

    def write_archive(filename: str, metadata: dict) -> None:
        with zipfile.ZipFile(save_dir / filename, "w") as zf:
            zf.writestr(
                "metadata.json",
                json.dumps(metadata, ensure_ascii=False),
            )

    write_archive(
        "school_a.zip",
        {"save_id": "a", "story_id": "school", "created_at": "2026-05-09T10:00:00"},
    )
    write_archive(
        "school_b.zip",
        {
            "save_id": "b",
            "parent_save_id": "a",
            "story_id": "school",
            "created_at": "2026-05-09T10:05:00",
        },
    )

    monkeypatch.setattr(save_manager, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(save_manager, "CHARACTERS_DIR", characters_dir)

    deleted, reason = save_manager.delete_save_leaf("school_a.zip")

    assert deleted is None
    assert reason == "has_children"
    assert (save_dir / "school_a.zip").exists()

    deleted, reason = save_manager.delete_save_leaf("school_b.zip")

    assert deleted == "school_b.zip"
    assert reason is None
    assert not (save_dir / "school_b.zip").exists()
    assert not (characters_dir / ".save_id").exists()
