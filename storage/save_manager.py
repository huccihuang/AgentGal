"""存档与游戏状态管理"""

import asyncio
import glob
import hashlib
import json
import os
import re
import shutil
import time
import traceback
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

from shared.config import CHARACTERS_DIR, PROJECT_ROOT, character_path, get_agent_names
from log_config.routing import routing_logger
from storage.agent_files import (
    PLAYER_NAME_FILENAME,
    extract_player_name,
    increment_turn_counter,
    read_agent_file,
    read_player_name,
    read_turn_counter,
    write_player_name,
)
from storage.history import append_message, load_conversation_history
from memory.parser import (
    canonical_cn_date,
    extract_status_field,
    parse_jsonl_line,
    serialize_episode,
)

TEMPLATES_DIR = PROJECT_ROOT / "data" / "templates"


# =============================================================================
# 开场白 / 消息加载
# =============================================================================


def load_prompt_file(filename: str) -> str:
    """从 prompts 目录加载文本文件（全局配置，如 opening_intro.txt）"""
    file_path = PROJECT_ROOT / "prompts" / filename
    if file_path.exists():
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[警告] 读取文件失败 {filename}: {e}")
    return ""


def load_story_file(story_id: str, filename: str) -> str:
    """从故事模板目录加载文本文件（如 opening.txt）"""
    file_path = TEMPLATES_DIR / story_id / filename
    if file_path.exists():
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[警告] 读取故事文件失败 {story_id}/{filename}: {e}")
    return ""


# =============================================================================
# 存档检查 / 重置
# =============================================================================


def has_existing_save() -> bool:
    """检查是否有已存在的存档（narrator 有 jsonl 文件）"""
    raw_dir = character_path("narrator", "raw")
    if os.path.exists(raw_dir):
        if glob.glob(f"{raw_dir}/*.jsonl"):
            return True
    return False


def reset_logs():
    """重置日志文件 - 清空 logs 目录下所有 .log 和 .jsonl 文件"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        print(f"  [info] 日志目录不存在: {log_dir}", flush=True)
        return

    for root, _, files in os.walk(log_dir):
        for filename in files:
            if filename.endswith(".log") or filename.endswith(".jsonl"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "w"):
                        pass
                    print(f"  已清空: {filepath}", flush=True)
                except Exception as e:
                    print(f"  清空失败 {filepath}: {e}", flush=True)


async def reset_game(story_id: str = "school") -> tuple[str, str]:
    """重置游戏，从 templates/{story_id} 重新创建 characters 目录

    Args:
        story_id: 故事ID，对应 data/templates/ 下的子目录名

    Returns:
        (opening_intro 文本, opening 文本)
    """
    try:
        print(f"\n{'=' * 40}", flush=True)
        print("重置游戏...", flush=True)
        print(f"{'=' * 40}\n", flush=True)

        # 1. 删除向量记忆
        from storage.vector_store import vector_store

        all_agents = get_agent_names()
        if all_agents:
            print("[Reset] 清理向量记忆...", flush=True)
            await vector_store.delete_all_agents(all_agents)

        # 2. 删除整个 characters 目录（如果存在）
        characters_dir = str(CHARACTERS_DIR)
        if os.path.exists(characters_dir):
            try:
                shutil.rmtree(characters_dir)
                print(f"  已删除: {characters_dir}", flush=True)
            except Exception as e:
                print(f"  删除失败 {characters_dir}: {e}", flush=True)

        # 3. 从对应故事模板完整复制
        story_template_dir = str(TEMPLATES_DIR / story_id)
        if os.path.exists(story_template_dir):
            try:
                shutil.copytree(story_template_dir, characters_dir)
                print(f"  已复制: {story_template_dir} -> {characters_dir}", flush=True)
            except Exception as e:
                print(f"  复制失败: {e}", flush=True)
        else:
            print(f"  [警告] 故事模板目录不存在: {story_template_dir}", flush=True)

        # 4. 只为 narrator 创建 raw 目录（对话历史集中存储）
        raw_dir = character_path("narrator", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        print(f"  已创建: {raw_dir}", flush=True)

        read_player_name.cache_clear()

        # 5. 写入 story_id，并清空当前存档节点。新开局的第一次保存是世界线根节点。
        story_id_path = os.path.join(characters_dir, ".story_id")
        with open(story_id_path, "w", encoding="utf-8") as f:
            f.write(story_id)
        print(f"  已写入: {story_id_path}", flush=True)

        _clear_save_id()
        print("  已清空: .save_id", flush=True)

        # 6. 重置日志
        print("[日志]", flush=True)
        reset_logs()

        print(f"\n{'=' * 40}", flush=True)
        print("重置完成", flush=True)
        print(f"{'=' * 40}\n", flush=True)
    except Exception as e:
        print(f"[ERROR] reset_game 执行异常: {e}", flush=True)
        import traceback

        traceback.print_exc()

    # 加载两个开场白：玩法介绍从 prompts/ 取（全局），故事开场从故事目录取
    intro_text = load_prompt_file("opening_intro.txt")
    opening_text = load_story_file(story_id, "opening.txt")

    # 将故事开场旁白写入 narrator 的历史（玩法介绍不写入）
    visible_agents = get_agent_names(include_narrator=False)
    opening_message = {
        "role": "narrator",
        "content": opening_text,
        "visible_to": visible_agents + ["narrator"],
        "turn": increment_turn_counter(),
    }
    await append_message(opening_message)

    return intro_text, opening_text


# =============================================================================
# 存档列表 / 导入
# =============================================================================


def _clean_meta_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _read_archive_metadata(save_path: Path) -> dict:
    try:
        with zipfile.ZipFile(str(save_path), "r") as zf:
            if "metadata.json" not in zf.namelist():
                return {}
            payload = json.loads(zf.read("metadata.json").decode("utf-8"))
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _story_id_from_metadata(meta: dict, save_path: Path) -> str:
    story_id = _clean_meta_text(meta.get("story_id")) or _clean_meta_text(
        meta.get("theme")
    )
    if story_id:
        return story_id

    stem = save_path.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return "unknown"


def _format_archive_time(value: str) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def _read_archive_turn(meta: dict, zip_file: Path) -> int:
    try:
        if meta.get("turn") is not None:
            return int(meta.get("turn") or 0)
    except (TypeError, ValueError):
        pass

    try:
        with zipfile.ZipFile(str(zip_file), "r") as zf:
            if ".turn_counter.json" in zf.namelist():
                data = json.loads(zf.read(".turn_counter.json").decode("utf-8"))
                return int(data.get("turn", 0))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, zipfile.BadZipFile):
        pass
    return 0


def _save_id_from_metadata(meta: dict, save_path: Path) -> str:
    save_id = _clean_meta_text(meta.get("save_id"))
    if save_id:
        return save_id
    stem = save_path.stem
    return stem.rsplit("_", 1)[-1] if "_" in stem else stem


def _read_save_archive_entry(zip_file: Path) -> dict:
    meta = _read_archive_metadata(zip_file)
    turn = _read_archive_turn(meta, zip_file)

    export_time = _clean_meta_text(meta.get("export_time")) or _clean_meta_text(
        meta.get("created_at")
    )
    display_time = _format_archive_time(export_time) or zip_file.name
    focus = _clean_meta_text(meta.get("focus"))
    title = _clean_meta_text(meta.get("title")) or focus or display_time
    story_id = _story_id_from_metadata(meta, zip_file)
    save_id = _save_id_from_metadata(meta, zip_file)
    parent_save_id = _clean_meta_text(meta.get("parent_save_id"))

    return {
        "filename": zip_file.name,
        "display_time": display_time,
        "display": display_time,
        "created_at": export_time,
        "save_id": save_id,
        "parent_save_id": parent_save_id,
        "story_id": story_id,
        "title": title,
        "focus": focus,
        "turn": turn,
        "version": _clean_meta_text(meta.get("version")),
        "_sort_key": export_time or f"{zip_file.stat().st_mtime:.6f}",
    }


def list_save_archives() -> list[dict]:
    """列出 saves/ 目录下所有存档，按时间倒序

    Returns:
        存档信息列表，每项包含 filename、display_time
    """
    save_dir = PROJECT_ROOT / "saves"
    if not save_dir.exists():
        return []

    saves = []
    for zip_file in save_dir.glob("*.zip"):
        saves.append(_read_save_archive_entry(zip_file))

    saves.sort(key=lambda s: s.get("_sort_key", ""), reverse=True)
    for s in saves:
        s.pop("_sort_key", None)
    return saves


def _save_node_sort_key(node: dict) -> str:
    return _clean_meta_text(node.get("created_at")) or _clean_meta_text(
        node.get("filename")
    )


def _sort_save_nodes(nodes: list[dict]) -> str:
    nodes.sort(key=lambda s: (s.get("created_at") or "", s.get("filename") or ""))
    latest = ""
    for node in nodes:
        child_latest = _sort_save_nodes(node.get("children", []))
        node["child_count"] = len(node.get("children", []))
        node_latest = max(_save_node_sort_key(node), child_latest)
        node["latest_created_at"] = node_latest
        latest = max(latest, node_latest)
    return latest


def _sort_game_roots(nodes: list[dict]) -> None:
    nodes.sort(
        key=lambda node: (
            _clean_meta_text(node.get("latest_created_at")),
            _clean_meta_text(node.get("created_at")),
            _clean_meta_text(node.get("filename")),
        ),
        reverse=True,
    )


def list_save_worldlines(saves: list[dict] | None = None) -> list[dict]:
    """按 story_id 把存档拼成临时世界线树。关系只来自各 zip 的 metadata。"""
    if saves is None:
        saves = list_save_archives()
    worlds: dict[str, list[dict]] = {}
    for save in saves:
        story_id = _clean_meta_text(save.get("story_id")) or "unknown"
        node = {**save, "children": [], "child_count": 0, "parent_missing": False}
        worlds.setdefault(story_id, []).append(node)

    result: list[dict] = []
    for story_id, nodes in worlds.items():
        by_id = {
            _clean_meta_text(node.get("save_id")): node
            for node in nodes
            if _clean_meta_text(node.get("save_id"))
        }
        roots: list[dict] = []
        orphans: list[dict] = []

        for node in nodes:
            parent_save_id = _clean_meta_text(node.get("parent_save_id"))
            parent = by_id.get(parent_save_id) if parent_save_id else None
            if parent and parent is not node:
                parent["children"].append(node)
            elif parent_save_id:
                node["parent_missing"] = True
                orphans.append(node)
            else:
                roots.append(node)

        _sort_save_nodes(roots)
        _sort_save_nodes(orphans)
        _sort_game_roots(roots)
        _sort_game_roots(orphans)
        result.append(
            {
                "story_id": story_id,
                "root_count": len(roots),
                "orphan_count": len(orphans),
                "save_count": len(nodes),
                "roots": roots,
                "orphans": orphans,
            }
        )

    result.sort(key=lambda w: w["story_id"])
    return result


def get_current_save_context() -> dict:
    return {
        "current_save_id": _read_save_id(),
        "current_story_id": _read_story_theme(),
    }


def delete_save_leaf(save_filename: str) -> tuple[str | None, str | None]:
    """只删除没有子分支的单个存档节点。

    返回 (deleted_filename, None) 表示成功；失败时返回 (None, reason)。
    """
    if not _is_safe_save_filename(save_filename):
        routing_logger.warning("[delete_leaf_save] 非法文件名: %s", save_filename)
        return None, "invalid"

    saves = list_save_archives()
    target = next((save for save in saves if save.get("filename") == save_filename), None)
    if not target:
        routing_logger.warning("[delete_leaf_save] 存档不存在: %s", save_filename)
        return None, "not_found"

    target_id = _clean_meta_text(target.get("save_id"))
    target_story = _clean_meta_text(target.get("story_id"))
    has_children = target_id and any(
        _clean_meta_text(s.get("story_id")) == target_story
        and _clean_meta_text(s.get("parent_save_id")) == target_id
        for s in saves
    )
    if has_children:
        routing_logger.warning("[delete_leaf_save] 存档仍有子分支: %s", save_filename)
        return None, "has_children"

    save_path = PROJECT_ROOT / "saves" / save_filename
    try:
        save_path.unlink()
    except FileNotFoundError:
        routing_logger.warning("[delete_leaf_save] 存档文件不存在: %s", save_path)
        return None, "not_found"

    if target_id and _read_save_id() == target_id:
        _clear_save_id()

    routing_logger.info("[delete_leaf_save] 已删除叶子存档: %s", save_filename)
    return save_filename, None


def delete_save_game(root_filename: str) -> list[str] | None:
    """删除以 root_filename 为根的整棵 Game 存档树。

    返回删除的文件名列表；文件名非法或根节点不存在时返回 None。
    """
    if not _is_safe_save_filename(root_filename):
        routing_logger.warning("[delete_game] 非法文件名: %s", root_filename)
        return None

    saves = list_save_archives()
    root = next((save for save in saves if save.get("filename") == root_filename), None)
    if not root:
        routing_logger.warning("[delete_game] 根存档不存在: %s", root_filename)
        return None

    story_id = _clean_meta_text(root.get("story_id"))
    children_by_parent: dict[str, list[dict]] = {}
    for save in saves:
        if _clean_meta_text(save.get("story_id")) != story_id:
            continue
        parent_id = _clean_meta_text(save.get("parent_save_id"))
        if parent_id:
            children_by_parent.setdefault(parent_id, []).append(save)

    to_delete: list[dict] = []
    stack = [root]
    seen_filenames: set[str] = set()
    while stack:
        current = stack.pop()
        filename = _clean_meta_text(current.get("filename"))
        if not filename or filename in seen_filenames:
            continue
        seen_filenames.add(filename)
        to_delete.append(current)
        save_id = _clean_meta_text(current.get("save_id"))
        if save_id:
            stack.extend(children_by_parent.get(save_id, []))

    save_dir = PROJECT_ROOT / "saves"
    deleted: list[str] = []
    deleted_ids = {_clean_meta_text(save.get("save_id")) for save in to_delete}
    for save in to_delete:
        filename = _clean_meta_text(save.get("filename"))
        if not filename:
            continue
        try:
            (save_dir / filename).unlink()
            deleted.append(filename)
        except FileNotFoundError:
            continue

    if _read_save_id() in deleted_ids:
        _clear_save_id()

    routing_logger.info("[delete_game] 已删除 Game: root=%s files=%s", root_filename, deleted)
    return deleted


def _restore_player_name_from_raw_history() -> None:
    """旧存档没有 .player_name 时，从第一条玩家 raw 消息恢复一次。"""
    read_player_name.cache_clear()
    if read_player_name():
        return

    for message in load_conversation_history():
        if message.get("role") == "player" and (
            player_name := extract_player_name(message.get("content") or "")
        ):
            write_player_name(player_name)
            return


async def import_save_archive(save_filename: str) -> bool:
    """从指定存档文件恢复游戏状态

    Args:
        save_filename: 存档文件名（仅文件名，不含路径）

    Returns:
        成功返回 True，失败返回 False
    """
    if not _is_safe_save_filename(save_filename):
        print(f"[读档] 非法文件名: {save_filename}", flush=True)
        return False

    save_path = PROJECT_ROOT / "saves" / save_filename
    if not save_path.exists():
        print(f"[读档] 存档文件不存在: {save_path}", flush=True)
        return False

    started_at = time.perf_counter()
    last_step_at = started_at

    def log_step(label: str) -> None:
        nonlocal last_step_at
        now = time.perf_counter()
        print(
            f"[读档] {label} 用时 {now - last_step_at:.2f}s，累计 {now - started_at:.2f}s",
            flush=True,
        )
        last_step_at = now

    try:
        from storage.vector_store import vector_store

        old_agents = get_agent_names()
        log_step(f"扫描当前角色（{len(old_agents)} 个）")

        archive_meta = _read_archive_metadata(save_path)

        characters_dir = str(CHARACTERS_DIR)
        if os.path.exists(characters_dir):
            shutil.rmtree(characters_dir)
            print(f"[读档] 已删除: {characters_dir}", flush=True)
        log_step("删除旧角色目录")

        os.makedirs(characters_dir, exist_ok=True)
        with zipfile.ZipFile(str(save_path), "r") as zf:
            for member in zf.namelist():
                if member == "metadata.json":
                    continue
                if os.path.basename(member) == "milestones.md":
                    print(f"[读档] 跳过旧里程碑文件: {member}", flush=True)
                    continue
                zf.extract(member, characters_dir)
                print(f"[读档] 已恢复: {member}", flush=True)
        log_step("解压存档")

        read_player_name.cache_clear()
        _restore_player_name_from_raw_history()
        log_step("校验玩家名")

        story_id_path = os.path.join(characters_dir, ".story_id")
        if not os.path.exists(story_id_path):
            story_id = _story_id_from_metadata(archive_meta, save_path)
            with open(story_id_path, "w", encoding="utf-8") as f:
                f.write(story_id)
            print(f"[读档] 旧存档无 story_id，已恢复为: {story_id}", flush=True)

        save_id = _read_save_id()
        if not save_id:
            save_id = _read_archive_save_id(save_path)
            _write_save_id(save_id)
            print(f"[读档] 旧存档无 save_id，已恢复为: {save_id}", flush=True)
        log_step("校验 save_id")

        restored_agents = get_agent_names()
        agents_to_clear = list(dict.fromkeys([*old_agents, *restored_agents]))
        if agents_to_clear:
            print(f"[读档] 清理向量记忆: {agents_to_clear}", flush=True)
            await vector_store.delete_all_agents(agents_to_clear)
        log_step(f"清理向量记忆（{len(agents_to_clear)} 个角色）")

        print("[读档] 并发重建向量库...", flush=True)
        from memory.indexer import rebuild_memory_index, rebuild_understanding_index
        await asyncio.gather(
            rebuild_memory_index(clear_existing=False),
            rebuild_understanding_index(),
        )
        log_step("重建向量库（EpisodeMemory + Understanding）")

        print(f"[读档] 读档完成: {save_filename}", flush=True)
        return True

    except Exception as e:
        print(f"[读档] 读档失败: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# 存档导出
# =============================================================================


def _get_agent_save_files(agent_name: str) -> list[str]:
    """获取指定角色需要存档的所有文件路径"""
    files = []
    base = character_path(agent_name)

    core_files = ["soul.md", "status.md"]
    if agent_name != "narrator":
        core_files.extend([
            "memory.jsonl", "memory_draft.jsonl", "understanding.jsonl",
        ])

    for filename in core_files:
        filepath = f"{base}/{filename}"
        if os.path.exists(filepath):
            files.append(filepath)

    for filename in ["states.md", "events.md"]:
        filepath = f"{base}/{filename}"
        if os.path.exists(filepath):
            files.append(filepath)

    for hidden_file in [".history_window_state.json"]:
        hidden_path = f"{base}/{hidden_file}"
        if os.path.exists(hidden_path):
            files.append(hidden_path)

    if agent_name == "narrator":
        raw_dir = character_path("narrator", "raw")
        if os.path.exists(raw_dir):
            for jsonl_file in glob.glob(f"{raw_dir}/*.jsonl"):
                files.append(jsonl_file)

        world_schedule_path = character_path("narrator", "world_schedule.json")
        if os.path.exists(world_schedule_path):
            files.append(world_schedule_path)

    return files


def _read_save_id() -> str:
    """读取当前游戏来源标记。它不再参与存档文件名选择。"""
    save_id_path = CHARACTERS_DIR / ".save_id"
    try:
        return save_id_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _write_save_id(save_id: str) -> None:
    """更新当前游戏来源标记。"""
    save_id_path = CHARACTERS_DIR / ".save_id"
    save_id_path.write_text(save_id, encoding="utf-8")


def _clear_save_id() -> None:
    save_id_path = CHARACTERS_DIR / ".save_id"
    save_id_path.unlink(missing_ok=True)


def _read_story_theme() -> str:
    """读取 .story_id 标记文件，返回故事主题（如 school / modern）"""
    story_id_path = CHARACTERS_DIR / ".story_id"
    try:
        return story_id_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _read_narrator_focus() -> str:
    """读取叙事焦点写入 metadata.json，供存档列表展示。"""
    try:
        status_text = read_agent_file("narrator", "status.md")
        focus = extract_status_field(status_text, "叙事焦点")
        if focus:
            safe = re.sub(r'[/\\:*?"<>|\n]', "", focus)[:20].strip()
            return safe
    except Exception:
        pass
    return ""


def _is_safe_save_filename(filename: str) -> bool:
    """只允许定位 saves/ 下的单个 zip 文件名。"""
    return (
        bool(filename)
        and filename.endswith(".zip")
        and "/" not in filename
        and "\\" not in filename
    )


def _read_archive_save_id(save_path: Path) -> str:
    """读取旧档位的来源标记，缺失时用文件名兜底。"""
    return _save_id_from_metadata(_read_archive_metadata(save_path), save_path)


def _recall_state_by_content_hash(
    recall_state: dict[str, dict[str, str]],
) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for entry in recall_state.values():
        date = canonical_cn_date(str(entry.get("date", ""))) or str(
            entry.get("date", "")
        ).strip()
        content_hash = str(entry.get("content_hash", "")).strip()
        recalled_at = canonical_cn_date(str(entry.get("last_recalled_at", ""))) or str(
            entry.get("last_recalled_at", "")
        ).strip()
        if date and content_hash and recalled_at:
            result[(date, content_hash)] = recalled_at
    return result


def _memory_jsonl_archive_payload(
    agent_name: str,
    recall_state: dict[str, dict[str, str]],
) -> str | None:
    """返回写入存档的 memory.jsonl 内容，并合并 DB 中最新 recall 状态。"""
    memory_path = Path(character_path(agent_name, "memory.jsonl"))
    if not memory_path.exists():
        return None

    raw_text = memory_path.read_text(encoding="utf-8")
    if not recall_state:
        return raw_text

    recall_by_hash = _recall_state_by_content_hash(recall_state)
    output_lines: list[str] = []

    for line in raw_text.splitlines():
        record = parse_jsonl_line(line)
        if record is None:
            output_lines.append(line)
            continue

        date = canonical_cn_date(record.date) or record.date
        content_hash = hashlib.sha1(record.content.encode("utf-8")).hexdigest()
        recalled_at = recall_by_hash.get((date, content_hash), record.last_recalled_at)
        recalled_at = canonical_cn_date(recalled_at) or recalled_at or date
        output_lines.append(
            serialize_episode(record.model_copy(update={"last_recalled_at": recalled_at}))
        )

    return "\n".join(output_lines) + "\n"


def _build_new_save_path(theme: str, save_dir: Path) -> tuple[str, Path, str]:
    """为新档位生成不冲突的 uuid 文件名。"""
    for _ in range(20):
        save_id = uuid.uuid4().hex[:8]
        filename = f"{theme}_{save_id}.zip" if theme else f"{save_id}.zip"
        save_path = save_dir / filename
        if not save_path.exists():
            return filename, save_path, save_id

    save_id = uuid.uuid4().hex
    filename = f"{theme}_{save_id}.zip" if theme else f"{save_id}.zip"
    return filename, save_dir / filename, save_id


def _compute_content_hash() -> str:
    """计算当前游戏状态的内容哈希，用于存档去重。

    哈希覆盖：回合数、各角色 status.md、各角色 memory_draft.jsonl 末行、
    narrator 原始对话历史末条消息。任一组件变化都会产生不同哈希。
    """
    hasher = hashlib.sha256()

    turn = read_turn_counter()
    hasher.update(str(turn).encode())

    for agent in get_agent_names():
        status_text = read_agent_file(agent, "status.md")
        if status_text:
            hasher.update(status_text.encode())

        if agent == "narrator":
            continue

        draft_path = Path(character_path(agent, "memory_draft.jsonl"))
        if draft_path.exists():
            lines = draft_path.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                hasher.update(lines[-1].encode())

    raw_dir = character_path("narrator", "raw")
    if os.path.exists(raw_dir):
        jsonl_files = sorted(glob.glob(f"{raw_dir}/*.jsonl"))
        if jsonl_files:
            last_file = jsonl_files[-1]
            lines = Path(last_file).read_text(encoding="utf-8").strip().splitlines()
            if lines:
                hasher.update(lines[-1].encode())

    return hasher.hexdigest()


def _find_duplicate_save(parent_save_id: str, content_hash: str) -> str | None:
    """在当前分支上查找内容相同的已有存档。

    两步检查：
    1. 先找直接父节点（save_id == parent_save_id，即上一次存档）— 覆盖"存储两次"场景
    2. 再找兄弟节点（parent_save_id 相同）— 覆盖"读旧档后存储"场景

    只在同一分支内匹配，避免跨分支误判。
    返回匹配的文件名；无匹配返回 None。
    """
    save_dir = PROJECT_ROOT / "saves"
    if not save_dir.exists():
        return None

    parent_key = parent_save_id or None

    for zip_file in save_dir.glob("*.zip"):
        meta = _read_archive_metadata(zip_file)
        if meta.get("save_id") == parent_save_id and meta.get("content_hash") == content_hash:
            return zip_file.name

    for zip_file in save_dir.glob("*.zip"):
        meta = _read_archive_metadata(zip_file)
        meta_parent = meta.get("parent_save_id") or None
        if meta_parent == parent_key and meta.get("content_hash") == content_hash:
            return zip_file.name

    return None


async def export_save_archive_with_detail() -> tuple[str | None, str | None, bool]:
    """导出不可变存档节点，并返回路径或可直接展示的错误详情。

    自动去重：如果当前分支下已存在内容相同的存档，直接返回已有存档路径，
    不创建新节点。

    Returns:
        (save_path, error_detail, is_duplicate)
    """
    theme = _read_story_theme()
    parent_save_id = _read_save_id()
    turn = read_turn_counter()

    content_hash = _compute_content_hash()

    existing = _find_duplicate_save(parent_save_id, content_hash)
    if existing:
        existing_path = PROJECT_ROOT / "saves" / existing
        existing_meta = _read_archive_metadata(existing_path)
        existing_save_id = existing_meta.get("save_id") or existing.rsplit("_", 1)[-1].replace(".zip", "")
        _write_save_id(existing_save_id)
        print(f"[存档] 内容未变化，复用已有存档: {existing}")
        return str(existing_path), None, True

    save_dir = PROJECT_ROOT / "saves"
    os.makedirs(save_dir, exist_ok=True)

    filename, save_path, save_id = _build_new_save_path(theme, save_dir)
    print(f"[存档] 新建世界线节点: {filename}")

    focus = _read_narrator_focus()
    title = focus or (f"第 {turn} 轮" if turn else "新存档")

    all_agents = get_agent_names()
    if not all_agents:
        print("[存档] 没有找到任何角色")
        return None, "没有找到任何角色，无法创建存档。", False

    temp_path = save_dir / f".{filename}.{uuid.uuid4().hex}.tmp"

    try:
        # 运行期 recall 真值在 DB；存档时合并进 zip 内的 memory.jsonl。
        from storage.vector_store import vector_store

        character_agents = [a for a in all_agents if a != "narrator"]
        export_results = await asyncio.gather(
            *(
                vector_store.export_recall_state(agent_name)
                for agent_name in character_agents
            )
        )
        recall_states = dict(zip(character_agents, export_results))

        export_time = datetime.now().isoformat()
        with zipfile.ZipFile(str(temp_path), "w", zipfile.ZIP_DEFLATED) as zf:
            metadata = {
                "export_time": export_time,
                "created_at": export_time,
                "save_id": save_id,
                "parent_save_id": parent_save_id or None,
                "content_hash": content_hash,
                "filename": filename,
                "story_id": theme,
                "turn": turn,
                "title": title,
                "focus": focus,
                "agents": all_agents,
                "version": "2.0",
            }
            zf.writestr(
                "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2)
            )

            zf.writestr(".save_id", save_id)
            print(f"[存档] 已添加: .save_id")

            for marker in [".story_id", ".turn_counter.json", PLAYER_NAME_FILENAME]:
                marker_path = CHARACTERS_DIR / marker
                if marker_path.exists():
                    zf.write(str(marker_path), marker)
                    print(f"[存档] 已添加: {marker}")

            for agent_name in all_agents:
                agent_files = _get_agent_save_files(agent_name)
                for filepath in agent_files:
                    if os.path.exists(filepath):
                        arcname = os.path.relpath(filepath, start=str(CHARACTERS_DIR))
                        if (
                            agent_name != "narrator"
                            and os.path.basename(filepath) == "memory.jsonl"
                        ):
                            payload = _memory_jsonl_archive_payload(
                                agent_name,
                                recall_states.get(agent_name, {}),
                            )
                            if payload is not None:
                                zf.writestr(arcname, payload)
                                print(f"[存档] 已添加: {filepath} (merged recall)")
                                continue
                        zf.write(filepath, arcname)
                        print(f"[存档] 已添加: {filepath}")

        os.replace(temp_path, save_path)
        _write_save_id(save_id)
        print(f"[存档] 导出完成: {save_path}")
        return str(save_path), None, False

    except Exception as e:
        temp_path.unlink(missing_ok=True)
        detail = f"{type(e).__name__}: {e}"
        print(f"[存档] 导出失败: {detail}")
        routing_logger.error("[save] 导出异常: %s\n%s", detail, traceback.format_exc())
        return None, detail, False


async def export_save_archive() -> str | None:
    """兼容旧接口：仅返回存档路径。"""
    save_path, _, _ = await export_save_archive_with_detail()
    return save_path
