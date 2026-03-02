"""memory file/text helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from engine.config import character_path

_EMPTY_PLACEHOLDER = "（暂无）"
_GROWTH_TITLE = "# 心路历程"

STATUS_FIELDS: dict[str, list[str]] = {
    "lilith": ["身份", "心境", "我和他", "在意的事", "打算"],
    "mitsuki": ["心境", "我和他", "在意的事", "打算"],
    "narrator": ["故事阶段", "当前时间", "场景", "叙事焦点", "待触发事件"],
}

USER_FIELDS: dict[str, list[str]] = {
    "lilith": ["基本信息", "观察到的特质", "互动模式"],
    "mitsuki": ["基本信息", "观察到的特质", "互动模式"],
}

_DATE_RE = re.compile(r"(\d{1,2})月(\d{1,2})日")
_EVENT_TIME_RE = re.compile(r"^-\s+\*\*时间\*\*：(\d{1,2}月\d{1,2}日)")


def normalize(content: str) -> str:
    content = content.replace("\\n", "\n")
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
    out: list[str] = []
    for line in content.split("\n"):
        m = re.match(r"^(?:##\s*|\*\*)?(\d{1,2}月\d{1,2}日)(?:\*\*)?\s*$", line.strip())
        out.append(f"## {m.group(1)}" if m else line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))


def split_by_date(content: str) -> OrderedDict[str, str]:
    sections: OrderedDict[str, str] = OrderedDict()
    current_date: str | None = None
    current_lines: list[str] = []

    for line in content.split("\n"):
        m = re.match(r"^##\s*(\d{1,2}月\d{1,2}日)$", line.strip())
        if m:
            if current_date:
                body = "\n".join(current_lines).strip()
                sections[current_date] = (
                    f"{sections[current_date]}\n{body}" if current_date in sections else body
                )
            current_date, current_lines = m.group(1), []
            continue
        if current_date:
            current_lines.append(line)

    if current_date:
        body = "\n".join(current_lines).strip()
        sections[current_date] = f"{sections[current_date]}\n{body}" if current_date in sections else body

    return sections


def split_events_raw(content: str) -> list[tuple[str | None, str]]:
    events: list[tuple[str | None, str]] = []
    current_date: str | None = None
    current_lines: list[str] = []

    for line in content.split("\n"):
        m = _EVENT_TIME_RE.match(line.strip())
        if m:
            if current_lines:
                events.append((current_date, "\n".join(current_lines).strip()))
            current_date, current_lines = m.group(1), [line]
            continue
        if current_date is not None:
            current_lines.append(line)

    if current_lines:
        events.append((current_date, "\n".join(current_lines).strip()))

    return events


def split_into_events(day_content: str) -> list[str]:
    events = [event_text for _, event_text in split_events_raw(day_content) if event_text]
    return events if events else [day_content.strip()]


def extract_game_date(text: str) -> str | None:
    m = re.search(r"\*\*时间\*\*：\s*(\d{1,2}月\d{1,2}日)", text)
    return m.group(1) if m else None


def parse_cn_date(date_text: str) -> tuple[int, int] | None:
    m = _DATE_RE.fullmatch((date_text or "").strip())
    if not m:
        return None
    month, day = int(m.group(1)), int(m.group(2))
    return (month, day) if 1 <= month <= 12 and 1 <= day <= 31 else None


def is_date_before(date_text: str, cutoff_date: str) -> bool:
    left = parse_cn_date(date_text)
    right = parse_cn_date(cutoff_date)
    return bool(left and right and left < right)


def date_key(date_text: str) -> int | None:
    parsed = parse_cn_date(date_text)
    return None if parsed is None else parsed[0] * 100 + parsed[1]


def _get_fields_from_file(file_path: str) -> list[str] | None:
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return [line[3:].strip() for line in f.read().split("\n") if line.startswith("## ")]


def get_allowed_fields(agent_name: str, file_type: str) -> list[str]:
    file_path = character_path(agent_name, f"{file_type}.md")
    fields = _get_fields_from_file(file_path)
    if fields is not None:
        return fields
    defaults = STATUS_FIELDS if file_type == "status" else USER_FIELDS
    return defaults.get(agent_name, [])


def _parse_section_file(file_path: str, allowed_sections: list[str]) -> dict[str, str]:
    content = ""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

    sections: dict[str, str] = {}
    current_sec: str | None = None
    lines: list[str] = []

    for line in content.split("\n"):
        if line.startswith("## "):
            if current_sec is not None:
                sections[current_sec] = "\n".join(lines).strip()
            current_sec, lines = line[3:].strip(), []
            continue
        if current_sec is not None:
            lines.append(line)

    if current_sec is not None:
        sections[current_sec] = "\n".join(lines).strip()

    for field in allowed_sections:
        sections.setdefault(field, _EMPTY_PLACEHOLDER)
    return sections


def _write_section_file(
    file_path: str,
    sections: dict[str, str],
    allowed_sections: list[str],
    title_line: str,
) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    lines = [title_line, ""]
    for sec in allowed_sections:
        if sec in sections:
            lines.extend((f"## {sec}", sections[sec], ""))
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _read_title(file_path: str, default_title: str) -> str:
    if not os.path.exists(file_path):
        return default_title
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                return stripped if stripped.startswith("#") else default_title
    return default_title


def mark_event_triggered(agent_name: str, event_name: str) -> str:
    """将 status.md 中【event_name】对应的未触发事件行直接删除（触发即归档，不留痕迹）。

    Args:
        agent_name: 角色名称
        event_name: 事件名称（不含【】括号，如"破绽初现"）

    Returns:
        操作结果描述
    """
    status_path = character_path(agent_name, "status.md")
    if not os.path.exists(status_path):
        return "status.md 不存在"

    content = Path(status_path).read_text(encoding="utf-8")
    # 匹配整行（含行尾换行），直接删除
    pattern = rf"- \[ \] 【{re.escape(event_name)}】[^\n]*\n?"
    new_content, count = re.subn(pattern, "", content)

    if count == 0:
        return f"未找到事件【{event_name}】"

    Path(status_path).write_text(new_content, encoding="utf-8")
    return f"已触发并移除【{event_name}】"


def add_pending_event(agent_name: str, event_line: str) -> str:
    """在 status.md 的「待触发事件」区块中，于第一个未触发事件前插入新事件。
    插入前检查同名事件是否已存在（按【】内名称匹配），存在则跳过。

    Args:
        agent_name: 角色名称
        event_line: 新事件描述（如"【新事件】描述文字"，函数会自动补 `- [ ] ` 前缀）

    Returns:
        操作结果描述
    """
    status_path = character_path(agent_name, "status.md")
    if not os.path.exists(status_path):
        return "status.md 不存在"

    # 规范化事件行格式
    stripped = event_line.strip()
    if not stripped.startswith("- [ ]"):
        stripped = f"- [ ] {stripped}"

    content = Path(status_path).read_text(encoding="utf-8")

    # 去重：提取新事件名，检查文件中是否已存在同名事件
    name_match = re.search(r"【(.+?)】", stripped)
    if name_match:
        event_name = name_match.group(1)
        if re.search(rf"【{re.escape(event_name)}】", content):
            return f"事件【{event_name}】已存在，跳过"

    lines = content.split("\n")

    # 找到「待触发事件」区块内的第一个 [ ] 行，在其前插入
    in_section = False
    insert_idx = None
    for i, line in enumerate(lines):
        if line.startswith("## 待触发事件"):
            in_section = True
            continue
        if in_section:
            if line.startswith("## "):
                # 进入下一个 section，没找到 [ ]，追加到 section 末尾
                insert_idx = i
                break
            if re.match(r"- \[ \]", line):
                insert_idx = i
                break

    if insert_idx is None:
        lines.append(stripped)
    else:
        lines.insert(insert_idx, stripped)

    Path(status_path).write_text("\n".join(lines), encoding="utf-8")
    return f"已插入新事件: {stripped}"


def _update_section_file(
    file_path: str,
    field: str,
    content: str,
    allowed_sections: list[str],
    title_line: str,
) -> str:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sections = _parse_section_file(file_path, allowed_sections)
    sections[field] = content.replace("\\n", "\n").strip() or _EMPTY_PLACEHOLDER
    _write_section_file(file_path, sections, allowed_sections, title_line)
    return f"已更新 {field}"


def _append_section_file(
    file_path: str,
    field: str,
    content: str,
    allowed_sections: list[str],
    title_line: str,
) -> str:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sections = _parse_section_file(file_path, allowed_sections)
    addition = content.replace("\\n", "\n").strip()
    if not addition:
        return f"{field} 无新增内容"

    existing = sections.get(field, "").strip()
    sections[field] = addition if not existing or existing == _EMPTY_PLACEHOLDER else f"{existing}\n{addition}"
    _write_section_file(file_path, sections, allowed_sections, title_line)
    return f"已追加 {field}"


def read_growth_entries(agent_name: str) -> dict[str, str]:
    path = Path(character_path(agent_name, "growth.md"))
    if not path.exists():
        return {}

    content = path.read_text(encoding="utf-8")
    return {
        m.group(1): m.group(2).strip()
        for m in re.finditer(r"\[(\w+)\]\s*(.+?)(?=\n\[|$)", content, re.DOTALL)
    }


def write_growth_entries(agent_name: str, entries: dict[str, str]) -> None:
    path = Path(character_path(agent_name, "growth.md"))

    def _sort_key(value: str) -> int:
        raw = re.sub(r"[^0-9]", "", value)
        return int(raw) if raw else 0

    lines = [_GROWTH_TITLE, ""]
    for entry_id in sorted(entries.keys(), key=_sort_key):
        lines.extend((f"[{entry_id}] {entries[entry_id]}", ""))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def load_growth_for_prompt(agent_name: str, default: str = "（尚无人格沉淀）") -> str:
    path = Path(character_path(agent_name, "growth.md"))
    if not path.exists():
        return default

    content = path.read_text(encoding="utf-8").strip()
    if not content or content == _GROWTH_TITLE:
        return default

    return "\n".join(line for line in content.split("\n") if line.strip())


def load_text(path: Path) -> str:
    return "" if not path.exists() else path.read_text(encoding="utf-8")


def read_agent_file(agent_name: str, filename: str) -> str:
    return load_text(Path(character_path(agent_name, filename)))


def read_file_tail(file_path: str | Path, lines: int = 10) -> str:
    path = Path(file_path)
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        content_lines = [line for line in f.readlines() if line.strip()]
    return "".join(content_lines[-lines:] if len(content_lines) >= lines else content_lines).strip()


def cleanup_old_backups(bak_dir: Path, pattern: str, max_count: int = 10) -> int:
    bak_files = sorted(bak_dir.glob(pattern), key=lambda f: f.stat().st_mtime)
    deleted = 0
    for old_bak in bak_files[:-max_count] if len(bak_files) > max_count else []:
        old_bak.unlink()
        deleted += 1
    return deleted


def backup_file(src: Path, agent_name: str, prefix: str, max_backups: int = 10) -> Path:
    bak_dir = Path(character_path(agent_name, "bak"))
    bak_dir.mkdir(parents=True, exist_ok=True)
    bak_path = bak_dir / f"{prefix}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_pre.md"
    shutil.copy2(src, bak_path)
    cleanup_old_backups(bak_dir, f"{prefix}_*_pre.md", max_count=max_backups)
    return bak_path


def get_consolidation_state_path(agent_name: str) -> Path:
    return Path(character_path(agent_name, ".consolidation_state.json"))


def load_consolidation_state(agent_name: str) -> Optional[str]:
    p = get_consolidation_state_path(agent_name)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data.get("last_consolidated_date")


def save_consolidation_state(agent_name: str, last_date: str) -> None:
    p = get_consolidation_state_path(agent_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"last_consolidated_date": last_date}, ensure_ascii=False),
        encoding="utf-8",
    )


# ===== 安全写回（带并发保护） =====


def safe_write_memory(path: Path, sections: dict[str, str], agent_name: str, original_content: str) -> int:
    from log_config.routing import routing_logger

    current_content = path.read_text(encoding="utf-8")
    if current_content.startswith(original_content):
        appended = current_content[len(original_content):]
        if appended:
            routing_logger.info(
                "[Memory][FileOps] agent=%s op=safe_write_memory result=keep_appended appended_len=%s",
                agent_name,
                len(appended),
            )
    elif current_content == original_content:
        appended = ""
    else:
        routing_logger.warning(
            "[Memory][FileOps] agent=%s op=safe_write_memory result=conflict_skip",
            agent_name,
        )
        return -1

    parts = [f"# {agent_name} 的长期记忆", ""]
    for date, body in sections.items():
        parts.extend((f"## {date}", body.strip(), ""))
    result = "\n".join(parts).strip() + "\n"
    if appended:
        result += appended
    path.write_text(result, encoding="utf-8")
    return len(result)
