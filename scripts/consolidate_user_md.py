"""一次性脚本：整理指定角色的 user.md"""

import asyncio
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from memory.consolidator import MemoryConsolidator, _cleanup_old_backups, build_fields_definition


async def main(agent: str) -> None:
    p = Path(f"data/characters/{agent}/user.md")
    content = p.read_text(encoding="utf-8")
    print(f"原始: {len(content)} 字")

    prompt_tpl = Path("prompts/player_profile_consolidation_prompt.txt").read_text()
    fields_def = build_fields_definition(agent)
    prompt = prompt_tpl.format(fields_definition=fields_def, content=content)

    consolidator = MemoryConsolidator()
    print("调用 LLM 中...")
    result = await consolidator._call_llm(prompt)

    m = re.search(r"^.*第二步.*档案.*$", result, re.MULTILINE)
    if m:
        extracted = result[m.end() :].lstrip("\n").strip()
    else:
        extracted = result.strip()

    print(
        f"整理后: {len(extracted)} 字 (压缩 {(1 - len(extracted) / len(content)) * 100:.1f}%)"
    )
    print()
    print(extracted)
    print()

    confirm = input("写入文件? [y/N] ")
    if confirm.strip().lower() == "y":
        bak_dir = Path(f"data/characters/{agent}/bak")
        bak_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        shutil.copy2(p, bak_dir / f"user_{ts}_pre.md")
        # 只保留最近 10 个备份
        _cleanup_old_backups(bak_dir, "user_*_pre.md", max_count=10)
        p.write_text(extracted + "\n", encoding="utf-8")
        print("✅ 已备份并写入")
    else:
        print("已取消")


if __name__ == "__main__":
    agent = sys.argv[1] if len(sys.argv) > 1 else "mitsuki"
    asyncio.run(main(agent))
