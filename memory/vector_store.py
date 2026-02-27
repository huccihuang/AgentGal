"""sqlite-vec backed memory storage."""

from __future__ import annotations

import array
import asyncio
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import httpx

from engine.config import PROJECT_ROOT, character_path, get_agent_names
from log_config.routing import routing_logger
from memory.file_ops import (
    date_key,
    extract_game_date,
    is_date_before,
    load_consolidation_state,
    normalize,
    parse_cn_date,
    split_by_date,
    split_into_events,
)

DB_PATH = str(PROJECT_ROOT / "data" / "vectors.sqlite")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL_ID") or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
EMBED_API_KEY = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY", "")
EMBED_API_URL = os.getenv("EMBEDDING_API_URL") or os.getenv("LLM_API_URL", "")
EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_vec_blob(vec: list[float]) -> bytes:
    return array.array("f", [float(x) for x in vec]).tobytes()


def _validate_embed_config() -> None:
    if not EMBED_API_KEY:
        raise ValueError("EMBEDDING_API_KEY 或 LLM_API_KEY 未配置，无法计算向量")
    if not EMBED_API_URL:
        raise ValueError("EMBEDDING_API_URL 或 LLM_API_URL 未配置，无法计算向量")


async def _embed_async(texts: list[str]) -> list[list[float]]:
    _validate_embed_config()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            EMBED_API_URL,
            headers={"Authorization": f"Bearer {EMBED_API_KEY}", "Content-Type": "application/json"},
            json={"model": EMBED_MODEL, "input": texts},
        )
        resp.raise_for_status()
        return [d["embedding"] for d in resp.json()["data"]]


def _embed_sync(texts: list[str]) -> list[list[float]]:
    _validate_embed_config()
    resp = httpx.post(
        EMBED_API_URL,
        headers={"Authorization": f"Bearer {EMBED_API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBED_MODEL, "input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    return [d["embedding"] for d in resp.json()["data"]]


class VectorStore:
    def __init__(self):
        self._db: aiosqlite.Connection | None = None
        self._conv_game_date: dict[str, str] = {}
        self._memory_index_cutoff: dict[str, str] = {}
        self._background_tasks: set[asyncio.Task] = set()
        self._memory_index_tasks: set[asyncio.Task] = set()
        self._write_lock: asyncio.Lock | None = None
        self._write_lock_loop: asyncio.AbstractEventLoop | None = None
        self.character_path = character_path

    @staticmethod
    def _log(level: str, message: str, **ctx: Any) -> None:
        text = f"[Memory][VectorStore] {message}"
        if ctx:
            text += " " + " ".join(f"{k}={v}" for k, v in ctx.items())
        if level == "error":
            routing_logger.error(text)
        elif level == "warning":
            routing_logger.warning(text)
        else:
            routing_logger.info(text)

    async def _rollback(self, db: aiosqlite.Connection | None, **ctx: Any) -> None:
        if db is None:
            return
        try:
            await db.execute("ROLLBACK")
            self._log("warning", "rollback", result="ok", **ctx)
        except Exception:
            self._log("warning", "rollback", result="ignored", **ctx)

    def _get_write_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._write_lock is None or self._write_lock_loop is not loop:
            self._write_lock = asyncio.Lock()
            self._write_lock_loop = loop
        return self._write_lock

    async def _load_sqlite_vec(self, conn: aiosqlite.Connection):
        try:
            import sqlite_vec  # type: ignore

            await conn.enable_load_extension(True)
            await conn.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')")
        except Exception as e:
            raise RuntimeError(f"加载 sqlite-vec 扩展失败，请安装 sqlite_vec: {e}")

    @staticmethod
    def _load_sqlite_vec_sync(conn: sqlite3.Connection):
        try:
            import sqlite_vec  # type: ignore

            conn.enable_load_extension(True)
            conn.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')")
        except Exception as e:
            raise RuntimeError(f"加载 sqlite-vec 扩展失败: {e}")

    async def _get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            self._db = await aiosqlite.connect(DB_PATH)
            await self._db.execute("PRAGMA journal_mode=WAL;")
            await self._load_sqlite_vec(self._db)
        return self._db

    async def init_tables(self):
        db = await self._get_db()
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id TEXT NOT NULL UNIQUE,
                date TEXT,
                created_at TEXT,
                visible_to TEXT,
                content TEXT NOT NULL
            )
            """
        )
        await db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_round_id ON chunks(round_id)")
        cols = {row[1] for row in await (await db.execute("PRAGMA table_info(chunks)")).fetchall()}
        if "source" not in cols:
            await db.execute("ALTER TABLE chunks ADD COLUMN source TEXT NOT NULL DEFAULT 'round'")
        if "owner_agent" not in cols:
            await db.execute("ALTER TABLE chunks ADD COLUMN owner_agent TEXT")
        await db.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding F32[{EMBED_DIM}]
            )
            """
        )
        await db.commit()

    async def _upsert_vec_chunk(self, db: aiosqlite.Connection, rowid: int, embedding: list[float]) -> None:
        blob = _to_vec_blob(embedding)
        row = await (await db.execute("SELECT 1 FROM vec_chunks WHERE rowid = ?", (rowid,))).fetchone()
        if row:
            await db.execute("UPDATE vec_chunks SET embedding = ? WHERE rowid = ?", (blob, rowid))
        else:
            await db.execute("INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)", (rowid, blob))

    def add(
        self,
        visible_to: list[str],
        round_id: str,
        content: str,
        kind: str = "round",
        game_date: str | None = None,
    ):
        if kind in ("round", "dialogue"):
            self.add_round(visible_to, round_id, content, game_date=game_date)
            return
        raise ValueError(f"不支持的 kind: {kind}")

    def add_round(self, visible_to: list[str], round_id: str, content: str, game_date: str | None = None):
        task = asyncio.create_task(self._do_add_round(visible_to, round_id, content, game_date))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _do_add_round(
        self,
        visible_to: list[str],
        round_id: str,
        content: str,
        game_date: str | None = None,
    ):
        conv_id = round_id.rsplit("_", 1)[0]
        prev_game_date = self._conv_game_date.get(conv_id)
        visible = list(dict.fromkeys(visible_to))

        if game_date is None:
            game_date = extract_game_date(content)
        if game_date:
            self._conv_game_date[conv_id] = game_date
        game_date = game_date or self._conv_game_date.get(conv_id, "")

        try:
            embedding = (await _embed_async([content]))[0]
        except Exception as e:
            self._log("error", "embed_failed", op="add_round", round_id=round_id, error=e)
            return

        db: aiosqlite.Connection | None = None
        try:
            async with self._get_write_lock():
                await self.init_tables()
                db = await self._get_db()
                await db.execute("BEGIN")

                now_iso = _utcnow_iso()
                visible_json = json.dumps(visible, ensure_ascii=False)
                row = await (await db.execute("SELECT id FROM chunks WHERE round_id = ?", (round_id,))).fetchone()

                if row:
                    rowid = int(row[0])
                    await db.execute(
                        "UPDATE chunks SET date = ?, created_at = ?, visible_to = ?, content = ?, "
                        "source = 'round', owner_agent = NULL WHERE id = ?",
                        (game_date, now_iso, visible_json, content, rowid),
                    )
                else:
                    cur = await db.execute(
                        "INSERT INTO chunks(round_id, date, created_at, visible_to, content, source, owner_agent) "
                        "VALUES (?, ?, ?, ?, ?, 'round', NULL)",
                        (round_id, game_date, now_iso, visible_json, content),
                    )
                    rowid = int(cur.lastrowid or 0)

                if rowid:
                    await self._upsert_vec_chunk(db, rowid, embedding)
                await db.commit()

            self._log("info", "add_round", op="add_round", round_id=round_id, result="ok")
            if prev_game_date and game_date and game_date != prev_game_date:
                self._trigger_memory_indexing(visible, game_date)
        except Exception as e:
            await self._rollback(db, op="add_round", round_id=round_id)
            self._log("error", "add_round", op="add_round", round_id=round_id, result="failed", error=e)

    def _trigger_memory_indexing(self, visible_to: list[str], game_date: str):
        for agent in list(dict.fromkeys(visible_to)):
            self._log("info", "trigger_index", op="index_memory_before_date", agent=agent, game_date=game_date)
            task = asyncio.create_task(self._index_memory_before_date(agent, game_date))
            self._memory_index_tasks.add(task)
            task.add_done_callback(self._memory_index_tasks.discard)

    async def _index_memory_before_date(self, agent_name: str, fallback_cutoff: str):
        cutoff = load_consolidation_state(agent_name) or fallback_cutoff
        if not parse_cn_date(cutoff):
            self._log(
                "info",
                "index_skip",
                op="index_memory_before_date",
                agent=agent_name,
                reason="invalid_cutoff",
                cutoff=cutoff,
            )
            return
        if self._memory_index_cutoff.get(agent_name) == cutoff:
            self._log(
                "info",
                "index_skip",
                op="index_memory_before_date",
                agent=agent_name,
                reason="same_cutoff",
                cutoff=cutoff,
            )
            return

        path = Path(self.character_path(agent_name, "memory.md"))
        if not path.exists():
            alt = Path(self.character_path(agent_name, "Memory.md"))
            if not alt.exists():
                self._log(
                    "info",
                    "index_skip",
                    op="index_memory_before_date",
                    agent=agent_name,
                    reason="memory_not_found",
                )
                return
            path = alt

        payloads: list[tuple[str, str, str]] = []
        for date, body in split_by_date(normalize(path.read_text(encoding="utf-8"))).items():
            if not is_date_before(date, cutoff):
                continue
            for idx, event in enumerate(split_into_events(body), start=1):
                text = event.strip()
                if text:
                    payloads.append((f"memory::{agent_name}::{date}::{idx}", date, text))

        db: aiosqlite.Connection | None = None
        try:
            async with self._get_write_lock():
                await self.init_tables()
                db = await self._get_db()
                await db.execute("BEGIN")
                await db.execute(
                    "DELETE FROM vec_chunks WHERE rowid IN (SELECT id FROM chunks WHERE source = 'memory' AND owner_agent = ?)",
                    (agent_name,),
                )
                await db.execute("DELETE FROM chunks WHERE source = 'memory' AND owner_agent = ?", (agent_name,))

                if payloads:
                    embeddings = await _embed_async([item[2] for item in payloads])
                    visible_json = json.dumps([agent_name], ensure_ascii=False)
                    now_iso = _utcnow_iso()
                    for i, (chunk_id, date, text) in enumerate(payloads):
                        row = await (await db.execute("SELECT id FROM chunks WHERE round_id = ?", (chunk_id,))).fetchone()
                        if row:
                            rowid = int(row[0])
                            await db.execute(
                                "UPDATE chunks SET date = ?, created_at = ?, visible_to = ?, content = ?, "
                                "source = 'memory', owner_agent = ? WHERE id = ?",
                                (date, now_iso, visible_json, text, agent_name, rowid),
                            )
                        else:
                            cur = await db.execute(
                                "INSERT INTO chunks(round_id, date, created_at, visible_to, content, source, owner_agent) "
                                "VALUES (?, ?, ?, ?, ?, 'memory', ?)",
                                (chunk_id, date, now_iso, visible_json, text, agent_name),
                            )
                            rowid = int(cur.lastrowid or 0)
                        if rowid:
                            await self._upsert_vec_chunk(db, rowid, embeddings[i])

                await db.commit()
                self._memory_index_cutoff[agent_name] = cutoff
                self._log("info", "index_done", op="index_memory_before_date", agent=agent_name, cutoff=cutoff, events=len(payloads))
        except Exception as e:
            await self._rollback(db, op="index_memory_before_date", agent=agent_name, cutoff=cutoff)
            self._log("error", "index_failed", op="index_memory_before_date", agent=agent_name, cutoff=cutoff, error=e)

    async def add_memory(self, agent_name: str, date: str):
        if not parse_cn_date(date):
            self._log("warning", "add_memory_skip", op="add_memory", agent=agent_name, date=date, reason="invalid_date")
            return

        path = Path(self.character_path(agent_name, "memory.md"))
        if not path.exists():
            alt = Path(self.character_path(agent_name, "Memory.md"))
            if not alt.exists():
                self._log("info", "add_memory_skip", op="add_memory", agent=agent_name, date=date, reason="memory_not_found")
                return
            path = alt

        body = split_by_date(normalize(path.read_text(encoding="utf-8"))).get(date, "")
        payloads = [
            (f"memory::{agent_name}::{date}::{idx}", event.strip())
            for idx, event in enumerate(split_into_events(body), start=1)
            if event.strip()
        ]
        if not payloads:
            self._log("info", "add_memory_skip", op="add_memory", agent=agent_name, date=date, reason="no_events")
            return

        db: aiosqlite.Connection | None = None
        try:
            async with self._get_write_lock():
                await self.init_tables()
                db = await self._get_db()
                await db.execute("BEGIN")
                await db.execute(
                    "DELETE FROM vec_chunks WHERE rowid IN (SELECT id FROM chunks WHERE source = 'memory' AND owner_agent = ? AND date = ?)",
                    (agent_name, date),
                )
                await db.execute(
                    "DELETE FROM chunks WHERE source = 'memory' AND owner_agent = ? AND date = ?",
                    (agent_name, date),
                )

                embeddings = await _embed_async([item[1] for item in payloads])
                visible_json = json.dumps([agent_name], ensure_ascii=False)
                now_iso = _utcnow_iso()
                for i, (chunk_id, text) in enumerate(payloads):
                    cur = await db.execute(
                        "INSERT INTO chunks(round_id, date, created_at, visible_to, content, source, owner_agent) "
                        "VALUES (?, ?, ?, ?, ?, 'memory', ?)",
                        (chunk_id, date, now_iso, visible_json, text, agent_name),
                    )
                    rowid = int(cur.lastrowid or 0)
                    if rowid:
                        await self._upsert_vec_chunk(db, rowid, embeddings[i])
                await db.commit()

            self._log("info", "add_memory", op="add_memory", agent=agent_name, date=date, events=len(payloads), result="ok")
        except Exception as e:
            await self._rollback(db, op="add_memory", agent=agent_name, date=date)
            self._log("error", "add_memory_failed", op="add_memory", agent=agent_name, date=date, error=e)

    async def rebuild(self, agent_name: str):
        import glob

        _ = agent_name
        await self.init_tables()
        db = await self._get_db()
        await db.execute("DELETE FROM vec_chunks")
        await db.execute("DELETE FROM chunks")
        await db.commit()

        files = sorted(glob.glob(str(Path(self.character_path("narrator", "raw")) / "*.jsonl")))
        if not files:
            self._log("info", "rebuild_skip", op="rebuild", reason="no_raw_files")
            return

        def iter_rounds():
            for fp in files:
                with open(fp, "r", encoding="utf-8") as f:
                    cur: list[dict] = []
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        if obj.get("role") == "player" and cur:
                            yield cur
                            cur = [obj]
                        else:
                            cur.append(obj)
                    if cur:
                        yield cur

        def format_round(msgs: list[dict]) -> tuple[str, list[str], str | None]:
            parts = []
            vis_set: set[str] = set()
            vis: list[str] = []
            game_date: str | None = None
            for m in msgs:
                role = m.get("role", "unknown")
                text = m.get("content", "")
                parts.append(f"{'玩家' if role == 'player' else '旁白' if role == 'narrator' else role}: {text}")

                v = m.get("visible_to", [])
                if isinstance(v, str):
                    try:
                        v = json.loads(v)
                    except Exception:
                        v = []
                for x in v or []:
                    x = str(x)
                    if x not in vis_set:
                        vis_set.add(x)
                        vis.append(x)
                if role == "narrator" and not game_date:
                    game_date = extract_game_date(str(text))

            if "narrator" not in vis_set:
                vis.append("narrator")
            return "\n".join(parts), vis, game_date

        counter = 0
        for msgs in iter_rounds():
            counter += 1
            content, vis, game_date = format_round(msgs)
            await self._do_add_round(vis, f"rebuild_{counter}", content, game_date)

        self._log("info", "rebuild", op="rebuild", rounds=counter, result="ok")

    def search(
        self,
        agent_name: str,
        query: str,
        limit: int | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []
        if not isinstance(limit, int) or limit <= 0:
            try:
                limit = int(os.getenv("VECTOR_SEARCH_LIMIT", "5"))
            except ValueError:
                limit = 5

        try:
            qvec = _embed_sync([query])[0]
        except Exception as e:
            self._log("error", "search_embed_failed", op="search", agent=agent_name, error=e)
            return []

        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(DB_PATH)
            self._load_sqlite_vec_sync(conn)

            if kind == "round":
                scope_sql = (
                    "SELECT id FROM chunks WHERE source = 'round' "
                    "AND EXISTS (SELECT 1 FROM json_each(chunks.visible_to) WHERE json_each.value = ?)"
                )
                scope_params: tuple[Any, ...] = (agent_name,)
            elif kind == "memory":
                scope_sql = "SELECT id FROM chunks WHERE source = 'memory' AND owner_agent = ?"
                scope_params = (agent_name,)
            else:
                cutoff = load_consolidation_state(agent_name)
                cutoff_valid = bool(parse_cn_date(cutoff or ""))
                cutoff_value = date_key(cutoff or "") or -1
                scope_sql = """
                SELECT id FROM chunks
                WHERE (
                  source = 'round'
                  AND EXISTS (SELECT 1 FROM json_each(chunks.visible_to) WHERE json_each.value = ?)
                ) OR (
                  source = 'memory'
                  AND owner_agent = ?
                  AND ? = 1
                  AND instr(date, '月') > 1
                  AND instr(date, '日') > instr(date, '月')
                  AND (
                    CAST(substr(date, 1, instr(date, '月') - 1) AS INTEGER) * 100
                    + CAST(substr(date, instr(date, '月') + 1, instr(date, '日') - instr(date, '月') - 1) AS INTEGER)
                  ) < ?
                )
                """
                scope_params = (agent_name, agent_name, 1 if cutoff_valid else 0, cutoff_value)

            candidate_limit = max(limit * 10, 50)
            rows = conn.execute(
                f"""
                WITH scope AS ({scope_sql}),
                vec_results AS (
                  SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? LIMIT ?
                )
                SELECT c.id, c.content, v.distance, c.source
                FROM vec_results v
                JOIN scope s ON s.id = v.rowid
                JOIN chunks c ON c.id = v.rowid
                ORDER BY v.distance
                LIMIT ?
                """,
                (*scope_params, _to_vec_blob(qvec), candidate_limit, limit),
            ).fetchall()

            self._log(
                "info",
                "search",
                op="search",
                agent=agent_name,
                kind=kind or "all",
                limit=limit,
                hits=len(rows),
                round_hits=sum(1 for r in rows if len(r) > 3 and r[3] == "round"),
                memory_hits=sum(1 for r in rows if len(r) > 3 and r[3] == "memory"),
            )
            return [{"id": str(r[0]), "content": r[1], "score": float(r[2])} for r in rows]
        except Exception as e:
            self._log("error", "search_failed", op="search", agent=agent_name, kind=kind or "all", error=e)
            return []
        finally:
            if conn is not None:
                conn.close()

    async def delete(self, agent_name: str) -> bool:
        try:
            await self.init_tables()
            db = await self._get_db()
            await db.execute(
                "DELETE FROM vec_chunks WHERE rowid IN "
                "(SELECT id FROM chunks WHERE EXISTS "
                "(SELECT 1 FROM json_each(visible_to) WHERE json_each.value = ?))",
                (agent_name,),
            )
            await db.execute(
                "DELETE FROM chunks WHERE EXISTS "
                "(SELECT 1 FROM json_each(visible_to) WHERE json_each.value = ?)",
                (agent_name,),
            )
            await db.commit()
            self._log("info", "delete", op="delete", agent=agent_name, result="ok")
            return True
        except Exception as e:
            self._log("error", "delete_failed", op="delete", agent=agent_name, error=e)
            return False

    async def delete_all_agents(self, agent_names: list[str]) -> dict[str, bool]:
        unique_names = list(dict.fromkeys(agent_names))
        if not unique_names:
            return {}

        all_agents = set(get_agent_names())
        if all_agents and set(unique_names) == all_agents:
            db: aiosqlite.Connection | None = None
            try:
                await self.init_tables()
                db = await self._get_db()
                async with self._get_write_lock():
                    await db.execute("BEGIN")
                    await db.execute("DELETE FROM vec_chunks")
                    await db.execute("DELETE FROM chunks")
                    await db.commit()
                self._conv_game_date.clear()
                self._memory_index_cutoff.clear()
                self._log("info", "delete_all", op="delete_all_agents", mode="full", result="ok")
                return {name: True for name in unique_names}
            except Exception as e:
                await self._rollback(db, op="delete_all_agents", mode="full")
                self._log("warning", "delete_all_fallback", op="delete_all_agents", error=e)

        results = await asyncio.gather(*(self.delete(name) for name in unique_names))
        return dict(zip(unique_names, results))


vector_store = VectorStore()
