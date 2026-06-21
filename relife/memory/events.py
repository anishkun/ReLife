"""Tool-event log — the raw material for noticing patterns.

Every tool the agent uses is logged here (tool name + a short hint + which task
it belonged to). On its own this is just a journal; the consolidation pass reads
it to find **recurring action sequences** and promote them into reusable skills
and workflows — how ReLife learns to do multi-step things by itself.

Deliberately tiny and dependency-free. Lives in the same SQLite file as the
memory store but in its own table. Implemented as an injectable ``EventLog``
bound to one ``db_path``; the module-level functions delegate to a default
instance, and ``_DB_PATH`` stays a reassignable global (tests rely on it).
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .. import config

_DB_PATH = config.DATA_DIR / "relife.db"


@dataclass
class Event:
    id: int
    task_id: str
    tool: str
    brief: str
    created_at: float


class EventLog:
    """Append-only tool-event journal bound to a single SQLite file."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id    TEXT NOT NULL DEFAULT '',
                    tool       TEXT NOT NULL,
                    brief      TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS events_task ON events(task_id, id)"
            )

    def log_event(self, tool: str, brief: str = "", task_id: str = "") -> int:
        tool = (tool or "").strip()
        if not tool:
            return 0
        self.init_db()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO events (task_id, tool, brief, created_at) VALUES (?, ?, ?, ?)",
                (task_id or "", tool, (brief or "")[:200], time.time()),
            )
            return int(cur.lastrowid)

    def recent_events(self, limit: int = 500) -> list[Event]:
        self.init_db()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        rows.reverse()  # chronological
        return [
            Event(r["id"], r["task_id"], r["tool"], r["brief"], r["created_at"])
            for r in rows
        ]

    def events_by_task(self, limit: int = 500) -> dict[str, list[Event]]:
        """Recent events grouped by task_id, chronological within each task."""
        grouped: dict[str, list[Event]] = {}
        for e in self.recent_events(limit):
            grouped.setdefault(e.task_id, []).append(e)
        return grouped

    def count(self) -> int:
        self.init_db()
        with self._connect() as conn:
            return int(
                conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
            )


# --- default-instance plumbing + back-compat module API --------------------
_default: EventLog | None = None


def _log() -> EventLog:
    global _default
    if _default is None or _default.db_path != Path(_DB_PATH):
        _default = EventLog(_DB_PATH)
    return _default


def init_db() -> None:
    _log().init_db()


def log_event(tool: str, brief: str = "", task_id: str = "") -> int:
    return _log().log_event(tool, brief=brief, task_id=task_id)


def recent_events(limit: int = 500) -> list[Event]:
    return _log().recent_events(limit)


def events_by_task(limit: int = 500) -> dict[str, list[Event]]:
    return _log().events_by_task(limit)


def count() -> int:
    return _log().count()
