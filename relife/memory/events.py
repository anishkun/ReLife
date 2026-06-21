"""Tool-event log — the raw material for noticing patterns.

Every tool the agent uses is logged here (tool name + a short hint + which task
it belonged to). On its own this is just a journal; the consolidation pass reads
it to find **recurring action sequences** and promote them into reusable skills
and workflows — how ReLife learns to do multi-step things by itself.

Deliberately tiny and dependency-free. Lives in the same SQLite file as the
memory store but in its own table; ``_DB_PATH`` is a reassignable module global so
tests can isolate it.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from .. import config

_DB_PATH = config.DATA_DIR / "relife.db"


def _connect() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
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


@dataclass
class Event:
    id: int
    task_id: str
    tool: str
    brief: str
    created_at: float


def log_event(tool: str, brief: str = "", task_id: str = "") -> int:
    tool = (tool or "").strip()
    if not tool:
        return 0
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO events (task_id, tool, brief, created_at) VALUES (?, ?, ?, ?)",
            (task_id or "", tool, (brief or "")[:200], time.time()),
        )
        return int(cur.lastrowid)


def recent_events(limit: int = 500) -> list[Event]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    rows.reverse()  # chronological
    return [Event(r["id"], r["task_id"], r["tool"], r["brief"], r["created_at"]) for r in rows]


def events_by_task(limit: int = 500) -> dict[str, list[Event]]:
    """Recent events grouped by task_id, chronological within each task."""
    grouped: dict[str, list[Event]] = {}
    for e in recent_events(limit):
        grouped.setdefault(e.task_id, []).append(e)
    return grouped


def count() -> int:
    init_db()
    with _connect() as conn:
        return int(conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"])
