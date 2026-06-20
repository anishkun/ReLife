"""Long-term memory store (retrieval layer A).

SQLite-backed facts/episodes with keyword + recency recall. Deliberately simple
and dependency-free for v1 — no embeddings. We add a vector index later when
volume justifies it; the public API (``save`` / ``recall``) stays the same.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from .. import config
from ._text import tokenize as _tokens

_DB_PATH = config.DATA_DIR / "relife.db"

_VALID_KINDS = {"fact", "preference", "episode"}


def _connect() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                kind       TEXT NOT NULL,
                text       TEXT NOT NULL,
                tags       TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            )
            """
        )


@dataclass
class Memory:
    id: int
    kind: str
    text: str
    tags: str
    created_at: float


def save(text: str, kind: str = "fact", tags: str = "") -> int:
    """Persist a memory. Returns its id. Skips exact-duplicate text."""
    text = (text or "").strip()
    if not text:
        raise ValueError("memory text is empty")
    if kind not in _VALID_KINDS:
        kind = "fact"
    init_db()
    with _connect() as conn:
        existing = conn.execute(
            "SELECT id FROM memories WHERE text = ?", (text,)
        ).fetchone()
        if existing:
            # Refresh recency + merge tags rather than duplicate.
            conn.execute(
                "UPDATE memories SET created_at = ?, tags = ? WHERE id = ?",
                (time.time(), tags or "", existing["id"]),
            )
            return int(existing["id"])
        cur = conn.execute(
            "INSERT INTO memories (kind, text, tags, created_at) VALUES (?, ?, ?, ?)",
            (kind, text, tags or "", time.time()),
        )
        return int(cur.lastrowid)


def recall(query: str, k: int = 5) -> list[Memory]:
    """Return up to ``k`` memories relevant to ``query``.

    Score = keyword overlap (text + tags) + a small recency bonus. Memories with
    no keyword overlap are excluded, so an unrelated task surfaces nothing.
    """
    init_db()
    q_terms = _tokens(query)
    if not q_terms:
        return []
    now = time.time()
    scored: list[tuple[float, Memory]] = []
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, kind, text, tags, created_at FROM memories"
        ).fetchall()
    for r in rows:
        hay = _tokens(r["text"] + " " + r["tags"])
        overlap = len(q_terms & hay)
        if overlap == 0:
            continue
        age_days = max(0.0, (now - r["created_at"]) / 86400.0)
        recency = 1.0 / (1.0 + age_days)
        score = overlap + 0.1 * recency
        scored.append(
            (score, Memory(r["id"], r["kind"], r["text"], r["tags"], r["created_at"]))
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:k]]


def count() -> int:
    init_db()
    with _connect() as conn:
        return int(conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"])
