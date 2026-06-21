"""Long-term memory store (retrieval layer A) — cognitive edition.

SQLite-backed facts/preferences/episodes/patterns whose relevance behaves like a
brain's: it **rises when a memory is used** (reinforcement) and **fades when it
is ignored** (decay), and recall fuses four signals into one ranking score:

    semantic similarity  (local embeddings, optional)
  + keyword overlap      (FTS5 index, scales to large stores)
  + cognitive activation (decay/reinforcement — see ``cognitive.py``)
  + explicit importance

Recall is **two-stage** so it stays cheap as the store grows: a bounded set of
candidates is pulled from the FTS5 keyword index (and, when embeddings are on, a
semantic scan), then the full fused score is computed only over that candidate
set — not the whole table.

The public API (``save`` / ``recall`` / ``count``) stays backward-compatible
with v1; new behavior is additive (extra optional args, extra columns added by an
idempotent migration).
"""

from __future__ import annotations

import sqlite3
import time
from array import array
from dataclasses import dataclass

from .. import config
from . import cognitive, embeddings
from ._text import tokenize as _tokens

_DB_PATH = config.DATA_DIR / "relife.db"

_VALID_KINDS = {"fact", "preference", "episode", "pattern"}

# Whether the SQLite build supports FTS5. Determined at init_db(); when False we
# fall back to a full-table keyword scan (fine for small stores).
_fts_ok = False


def _connect() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _columns(conn) -> set[str]:
    return {r["name"] for r in conn.execute("PRAGMA table_info(memories)")}


def init_db() -> None:
    """Create or migrate the schema (idempotent). Safe to call on every access."""
    global _fts_ok
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                kind         TEXT NOT NULL,
                text         TEXT NOT NULL,
                tags         TEXT NOT NULL DEFAULT '',
                importance   REAL NOT NULL DEFAULT 0.5,
                created_at   REAL NOT NULL,
                last_used_at REAL NOT NULL DEFAULT 0,
                use_count    INTEGER NOT NULL DEFAULT 0,
                status       TEXT NOT NULL DEFAULT 'active',
                embedding    BLOB
            )
            """
        )
        # --- migrate a v1 table that predates the cognitive columns ----------
        cols = _columns(conn)
        migrations = {
            "importance": "ALTER TABLE memories ADD COLUMN importance REAL NOT NULL DEFAULT 0.5",
            "last_used_at": "ALTER TABLE memories ADD COLUMN last_used_at REAL NOT NULL DEFAULT 0",
            "use_count": "ALTER TABLE memories ADD COLUMN use_count INTEGER NOT NULL DEFAULT 0",
            "status": "ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
            "embedding": "ALTER TABLE memories ADD COLUMN embedding BLOB",
        }
        for col, ddl in migrations.items():
            if col not in cols:
                conn.execute(ddl)
        # Backfill last_used_at from created_at for rows that never had it.
        conn.execute(
            "UPDATE memories SET last_used_at = created_at WHERE last_used_at = 0"
        )
        _fts_ok = _init_fts(conn)


def _init_fts(conn) -> bool:
    """Create the FTS5 mirror + sync triggers. Returns False if FTS5 is absent."""
    existed = (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        ).fetchone()
        is not None
    )
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts "
            "USING fts5(text, tags, content='memories', content_rowid='id')"
        )
    except sqlite3.OperationalError:
        return False  # SQLite built without FTS5 — fall back to scan
    conn.executescript(
        """
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, text, tags) VALUES (new.id, new.text, new.tags);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, text, tags)
            VALUES ('delete', old.id, old.text, old.tags);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, text, tags)
            VALUES ('delete', old.id, old.text, old.tags);
            INSERT INTO memories_fts(rowid, text, tags) VALUES (new.id, new.text, new.tags);
        END;
        """
    )
    # First time the index is created over an existing table (e.g. a migrated v1
    # DB): backfill it. (For external-content FTS5, COUNT(*) reads the content
    # table, so it can't be used to detect an empty index — rely on `existed`.)
    if not existed and conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]:
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
    return True


@dataclass
class Memory:
    id: int
    kind: str
    text: str
    tags: str
    created_at: float
    importance: float = 0.5
    last_used_at: float = 0.0
    use_count: int = 0
    status: str = "active"

    def activation(self, now: float | None = None) -> float:
        return cognitive.activation(
            use_count=self.use_count,
            last_used_at=self.last_used_at or self.created_at,
            importance=self.importance,
            now=now,
        )


def _row_to_memory(r) -> Memory:
    return Memory(
        id=r["id"],
        kind=r["kind"],
        text=r["text"],
        tags=r["tags"],
        created_at=r["created_at"],
        importance=r["importance"],
        last_used_at=r["last_used_at"] or r["created_at"],
        use_count=r["use_count"],
        status=r["status"],
    )


# --- embedding (de)serialization -------------------------------------------
def _pack(vec: list[float] | None) -> bytes | None:
    return array("f", vec).tobytes() if vec else None


def _unpack(blob) -> list[float] | None:
    if not blob:
        return None
    a = array("f")
    a.frombytes(blob)
    return list(a)


# --- write -----------------------------------------------------------------
def save(
    text: str,
    kind: str = "fact",
    tags: str = "",
    importance: float | None = None,
) -> int:
    """Persist a memory. Returns its id.

    Saving the same text again does not duplicate it — it **reinforces** the
    existing memory (refreshes recency, bumps use_count, keeps the higher
    importance), exactly as recalling it would.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("memory text is empty")
    if kind not in _VALID_KINDS:
        kind = "fact"
    imp = 0.5 if importance is None else max(0.0, min(1.0, float(importance)))
    init_db()
    now = time.time()
    with _connect() as conn:
        existing = conn.execute(
            "SELECT id, importance, use_count FROM memories WHERE text = ?", (text,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE memories SET last_used_at = ?, use_count = use_count + 1, "
                "tags = ?, importance = ?, status = 'active' WHERE id = ?",
                (
                    now,
                    tags or "",
                    max(existing["importance"], imp),
                    existing["id"],
                ),
            )
            return int(existing["id"])
        emb = _pack(embeddings.embed_one(text))
        cur = conn.execute(
            "INSERT INTO memories (kind, text, tags, importance, created_at, "
            "last_used_at, use_count, status, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, 'active', ?)",
            (kind, text, tags or "", imp, now, now, emb),
        )
        return int(cur.lastrowid)


def reinforce(mem_id: int, now: float | None = None) -> None:
    """Strengthen a memory: bump its use_count and refresh recency."""
    now = time.time() if now is None else now
    init_db()
    with _connect() as conn:
        conn.execute(
            "UPDATE memories SET use_count = use_count + 1, last_used_at = ?, "
            "status = 'active' WHERE id = ?",
            (now, mem_id),
        )


def archive(mem_id: int) -> None:
    """Soft-forget a memory (kept, but excluded from default recall)."""
    init_db()
    with _connect() as conn:
        conn.execute("UPDATE memories SET status = 'archived' WHERE id = ?", (mem_id,))


def delete(mem_id: int) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))


# --- read ------------------------------------------------------------------
def _candidates(conn, q_terms: set[str], q_vec, include_archived: bool):
    """Stage 1: a bounded candidate set from FTS5 keyword + semantic scan."""
    status_clause = "" if include_archived else "WHERE m.status = 'active'"
    rows: dict[int, sqlite3.Row] = {}

    # Keyword candidates (indexed, scales to large stores).
    if q_terms and _fts_ok:
        match = " OR ".join(sorted(q_terms))
        active_only = "" if include_archived else "AND m.status = 'active'"
        sql = (
            "SELECT m.* FROM memories_fts f JOIN memories m ON m.id = f.rowid "
            f"WHERE memories_fts MATCH ? {active_only} "
            "ORDER BY bm25(memories_fts) LIMIT ?"
        )
        try:
            for r in conn.execute(sql, (match, config.CANDIDATE_TOPN)):
                rows[r["id"]] = r
        except sqlite3.OperationalError:
            pass  # malformed MATCH — fall through to scan below

    # Fallback keyword scan when FTS5 is unavailable (or returned nothing).
    if q_terms and not _fts_ok:
        for r in conn.execute(f"SELECT m.* FROM memories m {status_clause}"):
            if _tokens(r["text"] + " " + r["tags"]) & q_terms:
                rows[r["id"]] = r

    # Semantic candidates (brute-force cosine over embedded rows; ANN deferred).
    if q_vec is not None:
        scan = conn.execute(
            f"SELECT m.* FROM memories m {status_clause} "
            f"{'AND' if status_clause else 'WHERE'} m.embedding IS NOT NULL"
        )
        sem_scored = []
        for r in scan:
            sim = embeddings.cosine(q_vec, _unpack(r["embedding"]))
            if r["id"] in rows or sim >= config.SEM_CANDIDATE_THRESHOLD:
                sem_scored.append((sim, r))
        sem_scored.sort(key=lambda x: x[0], reverse=True)
        for _, r in sem_scored[: config.CANDIDATE_TOPN]:
            rows.setdefault(r["id"], r)

    return list(rows.values())


def recall(
    query: str,
    k: int = 5,
    *,
    reinforce: bool = False,
    include_archived: bool = False,
) -> list[Memory]:
    """Return up to ``k`` memories most relevant to ``query``.

    Relevance fuses semantic similarity, keyword overlap, cognitive activation,
    and importance (see ``cognitive.fused_score``). A row qualifies only if it
    has keyword overlap or strong semantic similarity, so an unrelated query
    surfaces nothing. Pass ``reinforce=True`` to strengthen what was surfaced
    (recall is itself a use) — the auto-recall hook does this.
    """
    init_db()
    q_terms = _tokens(query)
    q_vec = embeddings.embed_one(query) if embeddings.available() else None
    if not q_terms and q_vec is None:
        return []
    now = time.time()
    scored: list[tuple[float, Memory]] = []
    with _connect() as conn:
        cands = _candidates(conn, q_terms, q_vec, include_archived)
        for r in cands:
            hay = _tokens(r["text"] + " " + r["tags"])
            overlap = len(q_terms & hay)
            sem = embeddings.cosine(q_vec, _unpack(r["embedding"])) if q_vec else 0.0
            # Candidate gate: keep today's "unrelated query → nothing" guarantee.
            if overlap == 0 and sem < config.SEM_CANDIDATE_THRESHOLD:
                continue
            kw = overlap / max(1, len(q_terms))
            m = _row_to_memory(r)
            score = cognitive.fused_score(
                semantic=max(0.0, sem),
                keyword=kw,
                act=m.activation(now),
                importance=m.importance,
            )
            scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [m for _, m in scored[:k]]
    if reinforce:
        for m in top:
            reinforce_one(m.id, now)
    return top


def reinforce_one(mem_id: int, now: float | None = None) -> None:
    # Separate name so recall()'s ``reinforce`` kwarg doesn't shadow the helper.
    reinforce(mem_id, now)


def all_memories(include_archived: bool = True) -> list[Memory]:
    """Every memory (for the consolidation sweep)."""
    init_db()
    clause = "" if include_archived else "WHERE status = 'active'"
    with _connect() as conn:
        rows = conn.execute(f"SELECT * FROM memories {clause}").fetchall()
    return [_row_to_memory(r) for r in rows]


def get(mem_id: int) -> Memory | None:
    init_db()
    with _connect() as conn:
        r = conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()
    return _row_to_memory(r) if r else None


def count(include_archived: bool = True) -> int:
    init_db()
    clause = "" if include_archived else "WHERE status = 'active'"
    with _connect() as conn:
        return int(
            conn.execute(f"SELECT COUNT(*) AS n FROM memories {clause}").fetchone()["n"]
        )
