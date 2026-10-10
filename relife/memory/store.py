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

The store is an injectable ``MemoryStore`` bound to one ``db_path``. The
module-level ``save`` / ``recall`` / ``count`` / … functions delegate to a lazily
constructed **default** instance bound to ``config.DATA_DIR/relife.db``. The
``_DB_PATH`` module global remains reassignable (tests rely on it): the default
instance is rebuilt whenever it changes, so the public API stays
backward-compatible with v1 while the implementation lives on the class.
"""

from __future__ import annotations

import sqlite3
import time
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .. import config
from . import cognitive, embeddings, vector_index
from ._text import tokenize as _tokens
from .spaces import DEFAULT_SPACE, validate_space

# Reassignable module global the default store binds to (tests redirect it).
_DB_PATH = config.DATA_DIR / "relife.db"

_VALID_KINDS = {"fact", "preference", "episode", "pattern"}


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
    # Which memory space (agent partition) holds it, and who wrote it: "" is the
    # main ReLife agent, else an agent name or "import:<space>".
    space: str = DEFAULT_SPACE
    source: str = ""

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
        space=r["space"],
        source=r["source"],
    )


def _space_filter(spaces: Sequence[str] | None, col: str = "space") -> tuple[str, list[str]]:
    """SQL condition (no leading AND) + params restricting ``col`` to ``spaces``.

    ``None`` means unfiltered (the store is mechanism; scope policy lives in the
    service). An empty sequence matches nothing — a scope that can read no space
    must not fall through to "everything".
    """
    if spaces is None:
        return "", []
    spaces = list(spaces)
    if not spaces:
        return "0", []
    return f"{col} IN ({','.join('?' * len(spaces))})", spaces


def _where(*conds: str) -> str:
    live = [c for c in conds if c]
    return ("WHERE " + " AND ".join(live)) if live else ""


# --- embedding (de)serialization -------------------------------------------
def _pack(vec: list[float] | None) -> bytes | None:
    return array("f", vec).tobytes() if vec else None


def _unpack(blob) -> list[float] | None:
    if not blob:
        return None
    a = array("f")
    a.frombytes(blob)
    return list(a)


class MemoryStore:
    """A long-term memory store bound to a single SQLite file.

    All DB state (connection path + whether FTS5 is available) lives on the
    instance, so multiple isolated stores can coexist (e.g. one per test) and a
    later out-of-process split has a single owning object.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        # Whether the SQLite build supports FTS5. Determined at init_db(); when
        # False we fall back to a full-table keyword scan (fine for small stores).
        self._fts_ok = False

    # --- connection / schema ------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Wait briefly instead of erroring instantly on SQLITE_BUSY, so a local
        # write that races the memory daemon (a command run without
        # RELIFE_MEMORY_URL) retries rather than failing. Sole-writer discipline
        # still holds; this just removes the sharp edge.
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    @staticmethod
    def _columns(conn) -> set[str]:
        return {r["name"] for r in conn.execute("PRAGMA table_info(memories)")}

    # Current on-disk schema version. Tracked via SQLite's built-in
    # ``PRAGMA user_version`` so upgrades apply once, in order, on a long-lived
    # store. v2 = the cognitive columns (importance/last_used_at/use_count/
    # status/embedding); v3 = memory spaces (space/source + index). Add future
    # steps to ``_apply_migrations``.
    SCHEMA_VERSION = 3

    def init_db(self) -> None:
        """Create or migrate the schema (idempotent). Safe on every access."""
        with self._connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                # A fresh DB, or one that predates version tracking (a v1 store,
                # or an already-v2 store with user_version never set). Create the
                # table if absent, then idempotently ensure the v2 columns exist
                # — covers all three cases — and stamp **2**, not the current
                # version: the later steps below must still run on a fresh DB.
                self._create_schema(conn)
                self._migrate_to_v2(conn)
                conn.execute("PRAGMA user_version = 2")
                version = 2
            self._apply_migrations(conn, version)
            self._fts_ok = self._init_fts(conn)

    @staticmethod
    def _create_schema(conn) -> None:
        """Create the current (v2) memories table if it doesn't exist yet."""
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

    def _migrate_to_v2(self, conn) -> None:
        """Bring a pre-cognitive (v1) table up to v2: add columns + backfill.

        Idempotent — each column is added only if missing — so it is a no-op on
        a table that is already v2.
        """
        cols = self._columns(conn)
        adds = {
            "importance": "ALTER TABLE memories ADD COLUMN importance REAL NOT NULL DEFAULT 0.5",
            "last_used_at": "ALTER TABLE memories ADD COLUMN last_used_at REAL NOT NULL DEFAULT 0",
            "use_count": "ALTER TABLE memories ADD COLUMN use_count INTEGER NOT NULL DEFAULT 0",
            "status": "ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
            "embedding": "ALTER TABLE memories ADD COLUMN embedding BLOB",
        }
        for col, ddl in adds.items():
            if col not in cols:
                conn.execute(ddl)
        # Backfill last_used_at from created_at for rows that never had it.
        conn.execute(
            "UPDATE memories SET last_used_at = created_at WHERE last_used_at = 0"
        )

    def _apply_migrations(self, conn, from_version: int) -> None:
        """Apply ordered migrations for versions above ``from_version``.

        Each step is idempotent (columns are added only if missing) and stamps
        its version, so a long-lived store upgrades once, in order.
        """
        if from_version < 3:
            # v3: memory spaces. Every pre-existing row belongs to the default
            # space (the main agent's memory) — exactly what it was before.
            cols = self._columns(conn)
            if "space" not in cols:
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN space TEXT NOT NULL "
                    f"DEFAULT '{DEFAULT_SPACE}'"
                )
            if "source" not in cols:
                conn.execute("ALTER TABLE memories ADD COLUMN source TEXT NOT NULL DEFAULT ''")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS memories_space ON memories(space, status)"
            )
            conn.execute("PRAGMA user_version = 3")

    def _init_fts(self, conn) -> bool:
        """Create the FTS5 mirror + sync triggers. False if FTS5 is absent."""
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
        # First time the index is created over an existing table (e.g. a migrated
        # v1 DB): backfill it. (For external-content FTS5, COUNT(*) reads the
        # content table, so it can't detect an empty index — rely on `existed`.)
        if not existed and conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]:
            conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        return True

    # --- write --------------------------------------------------------------
    def save(
        self,
        text: str,
        kind: str = "fact",
        tags: str = "",
        importance: float | None = None,
        *,
        space: str = DEFAULT_SPACE,
        source: str = "",
    ) -> int:
        """Persist a memory into ``space``. Returns its id.

        Saving the same text again does not duplicate it — it **reinforces** the
        existing memory (refreshes recency, bumps use_count, keeps the higher
        importance), exactly as recalling it would. Duplicates are judged
        *within the space*: the same sentence in two agents' spaces is two
        memories, and one agent's save never strengthens another's.
        """
        validate_space(space)
        text = (text or "").strip()
        if not text:
            raise ValueError("memory text is empty")
        if kind not in _VALID_KINDS:
            kind = "fact"
        if importance is None:
            imp = config.DEFAULT_IMPORTANCE.get(kind, 0.5)
        else:
            imp = max(0.0, min(1.0, float(importance)))
        self.init_db()
        now = time.time()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id, importance, use_count FROM memories WHERE text = ? AND space = ?",
                (text, space),
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
            emb_vec = embeddings.embed_one(text)
            # Near-duplicate (paraphrase) reinforcement: if a very similar memory
            # already exists, strengthen it instead of inserting a clone. Only
            # active when embeddings are available; otherwise exact-match only.
            if emb_vec is not None:
                dup = self._semantic_duplicate(conn, emb_vec, config.SAVE_DEDUP_SIM, space)
                if dup is not None:
                    conn.execute(
                        "UPDATE memories SET last_used_at = ?, "
                        "use_count = use_count + 1, importance = ?, "
                        "status = 'active' WHERE id = ?",
                        (now, max(dup[1], imp), dup[0]),
                    )
                    return int(dup[0])
            cur = conn.execute(
                "INSERT INTO memories (kind, text, tags, importance, created_at, "
                "last_used_at, use_count, status, embedding, space, source) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, 'active', ?, ?, ?)",
                (kind, text, tags or "", imp, now, now, _pack(emb_vec), space, source or ""),
            )
            return int(cur.lastrowid)

    def _semantic_duplicate(
        self, conn, vec: list[float], threshold: float, space: str = DEFAULT_SPACE
    ) -> tuple[int, float] | None:
        """Best active memory in ``space`` whose embedding cosine to ``vec`` is >= threshold.

        Returns ``(id, importance)`` or ``None``. Goes through the vector index
        (brute force by default, ANN when available); only reached when
        embeddings are on, so the test path never runs it.
        """
        idx = vector_index.get_index(len(vec))
        for mid, sim in idx.search(conn, vec, 1, spaces=(space,)):
            if sim >= threshold:
                r = conn.execute(
                    "SELECT importance FROM memories WHERE id = ?", (mid,)
                ).fetchone()
                if r is not None:
                    return (int(mid), float(r["importance"]))
        return None

    def reinforce(self, mem_id: int, now: float | None = None) -> None:
        """Strengthen a memory: bump its use_count and refresh recency."""
        now = time.time() if now is None else now
        self.init_db()
        with self._connect() as conn:
            conn.execute(
                "UPDATE memories SET use_count = use_count + 1, last_used_at = ?, "
                "status = 'active' WHERE id = ?",
                (now, mem_id),
            )

    def archive(self, mem_id: int) -> None:
        """Soft-forget a memory (kept, but excluded from default recall)."""
        self.init_db()
        with self._connect() as conn:
            conn.execute(
                "UPDATE memories SET status = 'archived' WHERE id = ?", (mem_id,)
            )

    def delete(self, mem_id: int) -> None:
        self.init_db()
        with self._connect() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))

    def set_importance(self, mem_id: int, importance: float) -> None:
        """Set a memory's importance (clamped to 0..1). Used by the REM pass to
        recalibrate salience; does not touch use_count/recency."""
        imp = max(0.0, min(1.0, float(importance)))
        self.init_db()
        with self._connect() as conn:
            conn.execute(
                "UPDATE memories SET importance = ? WHERE id = ?", (imp, mem_id)
            )

    # --- read ---------------------------------------------------------------
    def _candidates(
        self,
        conn,
        q_terms: set[str],
        q_vec,
        include_archived: bool,
        spaces: Sequence[str] | None = None,
    ):
        """Stage 1: a bounded candidate set from FTS5 keyword + semantic scan."""
        active = "" if include_archived else "m.status = 'active'"
        in_space, space_params = _space_filter(spaces, "m.space")
        rows: dict[int, sqlite3.Row] = {}

        # Keyword candidates (indexed, scales to large stores).
        if q_terms and self._fts_ok:
            match = " OR ".join(sorted(q_terms))
            sql = (
                "SELECT m.* FROM memories_fts f JOIN memories m ON m.id = f.rowid "
                f"{_where('memories_fts MATCH ?', active, in_space)} "
                "ORDER BY bm25(memories_fts) LIMIT ?"
            )
            try:
                for r in conn.execute(sql, (match, *space_params, config.CANDIDATE_TOPN)):
                    rows[r["id"]] = r
            except sqlite3.OperationalError:
                pass  # malformed MATCH — fall through to scan below

        # Fallback keyword scan when FTS5 is unavailable (or returned nothing).
        if q_terms and not self._fts_ok:
            for r in conn.execute(
                f"SELECT m.* FROM memories m {_where(active, in_space)}", space_params
            ):
                if _tokens(r["text"] + " " + r["tags"]) & q_terms:
                    rows[r["id"]] = r

        # Semantic candidates via the pluggable vector index (brute force by
        # default; an ANN backend swaps in transparently for large stores).
        if q_vec is not None:
            idx = vector_index.get_index(len(q_vec))
            hits = idx.search(
                conn, q_vec, config.CANDIDATE_TOPN, include_archived, spaces=spaces
            )
            new_ids = [
                mid
                for mid, sim in hits
                if mid not in rows and sim >= config.SEM_CANDIDATE_THRESHOLD
            ]
            if new_ids:
                placeholders = ",".join("?" * len(new_ids))
                for r in conn.execute(
                    f"SELECT m.* FROM memories m WHERE m.id IN ({placeholders})",
                    new_ids,
                ):
                    rows.setdefault(r["id"], r)

        return list(rows.values())

    def _doc_freq(
        self,
        conn,
        terms: set[str],
        include_archived: bool,
        spaces: Sequence[str] | None = None,
    ) -> tuple[int, dict[str, int]]:
        """How many memories contain each query term (and how many there are).

        Counted over the spaces being searched — "common" means common in what
        this agent can see. One indexed COUNT per term with FTS5 — query terms
        are few — or a scan on builds without it (recall already scans there).
        """
        status = "" if include_archived else "status = 'active'"
        in_space, space_params = _space_filter(spaces)
        n = conn.execute(
            f"SELECT COUNT(*) FROM memories {_where(status, in_space)}", space_params
        ).fetchone()[0]
        df: dict[str, int] = {}
        if self._fts_ok:
            active = "" if include_archived else "m.status = 'active'"
            m_space, _ = _space_filter(spaces, "m.space")
            sql = (
                "SELECT COUNT(*) FROM memories_fts f JOIN memories m ON m.id = f.rowid "
                f"{_where('memories_fts MATCH ?', active, m_space)}"
            )
            for t in terms:
                try:
                    df[t] = conn.execute(sql, (f'"{t}"', *space_params)).fetchone()[0]
                except sqlite3.OperationalError:
                    df[t] = 0
        else:
            for r in conn.execute(
                f"SELECT text, tags FROM memories {_where(status, in_space)}", space_params
            ):
                for t in _tokens(r["text"] + " " + r["tags"]) & terms:
                    df[t] = df.get(t, 0) + 1
        return n, df

    def recall(
        self,
        query: str,
        k: int = 5,
        *,
        reinforce: bool = False,
        include_archived: bool = False,
        spaces: Sequence[str] | None = None,
    ) -> list[Memory]:
        """Return up to ``k`` memories most relevant to ``query``.

        Relevance fuses semantic similarity, keyword overlap, cognitive
        activation, and importance (see ``cognitive.fused_score``). A row
        qualifies only if it has keyword overlap or strong semantic similarity,
        so an unrelated query surfaces nothing. Pass ``reinforce=True`` to
        strengthen what was surfaced (recall is itself a use). ``spaces``
        restricts the search to those memory spaces (``None`` = every space;
        callers acting for an agent always pass its scope).
        """
        self.init_db()
        q_terms = _tokens(query)
        q_vec = embeddings.embed_one(query) if embeddings.available() else None
        if not q_terms and q_vec is None:
            return []
        now = time.time()
        scored: list[tuple[float, Memory]] = []
        with self._connect() as conn:
            cands = self._candidates(conn, q_terms, q_vec, include_archived, spaces)
            n_rows, df = (
                self._doc_freq(conn, q_terms, include_archived, spaces) if cands else (0, {})
            )
            # A term in a large share of the store ("write", "test" — words
            # that also name tools in episode/pattern text) is weak evidence.
            common_at = max(config.RECALL_COMMON_MIN_DOCS, config.RECALL_COMMON_TERM_FRACTION * n_rows)
            for r in cands:
                hay = _tokens(r["text"] + " " + r["tags"])
                shared = q_terms & hay
                overlap = len(shared)
                sem = embeddings.cosine(q_vec, _unpack(r["embedding"])) if q_vec else 0.0
                # Candidate gate: keep the "unrelated query → nothing" guarantee.
                if overlap == 0 and sem < config.SEM_CANDIDATE_THRESHOLD:
                    continue
                # Activation + importance alone clear RECALL_FLOOR, so the gate
                # is the only relevance check: one shared word must at least be
                # a distinctive one, or a single "write" in a long prompt
                # surfaces (and then reinforces) unrelated memories.
                if (
                    overlap == 1
                    and sem < config.SEM_CANDIDATE_THRESHOLD
                    and df.get(next(iter(shared)), 0) > common_at
                ):
                    continue
                kw = overlap / max(1, len(q_terms))
                m = _row_to_memory(r)
                score = cognitive.fused_score(
                    semantic=max(0.0, sem),
                    keyword=kw,
                    act=m.activation(now),
                    importance=m.importance,
                    kind=m.kind,
                )
                # Absolute relevance floor: drop barely-relevant tail matches.
                if score < config.RECALL_FLOOR:
                    continue
                scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [m for _, m in scored[:k]]
        if reinforce:
            for m in top:
                self.reinforce(m.id, now)
        return top

    def all_memories(
        self, include_archived: bool = True, spaces: Sequence[str] | None = None
    ) -> list[Memory]:
        """Every memory (for the consolidation sweep), optionally in ``spaces``."""
        self.init_db()
        status = "" if include_archived else "status = 'active'"
        in_space, params = _space_filter(spaces)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories {_where(status, in_space)}", params
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def get(self, mem_id: int) -> Memory | None:
        self.init_db()
        with self._connect() as conn:
            r = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (mem_id,)
            ).fetchone()
        return _row_to_memory(r) if r else None

    def count(
        self, include_archived: bool = True, spaces: Sequence[str] | None = None
    ) -> int:
        self.init_db()
        status = "" if include_archived else "status = 'active'"
        in_space, params = _space_filter(spaces)
        with self._connect() as conn:
            return int(
                conn.execute(
                    f"SELECT COUNT(*) AS n FROM memories {_where(status, in_space)}", params
                ).fetchone()["n"]
            )

    # --- spaces -------------------------------------------------------------
    def space_counts(self, include_archived: bool = False) -> dict[str, int]:
        """``{space: number of memories}`` for every space that holds any."""
        self.init_db()
        status = "" if include_archived else "WHERE status = 'active'"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT space, COUNT(*) AS n FROM memories {status} GROUP BY space ORDER BY space"
            ).fetchall()
        return {r["space"]: int(r["n"]) for r in rows}

    def copy_space(
        self,
        src: str,
        dst: str,
        *,
        ids: Sequence[int] | None = None,
        source: str | None = None,
    ) -> int:
        """Copy ``src``'s active memories (or just ``ids`` of them) into ``dst``.

        The fork/promote primitive. Each copy keeps its strength (importance,
        use count, recency, embedding) — an inherited lesson is as established
        as it was — and its provenance (``source`` overrides it when given).
        A text ``dst`` already holds is reinforced instead of cloned, the same
        rule ``save`` applies, so promoting twice never duplicates. Returns the
        number of memories copied or reinforced.
        """
        validate_space(src)
        validate_space(dst)
        if src == dst:
            return 0
        self.init_db()
        now = time.time()
        params: list[object] = [src]
        only = ""
        if ids is not None:
            ids = [int(i) for i in ids]
            if not ids:
                return 0
            only = f" AND id IN ({','.join('?' * len(ids))})"
            params += ids
        n = 0
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories WHERE space = ? AND status = 'active'{only} ORDER BY id",
                params,
            ).fetchall()
            held = {
                r["text"]: (r["id"], r["importance"])
                for r in conn.execute("SELECT id, text, importance FROM memories WHERE space = ?", (dst,))
            }
            for r in rows:
                if r["text"] in held:
                    mid, imp = held[r["text"]]
                    conn.execute(
                        "UPDATE memories SET use_count = use_count + 1, last_used_at = ?, "
                        "importance = ?, status = 'active' WHERE id = ?",
                        (now, max(imp, r["importance"]), mid),
                    )
                else:
                    cur = conn.execute(
                        "INSERT INTO memories (kind, text, tags, importance, created_at, "
                        "last_used_at, use_count, status, embedding, space, source) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)",
                        (
                            r["kind"], r["text"], r["tags"], r["importance"], r["created_at"],
                            r["last_used_at"], r["use_count"], r["embedding"], dst,
                            r["source"] if source is None else source,
                        ),
                    )
                    held[r["text"]] = (int(cur.lastrowid), r["importance"])
                n += 1
        return n

    def archive_space(self, space: str) -> int:
        """Archive every active memory in ``space`` (reversible, like forget).
        Consolidation's slow hard-delete tier reclaims them later."""
        validate_space(space)
        self.init_db()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE memories SET status = 'archived' WHERE space = ? AND status = 'active'",
                (space,),
            )
            return int(cur.rowcount)


# --- default-instance plumbing + back-compat module API --------------------
# The module-level functions below preserve the v1 call sites (and the tests
# that reassign ``_DB_PATH``). They delegate to a default ``MemoryStore`` rebuilt
# whenever ``_DB_PATH`` changes, so redirecting the global re-points the default.
_default: MemoryStore | None = None


def _store() -> MemoryStore:
    global _default
    if _default is None or _default.db_path != Path(_DB_PATH):
        _default = MemoryStore(_DB_PATH)
    return _default


def init_db() -> None:
    _store().init_db()


def save(
    text: str,
    kind: str = "fact",
    tags: str = "",
    importance: float | None = None,
    *,
    space: str = DEFAULT_SPACE,
    source: str = "",
) -> int:
    return _store().save(
        text, kind=kind, tags=tags, importance=importance, space=space, source=source
    )


def reinforce(mem_id: int, now: float | None = None) -> None:
    _store().reinforce(mem_id, now)


def reinforce_one(mem_id: int, now: float | None = None) -> None:
    _store().reinforce(mem_id, now)


def archive(mem_id: int) -> None:
    _store().archive(mem_id)


def delete(mem_id: int) -> None:
    _store().delete(mem_id)


def set_importance(mem_id: int, importance: float) -> None:
    _store().set_importance(mem_id, importance)


def recall(
    query: str,
    k: int = 5,
    *,
    reinforce: bool = False,
    include_archived: bool = False,
    spaces: Sequence[str] | None = None,
) -> list[Memory]:
    return _store().recall(
        query, k=k, reinforce=reinforce, include_archived=include_archived, spaces=spaces
    )


def all_memories(
    include_archived: bool = True, spaces: Sequence[str] | None = None
) -> list[Memory]:
    return _store().all_memories(include_archived=include_archived, spaces=spaces)


def get(mem_id: int) -> Memory | None:
    return _store().get(mem_id)


def count(include_archived: bool = True, spaces: Sequence[str] | None = None) -> int:
    return _store().count(include_archived=include_archived, spaces=spaces)


def space_counts(include_archived: bool = False) -> dict[str, int]:
    return _store().space_counts(include_archived=include_archived)
