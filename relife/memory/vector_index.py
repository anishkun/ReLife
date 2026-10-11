"""Pluggable semantic candidate index — the scalability seam for recall.

Recall's Stage-1 semantic candidate generation needs, for a query vector, the
memories whose embeddings are closest. At small scale a brute-force cosine scan
over the ``memories.embedding`` column is fine; at large scale an approximate
nearest-neighbour (ANN) index is wanted. This module hides that choice behind a
single ``VectorIndex`` contract so the store doesn't care which is in use.

Backends:
- ``BruteForceIndex`` — scans the embedding column (exactly today's behaviour).
  No extra storage, always available, the source of truth for vectors.
- ``SqliteVecIndex`` — a ``vec0`` virtual table via the optional ``sqlite-vec``
  extension (local, offline, no API key). **Soft-optional**, exactly like
  ``fastembed``: ``get_index()`` only returns it after a runtime self-test
  proves a correct round-trip on this machine; otherwise it falls back to
  brute force. So an unavailable or misbehaving extension can never break recall.

The ``memories.embedding`` BLOB column stays the source of truth either way; an
ANN backend is a synced accelerator, never the only copy.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from typing import Protocol

from . import embeddings


def _unpack(blob) -> list[float] | None:
    if not blob:
        return None
    a = array("f")
    a.frombytes(blob)
    return list(a)


class VectorIndex(Protocol):
    """Returns ``(memory_id, cosine_similarity)`` for vectors near a query."""

    name: str

    def search(
        self,
        conn,
        q_vec: list[float],
        limit: int,
        include_archived: bool = False,
        spaces: Sequence[str] | None = None,
    ) -> list[tuple[int, float]]:
        """``spaces`` restricts hits to those memory spaces (``None`` = all)."""
        ...

    def sync(self, conn) -> None:
        """Rebuild/refresh the index from the embedding column (no-op for brute force)."""
        ...


class BruteForceIndex:
    """Exhaustive cosine scan over the embedding column. Always correct."""

    name = "bruteforce"

    def search(
        self,
        conn,
        q_vec: list[float],
        limit: int,
        include_archived: bool = False,
        spaces: Sequence[str] | None = None,
    ) -> list[tuple[int, float]]:
        where = "embedding IS NOT NULL" + ("" if include_archived else " AND status='active'")
        params: list[str] = []
        if spaces is not None:
            params = list(spaces)
            where += f" AND space IN ({','.join('?' * len(params))})" if params else " AND 0"
        scored: list[tuple[int, float]] = []
        for r in conn.execute(f"SELECT id, embedding FROM memories WHERE {where}", params):
            sim = embeddings.cosine(q_vec, _unpack(r["embedding"]))
            scored.append((int(r["id"]), sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def sync(self, conn) -> None:  # nothing to sync — the column is the index
        return None


class SqliteVecIndex:
    """ANN candidate search backed by the ``sqlite-vec`` ``vec0`` virtual table.

    A mirror of the embedding column, kept in step **incrementally** and
    tracked **in the database itself** (``memories_vec_meta``): a signature of
    the indexable rows (count, max id, id sum — ids are AUTOINCREMENT, never
    reused) says whether anything changed, and only rows added or removed since
    are touched. It used to rebuild the whole table whenever a per-process
    counter moved — every save — and that counter followed the process, not
    the DB, so with two DBs (the daemon's, a test's) it could skip a sync it
    needed. The table is built for one dimension; a different model's vectors
    (another length) are left to brute force, and a table of another dimension
    is rebuilt for this one.

    Cosine is recomputed exactly by the caller from the column, so the vec
    table only needs to return the right *candidate set* — we use it to prune,
    not to rank, which makes us robust to distance-metric quirks.
    """

    name = "sqlite-vec"

    def __init__(self, dim: int):
        self.dim = dim

    @staticmethod
    def _load(conn) -> None:
        import sqlite_vec  # type: ignore

        loaded = getattr(conn, "loaded", None)
        if loaded is not None and "vec" in loaded:
            return  # connections are reused now; load once each
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        if loaded is not None:
            loaded.add("vec")

    def _ensure_table(self, conn) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memories_vec_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        row = conn.execute("SELECT value FROM memories_vec_meta WHERE key = 'dim'").fetchone()
        if row is not None and int(row[0]) != self.dim:
            # Built for another model: start over for this one.
            conn.execute("DROP TABLE IF EXISTS memories_vec")
            conn.execute("DELETE FROM memories_vec_meta")
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec "
            f"USING vec0(embedding float[{self.dim}])"
        )
        conn.execute(
            "INSERT OR REPLACE INTO memories_vec_meta(key, value) VALUES ('dim', ?)", (str(self.dim),)
        )

    def _indexable(self) -> str:
        return f"embedding IS NOT NULL AND length(embedding) = {4 * self.dim}"

    def sync(self, conn) -> None:
        """Bring the mirror in step with the embedding column (see class doc)."""
        self._load(conn)
        self._ensure_table(conn)
        where = self._indexable()
        n, top, total = conn.execute(
            f"SELECT COUNT(*), COALESCE(MAX(id), 0), COALESCE(SUM(id), 0) FROM memories WHERE {where}"
        ).fetchone()
        sig = f"{n}:{top}:{total}"
        row = conn.execute("SELECT value FROM memories_vec_meta WHERE key = 'sig'").fetchone()
        if row is not None and row[0] == sig:
            return
        have = {int(r[0]) for r in conn.execute("SELECT rowid FROM memories_vec")}
        want = {int(r[0]) for r in conn.execute(f"SELECT id FROM memories WHERE {where}")}
        gone = have - want
        if gone:
            conn.executemany("DELETE FROM memories_vec WHERE rowid = ?", [(i,) for i in gone])
        new = sorted(want - have)
        for i in range(0, len(new), 500):
            chunk = new[i : i + 500]
            rows = conn.execute(
                f"SELECT id, embedding FROM memories WHERE id IN ({','.join('?' * len(chunk))})",
                chunk,
            ).fetchall()
            conn.executemany(
                "INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                [(r[0], r[1]) for r in rows],
            )
        conn.execute(
            "INSERT OR REPLACE INTO memories_vec_meta(key, value) VALUES ('sig', ?)", (sig,)
        )

    def search(
        self,
        conn,
        q_vec: list[float],
        limit: int,
        include_archived: bool = False,
        spaces: Sequence[str] | None = None,
    ) -> list[tuple[int, float]]:
        if len(q_vec) != self.dim:
            return []
        self.sync(conn)
        q = array("f", q_vec).tobytes()
        allowed = None if spaces is None else set(spaces)
        total = conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
        # The vec table holds every space (and archived rows), filtered after
        # the neighbour search. Ask for more than needed, and widen until
        # enough pass the filter — a fixed 4x let a big space crowd a small
        # one's memories out of its own recall entirely.
        k = limit if allowed is None and include_archived else limit * 4
        while True:
            k = min(k, total)
            if k <= 0:
                return []
            rows = conn.execute(
                "SELECT v.rowid AS id, m.embedding AS embedding, m.status AS status, "
                "m.space AS space "
                "FROM memories_vec v JOIN memories m ON m.id = v.rowid "
                "WHERE v.embedding MATCH ? AND k = ? ORDER BY distance",
                (q, k),
            ).fetchall()
            out: list[tuple[int, float]] = []
            for r in rows:
                if not include_archived and r["status"] != "active":
                    continue
                if allowed is not None and r["space"] not in allowed:
                    continue
                out.append((int(r["id"]), embeddings.cosine(q_vec, _unpack(r["embedding"]))))
                if len(out) >= limit:
                    return out
            if k >= total:
                return out
            k *= 4


def _self_test(dim: int) -> bool:
    """Prove a correct sqlite-vec round-trip in an in-memory DB before trusting it."""
    import sqlite3

    try:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        SqliteVecIndex._load(conn)
        idx = SqliteVecIndex(dim)
        idx._ensure_table(conn)
        vec = array("f", [0.0] * dim)
        vec[0] = 1.0
        conn.execute(
            "INSERT INTO memories_vec(rowid, embedding) VALUES (1, ?)", (vec.tobytes(),)
        )
        hit = conn.execute(
            "SELECT rowid FROM memories_vec WHERE embedding MATCH ? AND k = 1",
            (vec.tobytes(),),
        ).fetchone()
        conn.close()
        return hit is not None and int(hit[0]) == 1
    except Exception:
        return False


# Per dimension: an index is built for one embedding length, and a process can
# meet two (a model change; tests with small fake vectors). One cache for the
# first dim seen made every later search of another length fail.
_cached: dict[int, VectorIndex] = {}


def get_index(dim: int | None = None) -> VectorIndex:
    """Return the best available vector index (cached).

    Prefers ``sqlite-vec`` when the package imports AND a self-test passes on
    this machine; otherwise the always-correct brute-force scan. ``dim`` is the
    embedding dimension (required to build the vec table); when unknown we can't
    use the ANN backend yet, so we return brute force.
    """
    if not dim:
        return BruteForceIndex()  # not cached: a later call with a known dim may upgrade
    idx = _cached.get(dim)
    if idx is None:
        idx = BruteForceIndex()
        try:
            import sqlite_vec  # noqa: F401

            if _self_test(dim):
                idx = SqliteVecIndex(dim)
        except Exception:
            pass
        _cached[dim] = idx
    return idx
