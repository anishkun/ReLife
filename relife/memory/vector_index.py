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
        self, conn, q_vec: list[float], limit: int, include_archived: bool = False
    ) -> list[tuple[int, float]]:
        ...

    def sync(self, conn) -> None:
        """Rebuild/refresh the index from the embedding column (no-op for brute force)."""
        ...


class BruteForceIndex:
    """Exhaustive cosine scan over the embedding column. Always correct."""

    name = "bruteforce"

    def search(
        self, conn, q_vec: list[float], limit: int, include_archived: bool = False
    ) -> list[tuple[int, float]]:
        where = "embedding IS NOT NULL" + ("" if include_archived else " AND status='active'")
        scored: list[tuple[int, float]] = []
        for r in conn.execute(f"SELECT id, embedding FROM memories WHERE {where}"):
            sim = embeddings.cosine(q_vec, _unpack(r["embedding"]))
            scored.append((int(r["id"]), sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def sync(self, conn) -> None:  # nothing to sync — the column is the index
        return None


class SqliteVecIndex:
    """ANN candidate search backed by the ``sqlite-vec`` ``vec0`` virtual table.

    Synced from the embedding column on each search (cheap relative to recall at
    the scales this matters). Cosine is recomputed exactly by the caller from the
    column, so the vec table only needs to return the right *candidate set* — we
    use it to prune, not to rank, which makes us robust to distance-metric
    quirks.
    """

    name = "sqlite-vec"

    def __init__(self, dim: int):
        self.dim = dim
        self._synced_count = -1

    @staticmethod
    def _load(conn) -> None:
        import sqlite_vec  # type: ignore

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

    def _ensure_table(self, conn) -> None:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec "
            f"USING vec0(embedding float[{self.dim}])"
        )

    def sync(self, conn) -> None:
        """Mirror active embeddings into the vec table when membership changed."""
        self._load(conn)
        self._ensure_table(conn)
        n = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        if n == self._synced_count:
            return
        conn.execute("DELETE FROM memories_vec")
        for r in conn.execute(
            "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
        ):
            conn.execute(
                "INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                (r["id"], r["embedding"]),
            )
        self._synced_count = n

    def search(
        self, conn, q_vec: list[float], limit: int, include_archived: bool = False
    ) -> list[tuple[int, float]]:
        self._load(conn)
        self.sync(conn)
        q = array("f", q_vec).tobytes()
        rows = conn.execute(
            "SELECT v.rowid AS id, m.embedding AS embedding, m.status AS status "
            "FROM memories_vec v JOIN memories m ON m.id = v.rowid "
            "WHERE v.embedding MATCH ? AND k = ? ORDER BY distance",
            (q, limit),
        ).fetchall()
        out: list[tuple[int, float]] = []
        for r in rows:
            if not include_archived and r["status"] != "active":
                continue
            out.append((int(r["id"]), embeddings.cosine(q_vec, _unpack(r["embedding"]))))
        return out


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


_cached: VectorIndex | None = None


def get_index(dim: int | None = None) -> VectorIndex:
    """Return the best available vector index (cached).

    Prefers ``sqlite-vec`` when the package imports AND a self-test passes on
    this machine; otherwise the always-correct brute-force scan. ``dim`` is the
    embedding dimension (required to build the vec table); when unknown we can't
    use the ANN backend yet, so we return brute force.
    """
    global _cached
    if _cached is not None:
        return _cached
    if dim:
        try:
            import sqlite_vec  # noqa: F401

            if _self_test(dim):
                _cached = SqliteVecIndex(dim)
                return _cached
        except Exception:
            pass
    return BruteForceIndex()  # not cached: a later call with a known dim may upgrade
