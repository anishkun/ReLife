"""Unit tests for the vector-index seam and schema migrations.

These don't need real embeddings — they feed vectors directly — so they run in
the default (embeddings-off) suite and pin the brute-force backend's behaviour
plus the ``PRAGMA user_version`` migration stamping.
"""

import sqlite3
import time
from array import array

import pytest

from relife.memory import store as store_mod
from relife.memory import vector_index


def _pack(vec):
    return array("f", vec).tobytes()


def _mem_db(tmp_path, vectors):
    """A minimal memories table seeded with id→embedding rows."""
    db = tmp_path / "vec.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    store_mod.MemoryStore(db).init_db()
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    now = time.time()
    for i, v in enumerate(vectors, start=1):
        conn.execute(
            "INSERT INTO memories (id, kind, text, tags, importance, created_at, "
            "last_used_at, use_count, status, embedding) "
            "VALUES (?,?,?,?,?,?,?,0,'active',?)",
            (i, "fact", f"m{i}", "", 0.5, now, now, _pack(v)),
        )
    conn.commit()
    return conn


def test_bruteforce_ranks_by_cosine(tmp_path):
    # id1 points along x, id2 along y, id3 between them.
    conn = _mem_db(tmp_path, [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    idx = vector_index.BruteForceIndex()
    hits = idx.search(conn, [1.0, 0.0], limit=3)
    assert [mid for mid, _ in hits] == [1, 3, 2]  # closest to x-axis first
    assert hits[0][1] > hits[1][1] > hits[2][1]


def test_bruteforce_limit_and_archived(tmp_path):
    conn = _mem_db(tmp_path, [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
    conn.execute("UPDATE memories SET status='archived' WHERE id=2")
    conn.commit()
    idx = vector_index.BruteForceIndex()
    assert len(idx.search(conn, [1.0, 0.0], limit=2)) == 2
    active = idx.search(conn, [1.0, 0.0], limit=10)
    assert 2 not in [mid for mid, _ in active]            # archived excluded
    witharch = idx.search(conn, [1.0, 0.0], limit=10, include_archived=True)
    assert 2 in [mid for mid, _ in witharch]              # …unless requested


def test_get_index_falls_back_to_bruteforce(tmp_path, monkeypatch):
    # Whether or not sqlite-vec is installed, a failing self-test (missing or
    # misbehaving extension) must degrade to the brute-force scan.
    monkeypatch.setattr(vector_index, "_cached", {})
    monkeypatch.setattr(vector_index, "_self_test", lambda dim: False)
    assert isinstance(vector_index.get_index(None), vector_index.BruteForceIndex)
    assert isinstance(vector_index.get_index(384), vector_index.BruteForceIndex)


def test_schema_version_stamped(tmp_path):
    db = tmp_path / "v.db"
    s = store_mod.MemoryStore(db)
    s.init_db()
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == s.SCHEMA_VERSION


def test_v1_db_migrates_and_stamps(tmp_path):
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "kind TEXT NOT NULL, text TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '', "
            "created_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO memories (kind, text, created_at) VALUES ('fact','x',?)",
            (time.time(),),
        )
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0  # unversioned

    store_mod.MemoryStore(db).init_db()
    with sqlite3.connect(db) as conn:
        # v1 → v2 → v3 in one pass: an unversioned store gets every later step.
        assert conn.execute("PRAGMA user_version").fetchone()[0] == store_mod.MemoryStore.SCHEMA_VERSION
        cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
        assert {"importance", "last_used_at", "use_count", "status", "embedding"} <= cols
        assert {"space", "source"} <= cols
        assert conn.execute("SELECT space FROM memories").fetchone()[0] == "default"


# --- sqlite-vec: per-DB, incremental, per-dimension (needs the [vector] extra) -----------
def _vec_db(tmp_path, name, rows):
    """A memories table with explicit embeddings, opened with sqlite-vec loaded."""
    pytest.importorskip("sqlite_vec")
    from relife.memory import _sqlite

    path = tmp_path / name
    s = store_mod.MemoryStore(path)
    s.init_db()
    conn = _sqlite.connect(path)
    for mid, vec, space, status in rows:
        conn.execute(
            "INSERT INTO memories (id, kind, text, created_at, embedding, space, status) "
            "VALUES (?, 'fact', ?, 0, ?, ?, ?)",
            (mid, f"m{mid}", array("f", vec).tobytes(), space, status),
        )
    conn.commit()
    return conn


def test_sqlite_vec_sync_is_per_db_and_incremental(tmp_path, monkeypatch):
    pytest.importorskip("sqlite_vec")
    idx = vector_index.SqliteVecIndex(2)
    a = _vec_db(tmp_path, "a.db", [(1, [1.0, 0.0], "default", "active")])
    b = _vec_db(tmp_path, "b.db", [(1, [0.0, 1.0], "default", "active")])
    assert [m for m, _ in idx.search(a, [1.0, 0.0], limit=1)] == [1]
    # Same row count in another DB: a per-process counter skipped this sync.
    assert idx.search(b, [0.0, 1.0], limit=1)[0][1] == pytest.approx(1.0)

    inserted = []
    real = a.executemany

    class Spy:
        def __getattr__(self, n):
            return getattr(a, n)

        def executemany(self, sql, rows):
            rows = list(rows)
            if sql.startswith("INSERT INTO memories_vec"):
                inserted.extend(r[0] for r in rows)
            return real(sql, rows)

    a.execute(
        "INSERT INTO memories (id, kind, text, created_at, embedding) VALUES (2, 'fact', 'm2', 0, ?)",
        (array("f", [0.9, 0.1]).tobytes(),),
    )
    idx.search(Spy(), [1.0, 0.0], limit=2)
    assert inserted == [2]  # only the new row; nothing rebuilt
    a.execute("DELETE FROM memories WHERE id = 1")
    assert [m for m, _ in idx.search(a, [1.0, 0.0], limit=5)] == [2]


def test_sqlite_vec_handles_two_dimensions(tmp_path):
    pytest.importorskip("sqlite_vec")
    vector_index._cached.clear()
    try:
        small = vector_index.get_index(2)
        big = vector_index.get_index(3)
        assert isinstance(small, vector_index.SqliteVecIndex) and small.dim == 2
        assert big.dim == 3 and big is not small
        conn = _vec_db(tmp_path, "d.db", [(1, [1.0, 0.0], "default", "active")])
        assert [m for m, _ in small.search(conn, [1.0, 0.0], limit=1)] == [1]
        # A 3-dim model now: the table is rebuilt for it, the 2-dim row left to brute force.
        assert big.search(conn, [1.0, 0.0, 0.0], limit=1) == []
        assert small.search(conn, [1.0, 0.0, 0.0], limit=1) == []  # wrong length → no ANN
    finally:
        vector_index._cached.clear()


def test_sqlite_vec_a_small_space_is_not_crowded_out(tmp_path):
    pytest.importorskip("sqlite_vec")
    rows = [(i, [1.0, 0.001 * i], "big", "active") for i in range(1, 201)]
    rows.append((500, [0.0, 1.0], "small", "active"))  # far from the query
    conn = _vec_db(tmp_path, "s.db", rows)
    idx = vector_index.SqliteVecIndex(2)
    hits = idx.search(conn, [1.0, 0.0], limit=3, spaces=["small"])
    assert [m for m, _ in hits] == [500]  # a fixed 4x window returned nothing
