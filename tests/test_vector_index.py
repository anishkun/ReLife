"""Unit tests for the vector-index seam and schema migrations.

These don't need real embeddings — they feed vectors directly — so they run in
the default (embeddings-off) suite and pin the brute-force backend's behaviour
plus the ``PRAGMA user_version`` migration stamping.
"""

import sqlite3
import time
from array import array

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


def test_get_index_falls_back_to_bruteforce(tmp_path):
    # sqlite-vec isn't installed in CI; the factory must degrade gracefully.
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
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
        assert {"importance", "last_used_at", "use_count", "status", "embedding"} <= cols
