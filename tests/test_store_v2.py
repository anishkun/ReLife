"""Tests for the cognitive store: migration, reinforcement, decay/archival."""

import sqlite3
import time

from relife.memory import store as store_mod


def _fresh(tmp_path):
    store_mod._DB_PATH = tmp_path / "relife.db"
    store_mod.init_db()
    return store_mod


def test_migrates_v1_schema(tmp_path):
    """An old v1 DB (no cognitive columns) is migrated in place."""
    db = tmp_path / "relife.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "kind TEXT NOT NULL, text TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '', "
            "created_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO memories (kind, text, tags, created_at) VALUES (?,?,?,?)",
            ("preference", "The user prefers ruff for Python.", "python", time.time()),
        )

    store_mod._DB_PATH = db
    store_mod.init_db()  # should ALTER in the new columns + backfill
    hits = store_mod.recall("what python linter?")
    assert hits and "ruff" in hits[0].text.lower()
    # Backfilled fields are present and sane.
    assert hits[0].use_count == 0
    assert hits[0].last_used_at == hits[0].created_at
    assert hits[0].status == "active"


def test_reinforcement_strengthens_and_reorders(tmp_path):
    s = _fresh(tmp_path)
    a = s.save("Python ruff is the linter.", tags="python")
    b = s.save("Python pytest is the test runner.", tags="python")

    # Both match "python" equally on keywords; reinforce A so it activates higher.
    for _ in range(8):
        s.reinforce(a)

    hits = s.recall("python")
    assert [m.id for m in hits][0] == a
    assert s.get(a).use_count >= 8


def test_recall_reinforce_flag_bumps_use_count(tmp_path):
    s = _fresh(tmp_path)
    mid = s.save("Deploys go through staging first.", tags="deploy")
    before = s.get(mid).use_count
    s.recall("how do deploys work", reinforce=True)
    assert s.get(mid).use_count == before + 1


def test_archived_excluded_unless_requested(tmp_path):
    s = _fresh(tmp_path)
    mid = s.save("The staging deploy pipeline runs nightly.", tags="deploy")
    s.archive(mid)
    assert s.recall("deploy pipeline") == []
    assert any(m.id == mid for m in s.recall("deploy pipeline", include_archived=True))


def test_unrelated_query_returns_nothing(tmp_path):
    s = _fresh(tmp_path)
    s.save("The user prefers ruff.", kind="preference", tags="python")
    assert s.recall("how tall is mount everest") == []


def test_duplicate_save_reinforces(tmp_path):
    s = _fresh(tmp_path)
    a = s.save("Exact same memory text.")
    b = s.save("Exact same memory text.")
    assert a == b
    assert s.count() == 1
    assert s.get(a).use_count >= 1  # the re-save counted as a use


def test_injectable_stores_are_isolated(tmp_path):
    """Two MemoryStore instances on different files don't share state."""
    a = store_mod.MemoryStore(tmp_path / "a.db")
    b = store_mod.MemoryStore(tmp_path / "b.db")
    a.save("Only in store A.", tags="alpha")
    assert a.count() == 1
    assert b.count() == 0
    assert a.recall("store A") and b.recall("store A") == []
