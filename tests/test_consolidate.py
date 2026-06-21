"""Tests for the consolidation ('sleep') pass: decay, dedupe, pattern learning."""

import sqlite3
import time

from relife import config
from relife.memory import consolidate, events, store, workflows


def _isolate(tmp_path):
    db = tmp_path / "relife.db"
    store._DB_PATH = db
    events._DB_PATH = db
    workflows._WORKFLOWS_DIR = tmp_path / "workflows"
    consolidate._STATE_PATH = tmp_path / "consolidate_state.json"
    store.init_db()
    events.init_db()


def _age_memory(mem_id: int, days: float):
    old = time.time() - days * 86400
    with sqlite3.connect(store._DB_PATH) as conn:
        conn.execute(
            "UPDATE memories SET created_at = ?, last_used_at = ? WHERE id = ?",
            (old, old, mem_id),
        )


def test_decay_archives_stale_keeps_preference(tmp_path):
    _isolate(tmp_path)
    stale = store.save("A finished one-off chore from long ago.", importance=0.1)
    pref = store.save("The user prefers ruff.", kind="preference", importance=0.1)
    _age_memory(stale, config.MIN_FORGET_AGE_DAYS + 60)
    _age_memory(pref, config.MIN_FORGET_AGE_DAYS + 60)

    report = consolidate.run_consolidation()

    assert report.archived >= 1
    assert store.get(stale).status == "archived"
    assert store.get(pref).status == "active"  # preferences never fade


def test_dedupe_merges_duplicates(tmp_path):
    _isolate(tmp_path)
    # Near-identical wording (token sets ~identical) → merged.
    store.save("Deploys go through the staging branch first always.")
    store.save("Deploys always go through the staging branch first.")
    report = consolidate.run_consolidation()
    assert report.merged >= 1


def test_recurring_tool_sequence_becomes_workflow(tmp_path):
    _isolate(tmp_path)
    # The same Read → Edit → Bash sequence across several tasks.
    for t in ("task1", "task2", "task3"):
        for tool in ("Read", "Edit", "Bash"):
            events.log_event(tool, task_id=t)

    report = consolidate.run_consolidation()

    assert report.workflows_created, "expected a synthesized workflow"
    assert workflows.count() >= 1
    # A pattern memory was recorded for it.
    patterns = [m for m in store.all_memories() if m.kind == "pattern"]
    assert any("Read" in m.text for m in patterns)


def test_recurring_episodes_become_pattern(tmp_path):
    _isolate(tmp_path)
    for name in ("alpha", "beta", "gamma"):
        store.save(
            f"Completed build of python cli project {name} with tests.",
            kind="episode",
        )
    report = consolidate.run_consolidation()
    assert report.patterns, "expected a recurring-episode pattern"
    assert any(m.kind == "pattern" for m in store.all_memories())
