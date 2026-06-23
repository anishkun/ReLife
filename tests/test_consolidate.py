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


def test_recurring_meaningful_sequence_becomes_workflow(tmp_path):
    _isolate(tmp_path)
    # The same git-clone → test → git-push procedure across several tasks. The
    # action is derived from the Bash command (brief), not the bare tool name.
    for t in ("task1", "task2", "task3"):
        events.log_event("Bash", "git clone https://github.com/x/y", task_id=t)
        events.log_event("Bash", "mvn test", task_id=t)
        events.log_event("Bash", "git push origin feat/x", task_id=t)

    report = consolidate.run_consolidation()

    assert report.workflows_created, "expected a synthesized workflow"
    assert workflows.count() >= 1
    # The pattern/workflow is named by action, not by "Bash".
    patterns = [m for m in store.all_memories() if m.kind == "pattern"]
    assert any("git-clone" in m.text or "git-push" in m.text for m in patterns)
    assert any("git" in name for name in report.workflows_created)


def test_trivial_tool_sequence_is_ignored(tmp_path):
    _isolate(tmp_path)
    # Pure editor motion (Write → Edit) repeated many times: a real regularity,
    # but NOT a reusable procedure — it must not become a workflow or a pattern.
    for t in ("task1", "task2", "task3", "task4"):
        events.log_event("Write", task_id=t)
        events.log_event("Edit", task_id=t)

    report = consolidate.run_consolidation()

    assert not report.workflows_created, "trivial editor motion must not synthesize a workflow"
    assert workflows.count() == 0
    seq_patterns = [
        m for m in store.all_memories()
        if m.kind == "pattern" and "→" in m.text
    ]
    assert not seq_patterns, "trivial sequence must not record a tool-sequence pattern"


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
