"""Tests for the semantic (embeddings-on) memory paths.

These are marked ``semantic`` so the autouse "embeddings off" fixture in
conftest leaves real local embeddings enabled, and they skip when ``fastembed``
isn't actually installed — so CI without the optional dep stays green while a
dev box with it gets real coverage of paraphrase dedup and semantic recall.
"""

import time

import pytest

from relife import config
from relife.memory import consolidate, embeddings, events, store, workflows

pytestmark = [
    pytest.mark.semantic,
    pytest.mark.skipif(
        not embeddings.available(), reason="local embeddings not installed"
    ),
]


def _isolate(tmp_path):
    db = tmp_path / "relife.db"
    store._DB_PATH = db
    events._DB_PATH = db
    workflows._WORKFLOWS_DIR = tmp_path / "workflows"
    consolidate._STATE_PATH = tmp_path / "consolidate_state.json"
    store.init_db()
    events.init_db()


def test_save_reinforces_paraphrase(tmp_path):
    """A near-identical reworded memory reinforces the original, not a clone."""
    s = store.MemoryStore(tmp_path / "a.db")
    a = s.save("The user prefers ruff and pytest for Python.", kind="preference")
    b = s.save("The user likes ruff and pytest for Python projects.", kind="preference")
    assert a == b, "paraphrase should reinforce the existing memory"
    assert s.count() == 1
    assert s.get(a).use_count >= 1


def test_save_keeps_distinct_facts_separate(tmp_path):
    """Distinct-but-related facts are NOT merged at save time."""
    s = store.MemoryStore(tmp_path / "b.db")
    s.save("Deploys go through the staging branch first.")
    s.save("The CI pipeline runs unit tests on every push.")
    assert s.count() == 2


def test_consolidation_merges_paraphrases(tmp_path, monkeypatch):
    """Consolidation semantic dedup merges paraphrases keyword overlap misses.

    Save-time dedup is disabled here (threshold raised out of range) so both
    rows persist and the consolidation pass is what does the merging.
    """
    monkeypatch.setattr(config, "SAVE_DEDUP_SIM", 2.0)  # never dedup on save
    _isolate(tmp_path)
    store.save("Production secrets are stored in the OS keyring.")
    store.save("Production secrets live inside the operating system keyring.")
    assert store.count(include_archived=False) == 2  # both saved
    report = consolidate.run_consolidation()
    assert report.merged >= 1
    assert store.count(include_archived=False) == 1


def test_semantic_recall_without_keyword_overlap(tmp_path):
    """A query with no shared tokens still recalls a semantically close memory."""
    s = store.MemoryStore(tmp_path / "c.db")
    s.save("The user prefers ruff and pytest for Python.", kind="preference")
    # No token overlap with the memory, but semantically close (cosine > gate).
    hits = s.recall("favorite code quality and unit testing tooling")
    assert hits and "ruff" in hits[0].text.lower()
