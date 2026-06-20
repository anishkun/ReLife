"""Unit tests for the memory store (keyword + recency recall)."""

import importlib

from relife.memory import store as store_mod


def _fresh_store(tmp_path):
    store_mod._DB_PATH = tmp_path / "test.db"  # redirect to a temp DB
    importlib.reload  # noqa: B018 - keep ref; module already imported
    store_mod.init_db()
    return store_mod


def test_save_and_recall_roundtrip(tmp_path):
    s = _fresh_store(tmp_path)
    s.save("The user prefers ruff and pytest for Python projects.", kind="preference", tags="python,tools")
    s.save("The weather API key lives in the OS keyring.", kind="fact", tags="secrets")

    hits = s.recall("what python tools should I use?")
    assert hits, "expected a relevant memory"
    assert "ruff" in hits[0].text.lower()


def test_unrelated_query_returns_nothing(tmp_path):
    s = _fresh_store(tmp_path)
    s.save("The user prefers ruff and pytest.", kind="preference", tags="python")
    assert s.recall("how tall is mount everest") == []


def test_duplicate_text_not_duplicated(tmp_path):
    s = _fresh_store(tmp_path)
    a = s.save("Same exact memory text.")
    b = s.save("Same exact memory text.")
    assert a == b
    assert s.count() == 1


def test_recall_ranks_overlap_first(tmp_path):
    s = _fresh_store(tmp_path)
    s.save("Deploys go through the staging branch first.", tags="deploy")
    s.save("The user likes concise commit messages.", tags="git")
    hits = s.recall("how do deploys work")
    assert hits and "deploy" in (hits[0].text + hits[0].tags).lower()
