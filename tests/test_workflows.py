"""Unit tests for the workflow store (multi-step procedural memory)."""

from relife.memory import workflows as wf


def _fresh(tmp_path):
    wf._WORKFLOWS_DIR = tmp_path / "workflows"
    return wf


def test_write_find_read_roundtrip(tmp_path):
    w = _fresh(tmp_path)
    w.write_workflow(
        "ship-new-service",
        "Standing up and publishing a brand-new service.",
        "1. scaffold\n2. add tests\n3. create repo\n4. push",
        trigger="scaffold,test,repo,push",
    )
    hits = w.find_workflows("publish a new service repo")
    assert hits and hits[0].slug == "ship-new-service"
    assert "scaffold" in hits[0].body
    assert "push" in hits[0].trigger

    read = w.read_workflow("ship-new-service")
    assert read and read.when_to_use.startswith("Standing up")


def test_rewrite_updates_not_duplicates(tmp_path):
    w = _fresh(tmp_path)
    w.write_workflow("ship", "Shipping.", "old")
    w.write_workflow("ship", "Shipping a service.", "new steps")
    assert w.count() == 1
    assert "new steps" in w.read_workflow("ship").body


def test_unrelated_query_no_match(tmp_path):
    w = _fresh(tmp_path)
    w.write_workflow("ship", "Shipping a service.", "steps")
    assert w.find_workflows("bake a chocolate cake") == []
