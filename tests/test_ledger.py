"""Unit tests for the build ledger (durable plan + progress, resume support)."""

import time

import pytest

from relife import config
from relife.build.ledger import BuildLedger


@pytest.fixture
def builds_dir(tmp_path, monkeypatch):
    """Point the ledger at a temp builds dir so tests don't touch real data."""
    d = tmp_path / "builds"
    monkeypatch.setattr(config, "BUILDS_DIR", d)
    return d


def test_create_persists_json_and_markdown(builds_dir):
    led = BuildLedger.create("build a todo app", "/ws")
    assert led.json_path.exists()
    assert led.md_path.exists()
    assert "build a todo app" in led.md_path.read_text("utf-8")


def test_set_plan_and_status_transitions(builds_dir):
    led = BuildLedger.create("spec", "/ws")
    led.set_plan(["Scaffold app", "Add models", "Write tests"])
    assert [m.id for m in led.milestones] == [1, 2, 3]
    assert len(led.pending()) == 3

    led.update_milestone(1, "in_progress")
    led.update_milestone(1, "done", "scaffolded FastAPI app; health route passes")
    assert len(led.done()) == 1
    assert led.milestones[0].summary.startswith("scaffolded")
    assert not led.is_complete()

    led.update_milestone(2, "done")
    led.update_milestone(3, "done")
    assert led.is_complete()


def test_update_validates_status_and_id(builds_dir):
    led = BuildLedger.create("spec", "/ws")
    led.set_plan(["only one"])
    with pytest.raises(ValueError):
        led.update_milestone(1, "bogus")
    with pytest.raises(KeyError):
        led.update_milestone(99, "done")


def test_load_roundtrip_preserves_state(builds_dir):
    led = BuildLedger.create("spec", "/ws")
    led.set_plan(["a", "b"])
    led.update_milestone(1, "done", "did a")
    led.set_session_id("sess-123")

    reloaded = BuildLedger.load(led.build_id)
    assert reloaded.session_id == "sess-123"
    assert reloaded.milestones[0].status == "done"
    assert reloaded.milestones[0].summary == "did a"
    assert reloaded.spec == "spec"


def test_latest_for_picks_most_recent_matching_workspace(builds_dir):
    old = BuildLedger.create("old", "/ws-a")
    time.sleep(0.01)
    new = BuildLedger.create("new", "/ws-a")
    other = BuildLedger.create("other", "/ws-b")

    found = BuildLedger.latest_for("/ws-a")
    assert found is not None
    assert found.build_id == new.build_id
    assert BuildLedger.latest_for("/ws-b").build_id == other.build_id
    assert BuildLedger.latest_for("/nope") is None


def test_status_brief_and_markdown_render(builds_dir):
    led = BuildLedger.create("spec", "/ws")
    led.set_plan(["first", "second"])
    led.update_milestone(1, "done", "done it")
    brief = led.status_brief()
    assert "1. [done] first — done it" in brief
    assert "2. [pending] second" in brief
    md = led.render_markdown()
    assert "first" in md and "second" in md
