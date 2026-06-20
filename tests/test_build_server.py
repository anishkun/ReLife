"""Unit tests for the build ledger MCP tools (mutating the ledger on disk)."""

import anyio
import pytest

from relife import config
from relife.build.ledger import BuildLedger
from relife.build.server import build_tools


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDS_DIR", tmp_path / "builds")
    return BuildLedger.create("build a todo app", "/ws")


def _by_name(tools):
    return {t.name: t.handler for t in tools}


def test_plan_set_then_update_then_status(ledger):
    h = _by_name(build_tools(ledger))

    res = anyio.run(h["build_plan_set"], {"milestones": ["Scaffold", "Models", "Tests"]})
    assert not res.get("is_error")
    assert len(ledger.milestones) == 3

    anyio.run(h["build_milestone_update"], {"id": 1, "status": "done", "summary": "scaffolded"})
    # reload from disk to prove it persisted, not just mutated in memory
    reloaded = BuildLedger.load(ledger.build_id)
    assert reloaded.milestones[0].status == "done"
    assert reloaded.milestones[0].summary == "scaffolded"

    status = anyio.run(h["build_status"], {})
    text = status["content"][0]["text"]
    assert "build a todo app" in text
    assert "[done] Scaffold" in text


def test_plan_set_rejects_empty(ledger):
    h = _by_name(build_tools(ledger))
    res = anyio.run(h["build_plan_set"], {"milestones": ["  ", ""]})
    assert res.get("is_error")
    assert ledger.milestones == []


def test_update_bad_id_returns_error(ledger):
    h = _by_name(build_tools(ledger))
    anyio.run(h["build_plan_set"], {"milestones": ["only"]})
    res = anyio.run(h["build_milestone_update"], {"id": 42, "status": "done"})
    assert res.get("is_error")
