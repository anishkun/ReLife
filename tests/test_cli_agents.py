"""`relife agent …`, `relife memory spaces|export|import`, and `relife mcp`
driven through Typer's CliRunner against an isolated store and registry."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from relife import config
from relife.cli import app
from relife.memory import client as client_mod
from relife.memory import consolidate as consol
from relife.memory import events as ev
from relife.memory import skills as sk
from relife.memory import spaces
from relife.memory import store as store_mod
from relife.memory import workflows as wf

runner = CliRunner()


@pytest.fixture
def cli(tmp_path, monkeypatch):
    db = tmp_path / "relife.db"
    monkeypatch.setattr(store_mod, "_DB_PATH", db)
    monkeypatch.setattr(ev, "_DB_PATH", db)
    monkeypatch.setattr(consol, "_STATE_PATH", tmp_path / "consolidate_state.json")
    monkeypatch.setattr(sk, "_SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(wf, "_WORKFLOWS_DIR", tmp_path / "workflows")
    monkeypatch.setattr(spaces, "_SPACES_DIR", tmp_path / "spaces")
    monkeypatch.setattr(config, "AGENTS_PATH", tmp_path / "agents.json")
    monkeypatch.setattr(config, "ensure_dirs", lambda: None)
    monkeypatch.setattr(client_mod, "_default", None)
    yield client_mod.default_client(), tmp_path


def run(*args: str, input: str | None = None):
    return runner.invoke(app, list(args), input=input)


def test_create_inherit_show_list(cli):
    r = run("agent", "create", "researcher", "--description", "digs into docs")
    assert r.exit_code == 0, r.output
    assert "writes: researcher" in r.output
    r = run("agent", "create", "writer", "--inherit", "researcher")
    assert r.exit_code == 0, r.output
    assert "reads:  writer, researcher, default" in r.output
    r = run("agent", "list")
    assert "researcher" in r.output and "writer" in r.output
    r = run("agent", "show", "writer")
    assert r.exit_code == 0 and "reads" in r.output and "researcher" in r.output


def test_create_errors_exit_2(cli):
    assert run("agent", "create", "default").exit_code == 2
    assert run("agent", "create", "x", "--inherit", "ghost").exit_code == 2
    assert run("agent", "show", "ghost").exit_code == 2


def test_external_agent_prints_connect_config_and_token(cli):
    r = run("agent", "create", "cursor", "--runtime", "external", "--isolated")
    assert r.exit_code == 0, r.output
    assert '"--agent"' in r.output and '"cursor"' in r.output and "RELIFE_HOME" in r.output
    r = run("agent", "token", "cursor")
    assert r.exit_code == 0 and "rla_" in r.output and "Authorization" in r.output
    r = run("agent", "token", "cursor", "--revoke")
    assert r.exit_code == 0 and "revoked" in r.output


def test_promote_asks_then_copies_into_default(cli):
    client, _ = cli
    run("agent", "create", "coder")
    from relife.agents import AgentStore
    from relife.memory.client import ScopedMemoryClient

    ScopedMemoryClient(client, AgentStore().require("coder").scope()).save("run mypy in CI too")
    r = run("agent", "promote", "coder", input="n\n")
    assert "left alone" in r.output and client.recall("mypy CI") == []
    r = run("agent", "promote", "coder", "--yes")
    assert r.exit_code == 0, r.output
    [m] = client.recall("mypy CI")
    assert m.source == "coder"


def test_memory_spaces_export_import(cli):
    client, tmp = cli
    client.save("staging uses a separate database", space="alpha")
    r = run("memory", "spaces")
    assert r.exit_code == 0 and "alpha" in r.output and "main agent" in r.output
    out = tmp / "alpha.json"
    r = run("memory", "export", "alpha", "-o", str(out))
    assert r.exit_code == 0, r.output
    assert json.loads(out.read_text(encoding="utf-8"))["format"] == "relife-memory-pack"
    r = run("memory", "import", str(out), "--space", "beta", "--yes")
    assert r.exit_code == 0 and "imported 1 memories" in r.output
    assert client.all_memories(spaces=["beta"])[0].source == "import:alpha"
    bad = tmp / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert run("memory", "import", str(bad), "--space", "beta", "--yes").exit_code == 2


def test_memory_search_and_list_by_space(cli):
    client, _ = cli
    client.save("alpha alone knows the cron schedule", space="alpha")
    assert "no matching" in run("memory", "search", "cron schedule").output
    r = run("memory", "search", "cron schedule", "--space", "alpha")
    assert "alpha/" in r.output
    assert "alpha/" in run("memory", "list").output


def test_mcp_refuses_an_unknown_agent(cli):
    r = run("mcp", "--agent", "ghost")
    assert r.exit_code == 2
    assert run("mcp").exit_code != 0  # --agent is required


def test_do_refuses_an_unknown_agent(cli):
    assert run("do", "anything", "--agent", "ghost").exit_code == 2
