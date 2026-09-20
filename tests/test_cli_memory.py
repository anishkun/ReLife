"""`relife memory search|list|show|forget` — driven through Typer's CliRunner
against an isolated store (no daemon, no model)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from relife.cli import app
from relife.memory import client as client_mod
from relife.memory import store

runner = CliRunner()


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    store._DB_PATH = tmp_path / "relife.db"
    monkeypatch.setattr(client_mod, "_default", None)  # rebuild against the temp DB
    monkeypatch.setattr("relife.config.ensure_dirs", lambda: None)
    c = client_mod.default_client()
    ids = {
        "ruff": c.save("Use ruff for linting in Python projects.", kind="preference", tags="python"),
        "h2": c.save("ApexPay tests run against H2, not Postgres.", kind="fact", tags="apexpay"),
        "ep": c.save("Task: push apex pay | Approach: Bash", kind="episode"),
    }
    return c, ids


def run(*args: str, input: str | None = None):
    return runner.invoke(app, ["memory", *args], input=input)


def test_search_finds_without_reinforcing(isolated):
    c, ids = isolated
    before = c.get(ids["ruff"]).use_count
    r = run("search", "ruff linting")
    assert r.exit_code == 0, r.output
    assert f"#{ids['ruff']}" in r.output and "preference" in r.output
    assert f"#{ids['h2']}" not in r.output
    # Looking must not change what the agent is later shown.
    assert c.get(ids["ruff"]).use_count == before


def test_search_no_hits(isolated):
    r = run("search", "quantum llama")
    assert r.exit_code == 0 and "no matching" in r.output


def test_list_filters_by_kind_and_archived(isolated):
    c, ids = isolated
    r = run("list", "--kind", "fact")
    assert f"#{ids['h2']}" in r.output and f"#{ids['ruff']}" not in r.output

    c.archive(ids["ep"])
    assert f"#{ids['ep']}" not in run("list").output
    r = run("list", "--archived")
    assert f"#{ids['ep']}" in r.output and "[archived]" in r.output


def test_list_rejects_bad_sort(isolated):
    assert run("list", "--sort", "sideways").exit_code == 2


def test_show_prints_full_record(isolated):
    _, ids = isolated
    r = run("show", str(ids["h2"]))
    assert r.exit_code == 0
    assert "ApexPay tests run against H2" in r.output
    assert "tags        apexpay" in r.output
    assert "importance" in r.output and "activation" in r.output


def test_show_unknown_id_fails(isolated):
    r = run("show", "424242")
    assert r.exit_code == 1 and "no memory #424242" in r.output


def test_forget_by_id_asks_then_archives(isolated):
    c, ids = isolated
    r = run("forget", str(ids["ep"]), input="n\n")
    assert r.exit_code == 0 and "left alone" in r.output
    assert c.get(ids["ep"]).status == "active"

    r = run("forget", str(ids["ep"]), input="y\n")
    assert r.exit_code == 0 and "archived 1" in r.output
    assert c.get(ids["ep"]).status == "archived"


def test_forget_multiple_ids_with_yes(isolated):
    c, ids = isolated
    r = run("forget", str(ids["ep"]), str(ids["h2"]), "--yes")
    assert "archived 2" in r.output
    assert c.count(include_archived=False) == 1


def test_forget_by_query_archives_best_match(isolated):
    c, ids = isolated
    r = run("forget", "--query", "H2 postgres tests", "--yes")
    assert r.exit_code == 0 and "archived 1" in r.output
    assert c.get(ids["h2"]).status == "archived"
    assert c.get(ids["ruff"]).status == "active"


def test_forget_unknown_id_reports_it(isolated):
    r = run("forget", "424242", "--yes")
    assert r.exit_code == 1 and "no memory #424242" in r.output


def test_forget_requires_exactly_one_selector(isolated):
    _, ids = isolated
    assert run("forget").exit_code == 2
    assert run("forget", str(ids["ep"]), "--query", "x").exit_code == 2
