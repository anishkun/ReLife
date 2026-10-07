"""Release-hardening tests: paths that were uncovered before the first release.

- ``config.resolve_home`` — an installed wheel must not keep state in
  ``site-packages`` (an upgrade would wipe every memory).
- ``make_permission_callback`` — the CLI's TTY approval gate: deny when
  non-interactive, deny on EOF, only an explicit yes allows.
- the memory MCP tools — round-trip against an isolated store, and fail soft
  (``is_error``) instead of raising when the memory backend is down.
"""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest
from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

from relife import config, permissions
from relife.memory import client as client_mod
from relife.memory import server as mem_server
from relife.memory import skills, store, workflows

# --- resolve_home -----------------------------------------------------------


def test_home_env_override_wins(tmp_path):
    pkg = tmp_path / "src" / "relife"
    pkg.mkdir(parents=True)
    (pkg.parent / "pyproject.toml").write_text("[project]\n")
    assert config.resolve_home(pkg, {"RELIFE_HOME": str(tmp_path / "h")}) == (tmp_path / "h").resolve()


def test_home_source_checkout_keeps_state_beside_code(tmp_path):
    pkg = tmp_path / "relife"
    pkg.mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    assert config.resolve_home(pkg, {}) == tmp_path


def test_home_installed_wheel_uses_user_dir_not_site_packages(tmp_path):
    pkg = tmp_path / "site-packages" / "relife"
    pkg.mkdir(parents=True)  # no pyproject.toml beside it
    home = config.resolve_home(pkg, {})
    assert home == Path.home() / ".relife"
    assert "site-packages" not in str(home)


@pytest.mark.skipif(
    not (config.PACKAGE_DIR.parent / "pyproject.toml").is_file(),
    reason="running against an installed wheel, not a source checkout",
)
def test_this_checkout_resolves_to_repo_root():
    # Back-compat: the editable install keeps using <repo>/data.
    assert config.PROJECT_ROOT == config.PACKAGE_DIR.parent


def test_installed_wheel_keeps_state_out_of_site_packages():
    if (config.PACKAGE_DIR.parent / "pyproject.toml").is_file():
        pytest.skip("source checkout")
    assert "site-packages" not in str(config.DATA_DIR)


# --- TTY permission callback ------------------------------------------------

WS = Path("/tmp/relife-ws").resolve()
OUTWARD = ("Bash", {"command": "curl -X POST https://example.com -d x"})


def _run(cb, name, inp):
    return anyio.run(cb, name, inp, None)


def test_tty_allow_case_needs_no_prompt(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("prompted on an allow-case"))
    cb = permissions.make_permission_callback(WS, interactive=True)
    assert isinstance(_run(cb, "Read", {"file_path": "/etc/hosts"}), PermissionResultAllow)


def test_tty_non_interactive_denies_without_prompting(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("prompted when non-interactive"))
    cb = permissions.make_permission_callback(WS, interactive=False)
    assert isinstance(_run(cb, *OUTWARD), PermissionResultDeny)


@pytest.mark.parametrize("answer,allowed", [
    ("y", True), ("YES", True), (" yes ", True),
    ("", False), ("n", False), ("yep", False), ("sure", False),
])
def test_tty_only_explicit_yes_allows(monkeypatch, answer, allowed):
    monkeypatch.setattr("builtins.input", lambda *_: answer)
    cb = permissions.make_permission_callback(WS, interactive=True)
    res = _run(cb, *OUTWARD)
    assert isinstance(res, PermissionResultAllow if allowed else PermissionResultDeny)


@pytest.mark.parametrize("exc", [EOFError, KeyboardInterrupt])
def test_tty_unreadable_prompt_fails_closed(monkeypatch, exc):
    def boom(*_):
        raise exc

    monkeypatch.setattr("builtins.input", boom)
    cb = permissions.make_permission_callback(WS, interactive=True)
    assert isinstance(_run(cb, *OUTWARD), PermissionResultDeny)


# --- memory MCP tools ---------------------------------------------------------


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_DB_PATH", tmp_path / "relife.db")
    monkeypatch.setattr(skills, "_SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(workflows, "_WORKFLOWS_DIR", tmp_path / "workflows")
    monkeypatch.setattr(client_mod, "_default", None)
    yield
    client_mod._default = None


def call(t, args):
    return anyio.run(t.handler, args)


def text(res):
    return res["content"][0]["text"]


def test_mcp_memory_round_trip(mem):
    r = call(mem_server.memory_save, {"text": "Deploys go through the staging cluster first.",
                                      "kind": "fact", "tags": "deploy"})
    assert "Saved memory #" in text(r) and not r.get("is_error")
    assert "staging cluster" in text(call(mem_server.memory_recall, {"query": "deploy staging"}))
    assert "Archived" in text(call(mem_server.memory_forget, {"query": "deploy staging cluster"}))
    assert text(call(mem_server.memory_recall, {"query": "deploy staging"})) == "(no relevant memories)"


def test_mcp_skill_and_workflow_round_trip(mem):
    r = call(mem_server.skill_write, {"name": "scaffold-cli", "when_to_use": "new python cli",
                                      "steps": "1. uv init\n2. add typer"})
    assert "scaffold-cli" in text(r)
    assert "add typer" in text(call(mem_server.skill_find, {"query": "python cli scaffold"}))
    r = call(mem_server.workflow_save, {"name": "ship-service", "when_to_use": "release a service",
                                        "steps": "test → tag → push", "trigger": "release"})
    assert "ship-service" in text(r)
    assert "tag" in text(call(mem_server.workflow_find, {"query": "release service"}))


@pytest.mark.parametrize("k", ["five", None, -3, 10_000])
def test_mcp_junk_k_does_not_raise(mem, k):
    call(mem_server.memory_save, {"text": "ruff is the linter here", "kind": "preference"})
    r = call(mem_server.memory_recall, {"query": "ruff linter", "k": k})
    assert not r.get("is_error")


class _Down:
    def __getattr__(self, name):
        def fail(*a, **kw):
            raise ConnectionError("memory daemon unreachable")
        return fail


@pytest.mark.parametrize("tool,args", [
    (mem_server.memory_save, {"text": "x"}),
    (mem_server.memory_recall, {"query": "x"}),
    (mem_server.memory_forget, {"query": "x"}),
    (mem_server.skill_write, {"name": "a", "when_to_use": "b", "steps": "c"}),
    (mem_server.skill_find, {"query": "x"}),
    (mem_server.workflow_save, {"name": "a", "when_to_use": "b", "steps": "c"}),
    (mem_server.workflow_find, {"query": "x"}),
    (mem_server.memory_consolidate, {}),
])
def test_mcp_tools_fail_soft_when_backend_down(monkeypatch, tool, args):
    monkeypatch.setattr(mem_server, "default_client", lambda: _Down())
    r = call(tool, args)
    assert r["is_error"] is True
    assert "unreachable" in text(r)


# --- DNS rebinding guard ------------------------------------------------------

from relife.server.security import host_allowed  # noqa: E402


@pytest.mark.parametrize("host,ok", [
    ("127.0.0.1:8600", True), ("localhost:8600", True), ("LOCALHOST", True),
    ("[::1]:8600", True), ("127.5.5.5", True),
    ("evil.example:8600", False), ("evil.example", False), ("192.168.1.10:8600", False),
    ("127.0.0.1.evil.example", False), ("localhost.evil.example:8600", False),
    ("", False), (None, False),
])
def test_tokenless_server_only_answers_loopback_hosts(host, ok):
    assert host_allowed(host, None) is ok


def test_extra_host_admitted_and_token_lifts_the_guard():
    assert host_allowed("relife.lan:8600", None, frozenset({"relife.lan"}))
    assert host_allowed("evil.example:8600", "s3cret")


def test_rebinding_request_refused_over_http(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from relife.server.app import create_app

    app = create_app(workspace_root=tmp_path, reap=False, schedules_path=tmp_path / "s.json",
                     runs_dir=tmp_path / "runs", run_scheduler=False, allowed_hosts=frozenset(),
                     session_factory=lambda ws: pytest.fail("rebinding page created a session"))
    with TestClient(app, base_url="http://127.0.0.1:8600") as c:
        assert c.get("/health").status_code == 200
        evil = {"Host": "evil.example:8600", "Origin": "http://evil.example:8600"}
        assert c.post("/sessions", json={}, headers=evil).status_code == 403
        assert c.get("/", headers=evil).status_code == 403
        assert c.get("/schedules", headers=evil).status_code == 403


# --- corrupt state is never silently overwritten ------------------------------

from relife.server.schedules import ScheduleStore  # noqa: E402


@pytest.mark.parametrize("content", ["{broken", "[1, 2]", '{"schedules": "nope"}'])
def test_unreadable_schedules_file_is_preserved_before_overwrite(tmp_path, content):
    path = tmp_path / "schedules.json"
    path.write_text(content, encoding="utf-8")
    store = ScheduleStore(path)
    assert store.count() == 0 and store.problem
    store.save()  # what the first schedule added in the UI would trigger
    backups = list(tmp_path.glob("schedules.json.corrupt-*"))
    assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == content
    assert "original kept as" in store.problem
    store.save()  # only once
    assert len(list(tmp_path.glob("schedules.json.corrupt-*"))) == 1


def test_one_bad_record_is_reported_not_silently_dropped(tmp_path):
    path = tmp_path / "schedules.json"
    path.write_text('{"version": 1, "schedules": [{"bogus": true}]}', encoding="utf-8")
    store = ScheduleStore(path)
    assert "1 unreadable" in store.problem


def test_healthy_schedules_file_has_no_problem(tmp_path):
    store = ScheduleStore(tmp_path / "schedules.json")
    store.save()
    assert ScheduleStore(tmp_path / "schedules.json").problem is None
    assert not list(tmp_path.glob("*.corrupt-*"))


def test_cli_explains_environment_failures(monkeypatch):
    import sqlite3

    from relife import cli

    assert "move it aside" in cli._friendly_error(sqlite3.DatabaseError("file is not a database"))
    monkeypatch.setattr(config, "MEMORY_URL", "http://127.0.0.1:8799")
    ConnectError = type("ConnectError", (Exception,), {})
    assert "relife memory serve" in cli._friendly_error(ConnectError("refused"))
    assert cli._friendly_error(RuntimeError("a real bug")) is None  # bugs keep their traceback


# --- recall: one common shared word is not relevance ---------------------------


def test_single_common_word_does_not_surface_unrelated_memories(tmp_path):
    s = store.MemoryStore(tmp_path / "r.db")
    # "write" shows up all over a real store (tool names in episodes/patterns).
    for i in range(8):
        s.save(f"Recurring tool sequence {i}: Edit then Write then test", kind="pattern")
    s.save("ApexPay data layer: Flyway migrations, outbox table, write-ahead idempotency keys",
           kind="fact")
    assert s.recall("write a python cli to convert temperatures") == []


def test_single_distinctive_word_still_recalls(tmp_path):
    s = store.MemoryStore(tmp_path / "r.db")
    for i in range(8):
        s.save(f"Recurring tool sequence {i}: Edit then Write then test", kind="pattern")
    s.save("ApexPay tests run against H2, not Postgres.", kind="fact")
    hits = s.recall("deploy the thing for apexpay please now quickly")
    assert hits and "ApexPay" in hits[0].text


def test_small_store_is_not_starved_by_the_common_word_rule(tmp_path):
    s = store.MemoryStore(tmp_path / "r.db")
    s.save("Write commit messages in the imperative mood.", kind="preference")
    assert s.recall("write the release notes")  # 1 of 1 rows, but below the min-docs floor


# --- approval brief shows the recipient --------------------------------------


def test_brief_puts_recipients_first_even_as_a_list():
    from relife.agent import _tool_brief

    # Gmail create_draft's real shape: subject/body before a list-valued `to`.
    brief = _tool_brief({"subject": "Digest", "body": "long " * 50,
                         "to": ["boss@example.com", "me@example.com"], "cc": []}, limit=400)
    assert brief.startswith("to=boss@example.com, me@example.com")
    assert "subject=Digest" in brief
    cal = _tool_brief({"summary": "Sync", "attendees": [{"email": "a@b.co"}]}, limit=400)
    assert cal.startswith("attendees=a@b.co")
