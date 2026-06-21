"""Deterministic test of the auto-recall hook: it surfaces relevant memories and
skills as injected context, without needing a live agent."""

import anyio

from relife import hooks
from relife.memory import events as ev
from relife.memory import skills as sk
from relife.memory import store
from relife.memory import workflows as wf


def _isolate(tmp_path):
    store._DB_PATH = tmp_path / "relife.db"
    ev._DB_PATH = tmp_path / "relife.db"
    sk._SKILLS_DIR = tmp_path / "skills"
    wf._WORKFLOWS_DIR = tmp_path / "workflows"
    store.init_db()


def test_hook_injects_memory_and_skill(tmp_path):
    _isolate(tmp_path)
    store.save("The user prefers the ruff linter for Python.", kind="preference", tags="python,ruff")
    sk.write_skill(
        "scaffold-python-cli",
        "Setting up a new Python command-line project.",
        "1. pyproject  2. typer entry  3. pip install -e .",
    )

    out = anyio.run(lambda: hooks._recall_hook({"prompt": "scaffold a new python cli project"}, None, None))
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "ruff" in ctx.lower()                      # memory surfaced
    assert "scaffold-python-cli" in ctx               # skill surfaced
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"


def test_hook_silent_when_irrelevant(tmp_path):
    _isolate(tmp_path)
    store.save("The user prefers ruff.", kind="preference", tags="python")
    out = anyio.run(lambda: hooks._recall_hook({"prompt": "what time is it in tokyo"}, None, None))
    assert out == {}


def test_hook_surfaces_workflow(tmp_path):
    _isolate(tmp_path)
    wf.write_workflow(
        "ship-new-service",
        "Standing up and publishing a brand-new service.",
        "1. scaffold  2. test  3. repo  4. push",
        trigger="scaffold,service,repo,push",
    )
    out = anyio.run(lambda: hooks._recall_hook({"prompt": "ship a new service to a repo"}, None, None))
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "ship-new-service" in ctx


def test_event_hook_logs_tool_use(tmp_path):
    _isolate(tmp_path)
    inp = {"tool_name": "Edit", "tool_input": {"file_path": "store.py"}, "session_id": "s1"}
    anyio.run(lambda: hooks._event_hook(inp, None, None))
    assert ev.count() == 1
    assert ev.recent_events()[0].tool == "Edit"
