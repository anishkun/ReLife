"""Deterministic test of the auto-recall hook: it surfaces relevant memories and
skills as injected context, without needing a live agent."""

import anyio

from relife import hooks
from relife.memory import skills as sk
from relife.memory import store


def _isolate(tmp_path):
    store._DB_PATH = tmp_path / "relife.db"
    sk._SKILLS_DIR = tmp_path / "skills"
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
