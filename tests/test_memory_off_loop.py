"""Memory work must not run on the event loop.

Under ``relife serve`` one loop carries every session's SSE stream and every
pending approval, and the memory hooks + memory MCP tools are async callers of
a *sync* client. Each test drives the real hook / tool against a deliberately
slow fake client while a 10ms ticker runs on the same loop: if the call ran
inline the ticker would freeze for its whole duration.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from relife import hooks
from relife.memory import server as mem_server

SLOW = 0.2  # seconds each fake client call blocks its thread


class SlowClient:
    """Every call blocks (time.sleep — a real block, not an await) and records
    which thread it ran on."""

    def __init__(self):
        self.threads: list[int] = []
        self.saved: list[str] = []
        self.logged: list[str] = []

    def _block(self):
        self.threads.append(threading.get_ident())
        time.sleep(SLOW)

    def recall(self, query, k=5, *, reinforce=False, include_archived=False):
        self._block()
        return [SimpleNamespace(kind="fact", text="tests run against H2", tags="")]

    def skill_find(self, query, k=3):
        self._block()
        return []

    def workflow_find(self, query, k=3):
        self._block()
        return []

    def events_for_task(self, task_id):
        self._block()
        return [SimpleNamespace(id=i, tool=t) for i, t in enumerate(["Read", "Edit", "Bash", "Bash"], 1)]

    def log_event(self, tool, brief, task_id=""):
        self._block()
        self.logged.append(tool)

    def save(self, text, kind="fact", tags="", importance=None):
        self._block()
        self.saved.append(text)
        return 1

    def consolidate(self):
        self._block()
        return SimpleNamespace(summary=lambda: "nothing to do", workflows_created=[], patterns=[])


async def _with_ticker(coro):
    """Await ``coro`` while counting 10ms ticks on the same loop."""
    ticks = 0
    done = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not done.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    t = asyncio.create_task(ticker())
    try:
        result = await coro
    finally:
        done.set()
        await t
    return result, ticks


@pytest.fixture
def slow(monkeypatch):
    client = SlowClient()
    monkeypatch.setattr(hooks, "default_client", lambda: client)
    monkeypatch.setattr(mem_server, "default_client", lambda: client)
    hooks._last_prompt.clear()
    yield client
    hooks._last_prompt.clear()


def _assert_off_loop(client, ticks, calls):
    main = threading.get_ident()
    assert client.threads and all(t != main for t in client.threads)
    # `calls` blocking calls of SLOW each; inline, the ticker would get ~0 ticks.
    assert ticks >= int(calls * SLOW / 0.01) // 2, ticks


def test_recall_hook_runs_off_loop(slow):
    async def flow():
        return await _with_ticker(hooks._recall_hook({"prompt": "how do tests run", "session_id": "s1"}, None, None))

    out, ticks = asyncio.run(flow())
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "tests run against H2" in ctx
    assert hooks._last_prompt["s1"] == ("how do tests run", 4)  # watermark still recorded
    _assert_off_loop(slow, ticks, calls=4)  # watermark + recall + skills + workflows


def test_event_hook_runs_off_loop(slow):
    async def flow():
        return await _with_ticker(
            hooks._event_hook({"tool_name": "Bash", "tool_input": {"command": "pytest"}, "session_id": "s1"}, None, None)
        )

    out, ticks = asyncio.run(flow())
    assert out == {} and slow.logged == ["Bash"]
    _assert_off_loop(slow, ticks, calls=1)


def test_episode_hook_runs_off_loop(slow):
    hooks._last_prompt["s1"] = ("fix the flaky test", 0)

    async def flow():
        return await _with_ticker(hooks._episode_hook({"session_id": "s1"}, None, None))

    _, ticks = asyncio.run(flow())
    assert slow.saved and slow.saved[0].startswith("Task: fix the flaky test")
    _assert_off_loop(slow, ticks, calls=2)  # read the journal + save the episode


@pytest.mark.parametrize(
    "tool, args, expect",
    [
        (mem_server.memory_recall, {"query": "tests"}, "tests run against H2"),
        (mem_server.memory_save, {"text": "use ruff"}, "Saved memory #1"),
        (mem_server.memory_consolidate, {}, "nothing to do"),
    ],
)
def test_memory_tools_run_off_loop(slow, tool, args, expect):
    async def flow():
        return await _with_ticker(tool.handler(args))

    out, ticks = asyncio.run(flow())
    assert expect in out["content"][0]["text"]
    _assert_off_loop(slow, ticks, calls=1)
