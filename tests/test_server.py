"""Tests for the always-on agent server.

Two layers, both deterministic and model-free (no ``ClaudeSDKClient`` is ever
opened — sessions are injected fakes, mirroring how ``rem`` injects ``ask_model``):

1. **Session logic** (`anyio.run`, no HTTP) — the real ``AgentSession`` pubsub +
   ``ApprovalBroker``: approve flow, timeout→deny flow, and backlog replay.
2. **HTTP layer** (FastAPI ``TestClient``) — routing, validation, auth, 404s, and
   that the real UI is served — driven by a recording fake session.

The server deps (fastapi) are an optional extra, so HTTP tests skip cleanly when
absent, mirroring the memory-daemon suite.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import anyio
import pytest

from relife.server.session import AgentSession


# =============================================================================
# Layer 1 — session logic (real pubsub + ApprovalBroker), no HTTP, no model
# =============================================================================
class ScriptedSession(AgentSession):
    """An AgentSession whose worker emits a fixed sequence instead of driving a
    model — but reuses the real broker / pubsub / ring so the approval + replay
    machinery under test is the production code."""

    def __init__(self, workspace: Path, **kw) -> None:
        super().__init__(workspace, **kw)
        self.timeout = 0.5  # per-approval wait; tests shorten it for the timeout case

    async def start(self) -> None:  # no ClaudeSDKClient
        self._worker = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            turn = await self._inbound.get()
            await self._publish({"type": "user", "text": turn})
            await self._publish({"type": "tool_use", "name": "Bash", "brief": "gh pr create"})
            approved = await self._broker.request(
                "Bash", {"command": "gh pr create -t x"}, "outward-facing", timeout=self.timeout
            )
            await self._publish(
                {"type": "tool_result", "brief": "created" if approved else "skipped"}
            )
            await self._publish({"type": "result", "cost_usd": 0.0123})


async def _collect_until(q, target, limit=25):
    got = []
    for _ in range(limit):
        _sid, ev = await asyncio.wait_for(q.get(), 2.0)
        got.append(ev)
        if ev["type"] == target:
            return got
    raise AssertionError(f"never saw event {target!r}; got {[e['type'] for e in got]}")


def test_approval_approve_flow():
    async def flow():
        s = ScriptedSession(Path("."))
        await s.start()
        q, backlog = s.subscribe()
        assert backlog == []
        await s.submit("open a PR")

        got = await _collect_until(q, "approval_request")
        req = got[-1]
        assert req["tool"] == "Bash" and req["brief"] == "gh pr create -t x"
        assert s.resolve_approval(req["approval_id"], True) is True

        rest = await _collect_until(q, "result")
        types = [e["type"] for e in rest]
        assert "approval_resolved" in types
        resolved = next(e for e in rest if e["type"] == "approval_resolved")
        assert resolved["approved"] is True
        tr = next(e for e in rest if e["type"] == "tool_result")
        assert tr["brief"] == "created"
        await s.aclose()

    anyio.run(flow)


def test_approval_timeout_denies():
    async def flow():
        s = ScriptedSession(Path("."))
        s.timeout = 0.05  # no one approves → broker times out → deny
        await s.start()
        q, _ = s.subscribe()
        await s.submit("open a PR")

        rest = await _collect_until(q, "result")
        resolved = next(e for e in rest if e["type"] == "approval_resolved")
        assert resolved["approved"] is False
        tr = next(e for e in rest if e["type"] == "tool_result")
        assert tr["brief"] == "skipped"
        await s.aclose()

    anyio.run(flow)


def test_resolve_unknown_approval_is_false():
    async def flow():
        s = ScriptedSession(Path("."))
        await s.start()
        assert s.resolve_approval("does-not-exist", True) is False
        await s.aclose()

    anyio.run(flow)


def test_backlog_replay_by_last_id():
    async def flow():
        s = ScriptedSession(Path("."))
        await s.start()
        q1, _ = s.subscribe()
        await s.submit("hello")
        got = await _collect_until(q1, "approval_request")
        s.resolve_approval(got[-1]["approval_id"], False)
        await _collect_until(q1, "result")

        # A fresh subscriber from the start replays the whole turn.
        _q2, backlog = s.subscribe(last_id=0)
        assert any(e["type"] == "user" for _sid, e in backlog)
        assert any(e["type"] == "result" for _sid, e in backlog)

        # A subscriber caught up to the latest id gets an empty backlog.
        _q3, backlog3 = s.subscribe(last_id=s._seq)
        assert backlog3 == []
        await s.aclose()

    anyio.run(flow)


# =============================================================================
# Layer 2 — HTTP layer (TestClient) with a recording fake session
# =============================================================================
class RecordingSession:
    """Records calls the HTTP handlers make; no worker, no model."""

    def __init__(self, workspace: Path, session_id: str = "sess-1") -> None:
        self.id = session_id
        self.workspace = workspace
        self.submitted: list[str] = []
        self.resolved: list[tuple[str, bool]] = []
        self.resolve_return = True
        self.closed = False

    async def start(self) -> None:
        pass

    async def aclose(self) -> None:
        self.closed = True

    async def submit(self, text: str) -> None:
        self.submitted.append(text)

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        self.resolved.append((approval_id, approved))
        return self.resolve_return

    def subscribe(self, last_id=None):
        return asyncio.Queue(), []

    def unsubscribe(self, q) -> None:
        pass


@pytest.fixture
def http():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from relife.server.app import create_app

    created: list[RecordingSession] = []

    def factory(workspace: Path) -> RecordingSession:
        s = RecordingSession(workspace)
        created.append(s)
        return s

    def make(token: str | None = None):
        app = create_app(token=token, session_factory=factory)
        return TestClient(app), created

    return make


def test_health_and_root_need_no_auth(http):
    tc, _ = http(token="secret")
    r = tc.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    home = tc.get("/")
    assert home.status_code == 200 and "agent console" in home.text.lower()


def test_create_session_and_submit(http):
    tc, created = http()
    r = tc.post("/sessions", json={})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert sid == "sess-1"

    r2 = tc.post(f"/sessions/{sid}/messages", json={"text": "do the thing"})
    assert r2.status_code == 200 and r2.json() == {"ok": True}
    assert created[0].submitted == ["do the thing"]


def test_empty_message_rejected(http):
    tc, _ = http()
    sid = tc.post("/sessions", json={}).json()["session_id"]
    r = tc.post(f"/sessions/{sid}/messages", json={"text": "   "})
    assert r.status_code == 400


def test_unknown_session_is_404(http):
    tc, _ = http()
    assert tc.post("/sessions/nope/messages", json={"text": "hi"}).status_code == 404
    assert tc.get("/sessions/nope/events").status_code == 404
    assert tc.get("/sessions/nope").status_code == 404


def test_get_session_lets_a_reloading_ui_reattach(http):
    """Without this probe the UI cannot tell a live session from a dead one, so
    every reload created a second agent and orphaned the first."""
    tc, created = http()
    sid = tc.post("/sessions", json={}).json()["session_id"]

    r = tc.get(f"/sessions/{sid}")
    assert r.status_code == 200 and r.json()["session_id"] == sid
    assert len(created) == 1  # probing must not spawn anything

    tc.delete(f"/sessions/{sid}")
    assert tc.get(f"/sessions/{sid}").status_code == 404


def test_get_session_requires_auth(http):
    tc, _ = http(token="secret")
    tc.post("/auth", json={"token": "secret"})
    sid = tc.post("/sessions", json={}).json()["session_id"]
    tc.post("/auth/logout")
    tc.cookies.clear()
    assert tc.get(f"/sessions/{sid}").status_code == 401


def test_approval_endpoint_routes_decision(http):
    tc, created = http()
    sid = tc.post("/sessions", json={}).json()["session_id"]
    r = tc.post(f"/sessions/{sid}/approvals/abc123", json={"decision": "allow"})
    assert r.status_code == 200 and r.json() == {"resolved": True}
    assert created[0].resolved == [("abc123", True)]

    tc.post(f"/sessions/{sid}/approvals/def456", json={"decision": "deny"})
    assert created[0].resolved[-1] == ("def456", False)


def test_approval_bad_decision_rejected(http):
    tc, _ = http()
    sid = tc.post("/sessions", json={}).json()["session_id"]
    r = tc.post(f"/sessions/{sid}/approvals/x", json={"decision": "maybe"})
    assert r.status_code == 400


def test_auth_required_when_token_set(http):
    tc, _ = http(token="secret")
    assert tc.post("/sessions", json={}).status_code == 401
    ok = tc.post("/sessions", json={}, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
