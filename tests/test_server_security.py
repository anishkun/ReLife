"""Hardening tests for the always-on agent server.

Three layers, all deterministic and model-free (no ``ClaudeSDKClient`` is ever
opened — same discipline as ``tests/test_server.py``):

1. **Policy** — the pure functions in ``relife/server/security.py``: token
   comparison, CSRF origin matching, the fail-closed bind guard, and workspace
   confinement.
2. **Resource ceilings** — the real ``AgentSession``/``SessionManager``: bounded
   turn queue, subscriber cap, drop-oldest on a lagging subscriber, session cap,
   idle reaping.
3. **HTTP** — cookie auth (the carrier the browser's ``EventSource`` can use),
   the auth throttle, CSRF refusal, confinement, and the 4xx/429 mapping.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import anyio
import pytest

from relife.server import security
from relife.server.session import (
    AgentSession,
    SessionLimitReached,
    SessionManager,
    TooManySubscribers,
    TurnQueueFull,
)


# =============================================================================
# Layer 1 — pure policy
# =============================================================================
def test_token_matches_is_permissive_only_when_auth_is_off():
    assert security.token_matches(None, None) is True  # auth disabled
    assert security.token_matches("s3cret", "s3cret") is True
    assert security.token_matches("s3cret", "wrong") is False
    assert security.token_matches("s3cret", None) is False
    assert security.token_matches("s3cret", "") is False


def test_presented_token_reads_either_carrier():
    assert security.presented_token("Bearer abc", None) == "abc"
    assert security.presented_token("bearer abc", None) == "abc"  # scheme is case-insensitive
    assert security.presented_token(None, "cookie-tok") == "cookie-tok"
    # The header wins when both are present, and a non-bearer scheme is ignored.
    assert security.presented_token("Bearer hdr", "cookie") == "hdr"
    assert security.presented_token("Basic xyz", "cookie") == "cookie"
    assert security.presented_token(None, None) is None


def test_same_origin_blocks_cross_site_but_allows_header_clients():
    assert security.same_origin("http://127.0.0.1:8600", "127.0.0.1:8600") is True
    assert security.same_origin("https://evil.example", "127.0.0.1:8600") is False
    # A different port is a different origin.
    assert security.same_origin("http://127.0.0.1:9999", "127.0.0.1:8600") is False
    # No Origin at all = a non-browser client (curl/SDK), which carries a bearer
    # header rather than an ambient cookie a third-party page could ride.
    assert security.same_origin(None, "127.0.0.1:8600") is True


def test_guard_bind_fails_closed_off_loopback():
    security.guard_bind("127.0.0.1", None)  # loopback, no token: fine
    security.guard_bind("localhost", None)
    security.guard_bind("::1", None)
    security.guard_bind("0.0.0.0", "a-token")  # reachable but authenticated
    with pytest.raises(ValueError, match="refusing to bind"):
        security.guard_bind("0.0.0.0", None)
    with pytest.raises(ValueError):
        security.guard_bind("192.168.1.20", None)


def test_resolve_workspace_confines_to_root(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    assert security.resolve_workspace(None, root) == root.resolve()
    assert security.resolve_workspace("   ", root) == root.resolve()
    assert security.resolve_workspace("proj", root) == (root / "proj").resolve()
    assert security.resolve_workspace(str(root / "proj"), root) == (root / "proj").resolve()

    # The whole point: the request body must not be able to widen the auto-allow
    # blast radius, whether by traversal or by an absolute path elsewhere.
    for escape in ["..", "../..", "proj/../..", str(tmp_path / "elsewhere")]:
        with pytest.raises(ValueError, match="must be inside"):
            security.resolve_workspace(escape, root)


def test_attempt_limiter_windows_per_key():
    lim = security.AttemptLimiter(max_attempts=3, window=60)
    assert [lim.allow("a", now=0) for _ in range(4)] == [True, True, True, False]
    assert lim.allow("b", now=0) is True  # other keys unaffected
    assert lim.allow("a", now=61) is True  # window rolled over
    lim2 = security.AttemptLimiter(max_attempts=1, window=60)
    assert lim2.allow("a", now=0) is True and lim2.allow("a", now=1) is False
    lim2.reset("a")  # a success clears the count
    assert lim2.allow("a", now=1) is True


# =============================================================================
# Layer 2 — resource ceilings on the real session machinery
# =============================================================================
class QuietSession(AgentSession):
    """A real session minus the model: no ClaudeSDKClient, no worker draining."""

    async def start(self) -> None:
        pass


def test_turn_queue_is_bounded(monkeypatch):
    from relife import config

    monkeypatch.setattr(config, "AGENT_MAX_QUEUED_TURNS", 2)

    async def flow():
        s = QuietSession(Path("."))
        await s.submit("one")
        await s.submit("two")
        with pytest.raises(TurnQueueFull):
            await s.submit("three")  # nothing is draining it

    anyio.run(flow)


def test_subscriber_count_is_capped(monkeypatch):
    from relife import config

    monkeypatch.setattr(config, "AGENT_MAX_SUBSCRIBERS", 2)

    async def flow():
        s = QuietSession(Path("."))
        q1, _ = s.subscribe()
        s.subscribe()
        with pytest.raises(TooManySubscribers):
            s.subscribe()
        s.unsubscribe(q1)
        s.subscribe()  # freeing one makes room again

    anyio.run(flow)


def test_lagging_subscriber_drops_oldest_instead_of_growing(monkeypatch):
    from relife.server import session as session_mod

    monkeypatch.setattr(session_mod, "_SUB_QUEUE_MAX", 3)

    async def flow():
        s = QuietSession(Path("."))
        q, _ = s.subscribe()
        for i in range(6):
            await s._publish({"type": "text", "text": str(i)})
        assert q.qsize() == 3  # bounded, publish never blocked
        seen = [q.get_nowait()[1]["text"] for _ in range(3)]
        assert seen == ["3", "4", "5"]  # oldest dropped, newest kept
        # The ring still holds the full history, so a reconnect replays it.
        _q2, backlog = s.subscribe(last_id=0)
        assert [e["text"] for _sid, e in backlog] == ["0", "1", "2", "3", "4", "5"]

    anyio.run(flow)


def test_streaming_a_turn_defers_reaping():
    """A long turn with no browser attached must not be reaped mid-run.

    The reaper spares sessions with a live SSE subscriber, but a programmatic
    client can submit a turn and never stream it — so every published event
    stamps activity too.
    """

    async def flow():
        s = QuietSession(Path("."))
        s.last_active = 0.0  # as if idle since process start
        await s._publish({"type": "text", "text": "still working"})
        assert s.last_active > 0.0

        mgr = SessionManager(session_factory=lambda ws: s, idle_timeout=10.0)
        await mgr.create(Path("."))
        assert await mgr.reap_idle() == []  # activity is fresh -> spared

    anyio.run(flow)


class FakeSession:
    """Minimal session double for manager-level lifecycle tests."""

    def __init__(self, workspace: Path, sid: str) -> None:
        self.id = sid
        self.workspace = workspace
        self.last_active = 0.0
        self.subscriber_count = 0
        self.closed = False

    async def start(self) -> None:
        pass

    async def aclose(self) -> None:
        self.closed = True

    def touch(self) -> None:
        self.last_active = time.monotonic()


def _manager(**kw) -> tuple[SessionManager, list[FakeSession]]:
    made: list[FakeSession] = []

    def factory(ws: Path) -> FakeSession:
        s = FakeSession(ws, f"s{len(made)}")
        made.append(s)
        return s

    return SessionManager(session_factory=factory, **kw), made


def test_session_cap_rejects_beyond_the_limit():
    async def flow():
        mgr, _ = _manager(max_sessions=2, idle_timeout=0)
        await mgr.create(Path("."))
        await mgr.create(Path("."))
        with pytest.raises(SessionLimitReached):
            await mgr.create(Path("."))
        assert mgr.count() == 2

    anyio.run(flow)


def test_idle_sessions_are_reaped_and_closed():
    async def flow():
        mgr, made = _manager(max_sessions=4, idle_timeout=10.0)
        await mgr.create(Path("."))
        await mgr.create(Path("."))
        stale, watched = made
        stale.last_active = 0.0  # far in the past (monotonic starts well above 0)
        watched.last_active = 0.0
        watched.subscriber_count = 1  # a browser is streaming this one

        closed = await mgr.reap_idle()
        assert closed == [stale.id]
        assert stale.closed is True and watched.closed is False
        assert mgr.get(stale.id) is None and mgr.count() == 1

        # Reaping frees room under the cap.
        mgr.max_sessions = 2
        await mgr.create(Path("."))
        assert mgr.count() == 2

    anyio.run(flow)


def test_close_and_aclose_release_sessions():
    async def flow():
        mgr, made = _manager(max_sessions=4, idle_timeout=0)
        s = await mgr.create(Path("."))
        assert await mgr.close(s.id) is True
        assert await mgr.close(s.id) is False  # already gone
        assert made[0].closed is True

        await mgr.create(Path("."))
        await mgr.aclose()  # server shutdown must not orphan subprocesses
        assert mgr.count() == 0 and made[1].closed is True

    anyio.run(flow)


# =============================================================================
# Layer 3 — HTTP surface
# =============================================================================
class RecordingSession:
    def __init__(self, workspace: Path, session_id: str) -> None:
        self.id = session_id
        self.workspace = workspace
        self.submitted: list[str] = []
        self.last_active = time.monotonic()
        self.subscriber_count = 0

    async def start(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    def touch(self) -> None:
        self.last_active = time.monotonic()

    async def submit(self, text: str) -> None:
        self.submitted.append(text)

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        return True

    def subscribe(self, last_id=None):
        return asyncio.Queue(), []

    def unsubscribe(self, q) -> None:
        pass


@pytest.fixture
def http(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from relife.server.app import create_app

    root = tmp_path / "workspace"
    root.mkdir()
    created: list[RecordingSession] = []

    def factory(workspace: Path) -> RecordingSession:
        s = RecordingSession(workspace, f"sess-{len(created)}")
        created.append(s)
        return s

    def make(token: str | None = None):
        app = create_app(
            token=token,
            session_factory=factory,
            workspace_root=root,
            reap=False,
            schedules_path=tmp_path / "schedules.json",
            run_scheduler=False,
        )
        return TestClient(app), created, root

    return make


def test_cookie_auth_unlocks_the_ui_including_the_event_stream(http):
    tc, _created, _root = http(token="s3cret")
    # No credentials: every route the UI needs is closed — the SSE stream too,
    # which is the route a browser cannot send a bearer header on.
    assert tc.post("/sessions", json={}).status_code == 401
    assert tc.get("/sessions/sess-0/events").status_code == 401

    assert tc.post("/auth", json={"token": "nope"}).status_code == 401
    r = tc.post("/auth", json={"token": "s3cret"})
    assert r.status_code == 200 and r.json()["auth_required"] is True
    from relife import config

    assert config.AGENT_COOKIE in tc.cookies
    # The TestClient now carries the cookie the browser would — no header.
    assert tc.post("/sessions", json={}).status_code == 200
    # ...and the stream is no longer 401 (404 = auth passed, session unknown).
    assert tc.get("/sessions/no-such-session/events").status_code == 404

    assert tc.post("/auth/logout").status_code == 200
    assert tc.post("/sessions", json={}).status_code == 401


def test_auth_status_tells_the_ui_whether_to_prompt(http):
    tc, _c, _r = http(token="s3cret")
    assert tc.get("/auth/status").json() == {"auth_required": True, "authenticated": False}
    tc.post("/auth", json={"token": "s3cret"})
    assert tc.get("/auth/status").json() == {"auth_required": True, "authenticated": True}

    tc2, _c2, _r2 = http()
    assert tc2.get("/auth/status").json() == {"auth_required": False, "authenticated": True}


def test_auth_attempts_are_throttled(http, monkeypatch):
    from relife import config

    monkeypatch.setattr(config, "AGENT_AUTH_MAX_ATTEMPTS", 3)
    tc, _c, _r = http(token="s3cret")
    codes = [tc.post("/auth", json={"token": "guess"}).status_code for _ in range(4)]
    assert codes == [401, 401, 401, 429]
    # Throttled means throttled: even the right token waits out the window.
    assert tc.post("/auth", json={"token": "s3cret"}).status_code == 429


def test_cross_origin_mutations_are_refused(http):
    tc, _c, _r = http()
    evil = {"origin": "https://evil.example"}
    assert tc.post("/sessions", json={}, headers=evil).status_code == 403
    assert tc.post("/auth", json={"token": "x"}, headers=evil).status_code == 403

    sid = tc.post("/sessions", json={}).json()["session_id"]
    assert tc.post(f"/sessions/{sid}/messages", json={"text": "hi"}, headers=evil).status_code == 403
    assert tc.post(f"/sessions/{sid}/approvals/a1", json={"decision": "allow"}, headers=evil).status_code == 403

    # Same-origin is fine (Host is testserver under the TestClient).
    ok = {"origin": "http://testserver"}
    assert tc.post(f"/sessions/{sid}/messages", json={"text": "hi"}, headers=ok).status_code == 200


def test_session_workspace_is_confined_to_the_root(http):
    tc, created, root = http()
    r = tc.post("/sessions", json={"workspace": "proj-a"})
    assert r.status_code == 200
    assert Path(r.json()["workspace"]) == (root / "proj-a").resolve()
    assert created[-1].workspace == (root / "proj-a").resolve()

    # An escape attempt must not create a session at all — the workspace decides
    # what the permission policy auto-allows writes to.
    before = len(created)
    for escape in ["../..", "..", "C:\\" if Path("C:\\").exists() else "/"]:
        r = tc.post("/sessions", json={"workspace": escape})
        assert r.status_code == 400, escape
    assert len(created) == before


def test_session_cap_returns_429(http, monkeypatch):
    from relife import config

    monkeypatch.setattr(config, "AGENT_MAX_SESSIONS", 2)
    tc, _c, _r = http()
    assert tc.post("/sessions", json={}).status_code == 200
    assert tc.post("/sessions", json={}).status_code == 200
    assert tc.post("/sessions", json={}).status_code == 429


def test_oversized_message_rejected(http, monkeypatch):
    from relife import config

    monkeypatch.setattr(config, "AGENT_MAX_MESSAGE_CHARS", 32)
    tc, created, _r = http()
    sid = tc.post("/sessions", json={}).json()["session_id"]
    assert tc.post(f"/sessions/{sid}/messages", json={"text": "x" * 33}).status_code == 413
    assert tc.post(f"/sessions/{sid}/messages", json={"text": "x" * 32}).status_code == 200
    assert created[0].submitted == ["x" * 32]


def test_saturated_turn_queue_returns_429(http):
    tc, created, _r = http()
    sid = tc.post("/sessions", json={}).json()["session_id"]

    async def full(text: str) -> None:
        raise TurnQueueFull("too many turns queued for this session")

    created[0].submit = full
    assert tc.post(f"/sessions/{sid}/messages", json={"text": "hi"}).status_code == 429


def test_session_can_be_closed(http):
    tc, _c, _r = http()
    sid = tc.post("/sessions", json={}).json()["session_id"]
    assert tc.delete(f"/sessions/{sid}").json() == {"closed": True}
    assert tc.delete(f"/sessions/{sid}").status_code == 404
    assert tc.post(f"/sessions/{sid}/messages", json={"text": "hi"}).status_code == 404


def test_health_reports_auth_mode_without_auth(http):
    tc, _c, _r = http(token="s3cret")
    body = tc.get("/health").json()
    assert body["status"] == "ok" and body["auth_required"] is True and body["sessions"] == 0
