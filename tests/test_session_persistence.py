"""Sessions sleep instead of dying (``server/session.py``).

A session's ``claude`` subprocess starts only when a turn arrives; the idle
reaper puts a session to sleep (subprocess released, a record kept under
``data/sessions``); the next request for it — or a restarted server — wakes
it: the transcript replays and the conversation resumes (``resume=``). No model
calls: ``ClaudeSDKClient`` is replaced by a scripted fake that records how it
was built.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from relife.server import session as sess_mod
from relife.server.session import AgentSession, SessionManager, SessionStore


class FakeClient:
    """Records each client built (with its options) and answers every query."""

    built: list["FakeClient"] = []
    fail_resume = False

    def __init__(self, options):
        self.options = options
        self.connected = False
        FakeClient.built.append(self)

    async def connect(self):
        if FakeClient.fail_resume and self.options.resume:
            raise RuntimeError("No conversation found")
        self.connected = True

    async def query(self, prompt):
        self.prompt = prompt

    async def receive_response(self):
        yield AssistantMessage(content=[TextBlock(text=f"echo: {self.prompt}")], model="m")
        yield ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
            num_turns=1, session_id="sdk-conv-1", total_cost_usd=0.01,
        )

    async def disconnect(self):
        self.connected = False


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch):
    FakeClient.built = []
    FakeClient.fail_resume = False
    monkeypatch.setattr(sess_mod, "ClaudeSDKClient", FakeClient)

    async def no_upkeep():
        return None

    monkeypatch.setattr(sess_mod, "maybe_consolidate_off_loop", no_upkeep)


async def _turn(session, text):
    q, _ = session.subscribe()
    try:
        await session.submit(text)
        while True:
            _, ev = await asyncio.wait_for(q.get(), 5)
            if ev["type"] in ("result", "error"):
                return ev
    finally:
        session.unsubscribe(q)


# --- the store ------------------------------------------------------------------------------
def test_the_store_names_files_only_by_server_ids(tmp_path):
    st = SessionStore(tmp_path)
    good = "a" * 32
    st.save({"id": good, "workspace": "w", "updated_at": 1e12, "events": []})
    assert st.load(good)["workspace"] == "w"
    for bad in ("../../etc/passwd", "A" * 32, "a" * 31, "", None):
        st.save({"id": bad, "updated_at": 1e12})
        assert st.load(bad) is None
    assert st.ids() == [good]
    assert st.delete(good) and st.load(good) is None


def test_the_store_is_bounded(tmp_path):
    import os
    import time

    st = SessionStore(tmp_path, keep=10, retention_days=1)
    now = time.time()
    for i in range(5):
        sid = f"{i:032x}"
        st.save({"id": sid, "updated_at": now, "events": []})
        os.utime(tmp_path / f"{sid}.json", (now - 100 + i, now - 100 + i))
    st.keep = 3
    st.prune()
    assert sorted(st.ids()) == [f"{i:032x}" for i in (2, 3, 4)]  # the newest three
    old = "f" * 32
    st.save({"id": old, "updated_at": now - 3 * 86400, "events": []})
    assert st.load(old) is None  # past retention


# --- the session ------------------------------------------------------------------------------
def test_a_session_starts_its_subprocess_only_for_a_turn(tmp_path):
    async def flow():
        s = AgentSession(tmp_path)
        await s.start()
        assert FakeClient.built == []  # opened, not used: no subprocess
        ev = await _turn(s, "hello")
        assert ev["type"] == "result" and len(FakeClient.built) == 1
        assert FakeClient.built[0].options.resume is None
        assert s.sdk_session_id == "sdk-conv-1"
        await _turn(s, "again")
        assert len(FakeClient.built) == 1  # one client across turns
        snap = s.snapshot()
        await s.aclose()
        return snap

    snap = asyncio.run(flow())
    assert snap["sdk_session_id"] == "sdk-conv-1"
    kinds = [ev["type"] for _, ev in snap["events"]]
    assert kinds.count("user") == 2 and kinds.count("result") == 2


def test_a_resume_that_fails_starts_fresh_and_says_so(tmp_path):
    FakeClient.fail_resume = True

    async def flow():
        s = AgentSession(tmp_path, resume="gone-conv")
        await s.start()
        q, _ = s.subscribe()
        await s.submit("hi")
        seen = []
        while not seen or seen[-1]["type"] != "result":
            seen.append((await asyncio.wait_for(q.get(), 5))[1])
        await s.aclose()
        return seen

    seen = asyncio.run(flow())
    assert any(e["type"] == "note" and "could not be resumed" in e["text"] for e in seen)
    assert [c.options.resume for c in FakeClient.built] == ["gone-conv", None]


# --- the manager: sleep and wake ---------------------------------------------------------------
def _manager(tmp_path, **kw):
    root = tmp_path / "ws"
    root.mkdir(exist_ok=True)
    from relife.server.security import resolve_workspace

    return SessionManager(
        store=SessionStore(tmp_path / "sessions"),
        workspace_ok=lambda raw: resolve_workspace(raw, root),
        idle_timeout=kw.pop("idle_timeout", 3600),
        **kw,
    ), root


def test_an_idle_session_sleeps_and_wakes_with_its_conversation(tmp_path):
    async def flow():
        m, root = _manager(tmp_path, idle_timeout=0.01)
        s = await m.create(root)
        await _turn(s, "remember the number 7")
        sid = s.id
        await asyncio.sleep(0.02)
        assert await m.reap_idle() == [sid]
        assert m.get(sid) is None and not FakeClient.built[0].connected  # subprocess released
        woke = await m.restore(sid)
        assert woke is not None and woke is not s and woke.id == sid
        _, backlog = woke.subscribe(last_id=0)
        texts = [ev.get("text") for _, ev in backlog]
        assert "remember the number 7" in texts and "echo: remember the number 7" in texts
        assert len(FakeClient.built) == 1  # woken, not yet used: still no subprocess
        await _turn(woke, "what was it?")
        assert FakeClient.built[-1].options.resume == "sdk-conv-1"  # the same conversation
        await m.aclose()

    asyncio.run(flow())


def test_a_restarted_server_finds_its_sessions(tmp_path):
    async def first():
        m, root = _manager(tmp_path)
        s = await m.create(root)
        await _turn(s, "before the restart")
        await m.aclose()  # shutdown
        return s.id

    async def second(sid):
        m, _ = _manager(tmp_path)  # a new process: nothing in memory
        s = await m.restore(sid)
        _, backlog = s.subscribe(last_id=0)
        seqs = [seq for seq, _ in backlog]
        await m.aclose()
        return s, backlog, seqs

    sid = asyncio.run(first())
    s, backlog, seqs = asyncio.run(second(sid))
    assert any(ev.get("text") == "before the restart" for _, ev in backlog)
    assert seqs == sorted(seqs) and s._seq == seqs[-1]  # numbering continues, not restarts


def test_new_session_discards_for_good(tmp_path):
    async def flow():
        m, root = _manager(tmp_path)
        s = await m.create(root)
        await _turn(s, "x")
        assert await m.close(s.id, forget=True)
        assert await m.restore(s.id) is None
        assert not await m.close(s.id, forget=True)

    asyncio.run(flow())


def test_a_record_pointing_outside_the_root_never_wakes(tmp_path):
    async def flow():
        m, root = _manager(tmp_path)
        sid = "b" * 32
        m.store.save({"id": sid, "workspace": str(tmp_path.parent), "updated_at": 1e12, "events": []})
        assert await m.restore(sid) is None  # a tampered record is not a way out of the root

    asyncio.run(flow())


def test_two_requests_wake_one_session(tmp_path):
    async def flow():
        m, root = _manager(tmp_path)
        s = await m.create(root)
        await _turn(s, "x")
        await m.close(s.id)
        a, b = await asyncio.gather(m.restore(s.id), m.restore(s.id))
        assert a is b and m.count() == 1
        await m.aclose()

    asyncio.run(flow())


# --- over HTTP ------------------------------------------------------------------------------------
def test_the_console_reattaches_after_a_restart(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from relife.server.app import create_app

    def app():
        return create_app(
            schedules_path=tmp_path / "schedules.json", runs_dir=tmp_path / "runs",
            run_scheduler=False, reap=False, workspace_root=tmp_path / "ws",
            agents_path=tmp_path / "agents.json", crews_dir=tmp_path / "crews",
            sessions_dir=tmp_path / "sessions",
        )

    (tmp_path / "ws").mkdir()
    with TestClient(app()) as tc:
        sid = tc.post("/sessions", json={}).json()["session_id"]
        assert tc.post(f"/sessions/{sid}/messages", json={"text": "hello"}).status_code == 200
        host = tc.app.state.manager.get(sid)
        for _ in range(200):
            if host.sdk_session_id:
                break
            import time

            time.sleep(0.01)
    rec = json.loads((tmp_path / "sessions" / f"{sid}.json").read_text(encoding="utf-8"))
    assert rec["sdk_session_id"] == "sdk-conv-1"

    with TestClient(app()) as tc:  # the restarted server
        assert tc.get(f"/sessions/{sid}").status_code == 200
        assert tc.app.state.manager.get(sid).sdk_session_id == "sdk-conv-1"
        assert tc.delete(f"/sessions/{sid}").json() == {"closed": True}
        assert tc.get(f"/sessions/{sid}").status_code == 404
    assert not (tmp_path / "sessions" / f"{sid}.json").exists()


# --- records a dead process left "running" ----------------------------------------------------
def _dead_pid():
    import subprocess
    import sys

    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_a_crew_record_whose_process_died_reads_as_interrupted(tmp_path):
    import os

    from relife.crew.record import CrewRunRecord, CrewRunStore

    store = CrewRunStore(tmp_path)
    spec = {"goal": "g", "agents": [], "tasks": []}
    dead = CrewRunRecord(id=store.reserve_id(), task="t", workspace=".", spec=spec, status="running", pid=_dead_pid())
    alive = CrewRunRecord(id=store.reserve_id(), task="t", workspace=".", spec=spec, status="running", pid=os.getpid())
    legacy = CrewRunRecord(id=store.reserve_id(), task="t", workspace=".", spec=spec, status="running")
    for r in (dead, alive, legacy):
        store.save(r)
    got = store.get(dead.id)
    assert got.status == "interrupted" and "ended before it finished" in got.error
    assert json.loads((tmp_path / dead.id / "record.json").read_text())["status"] == "interrupted"
    assert store.get(alive.id).status == "running"  # its owner is still at work
    assert store.get(legacy.id).status == "running"  # no pid: can't tell, left alone
    assert {r.id: r.status for r in store.list()}[dead.id] == "interrupted"


def test_runs_a_stopped_server_left_submitted_are_closed_at_startup(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from relife.server.app import create_app
    from relife.server.schedules import Schedule, ScheduleStore

    store = ScheduleStore(tmp_path / "schedules.json")
    s = store.add(Schedule.new(name="n", task="t", spec={"every": "1h"}, now=0.0))
    s.record_run(10.0, "done", run_id="20260101-000000-000")
    s.record_run(20.0, "submitted", run_id="20260101-000001-000")
    store.save()

    app = create_app(
        schedules_path=tmp_path / "schedules.json", runs_dir=tmp_path / "runs",
        run_scheduler=False, reap=False, workspace_root=tmp_path,
        session_factory=lambda ws, **kw: None,
    )
    with TestClient(app) as tc:  # the lifespan runs
        runs = tc.get(f"/schedules/{s.id}").json()["runs"]
    assert [r["status"] for r in runs] == ["done", "interrupted"]
    assert ScheduleStore(tmp_path / "schedules.json").get(s.id).last_status == "interrupted"
