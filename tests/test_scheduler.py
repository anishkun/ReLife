"""Tests for the scheduler (autonomous triggers on the agent server).

Three layers, all deterministic and model-free:

1. **Cadence logic** — pure functions in ``schedules.py`` at explicit instants.
2. **Scheduler policy** — ``Scheduler.tick``/``fire`` against a fake
   ``SessionManager`` (what fires, what's skipped, how the record advances).
3. **HTTP layer** — the ``/schedules`` routes over ``TestClient`` with the same
   recording fake session the rest of the server suite uses.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import anyio
import pytest

from relife import config
from relife.server.schedules import (
    Schedule,
    ScheduleStore,
    describe_spec,
    next_run,
    normalize_spec,
    parse_every,
)


def _local(y, mo, d, h, mi) -> float:
    """Epoch seconds of a naive *local* wall-clock time (what ``at`` schedules use)."""
    return datetime(y, mo, d, h, mi).timestamp()


# =============================================================================
# Layer 1 — cadence logic
# =============================================================================
def test_parse_every_units_and_bare_seconds():
    assert parse_every("30m") == 1800
    assert parse_every("2h") == 7200
    assert parse_every("1d") == 86400
    assert parse_every("90s") == 90
    assert parse_every(600) == 600
    for bad in ("", "soon", "5x", "0m", -1, True):
        with pytest.raises(ValueError):
            parse_every(bad)


def test_normalize_spec_requires_exactly_one_form():
    assert normalize_spec({"every": "30m"}, min_interval=0) == {"every": "30m"}
    assert normalize_spec({"every": 5400}, min_interval=0) == {"every": "90m"}
    assert normalize_spec({"at": "9:05"}) == {"at": "09:05"}
    assert normalize_spec({"at": "09:00", "days": ["Friday", "mon", "MON"]}) == {
        "at": "09:00",
        "days": ["mon", "fri"],
    }
    for bad in ({}, {"every": "1h", "at": "09:00"}, {"at": "25:00"}, {"at": "09:00", "days": ["funday"]}, "1h"):
        with pytest.raises(ValueError):
            normalize_spec(bad)


def test_interval_floor_refuses_budget_burning_schedules():
    """Every run spends Max budget: a one-minute schedule is a mistake."""
    with pytest.raises(ValueError, match="at least"):
        normalize_spec({"every": "1m"}, min_interval=300)
    assert normalize_spec({"every": "5m"}, min_interval=300) == {"every": "5m"}


def test_next_run_interval_is_from_now():
    assert next_run({"every": "30m"}, 1000.0) == 2800.0


def test_next_run_daily_at_today_if_still_ahead_else_tomorrow():
    now = _local(2026, 9, 21, 8, 30)  # a Monday
    assert next_run({"at": "09:00"}, now) == _local(2026, 9, 21, 9, 0)
    assert next_run({"at": "08:00"}, now) == _local(2026, 9, 22, 8, 0)
    # Exactly on the slot counts as passed (strictly after `now`).
    assert next_run({"at": "08:30"}, now) == _local(2026, 9, 22, 8, 30)


def test_next_run_daily_at_honours_weekdays():
    now = _local(2026, 9, 21, 12, 0)  # Monday noon
    assert next_run({"at": "09:00", "days": ["mon", "fri"]}, now) == _local(2026, 9, 25, 9, 0)
    assert next_run({"at": "09:00", "days": ["mon"]}, now) == _local(2026, 9, 28, 9, 0)
    assert next_run({"at": "13:00", "days": ["mon"]}, now) == _local(2026, 9, 21, 13, 0)


def test_describe_spec():
    assert describe_spec({"every": "2h"}) == "every 2h"
    assert describe_spec({"at": "09:00"}) == "daily at 09:00"
    assert describe_spec({"at": "09:00", "days": ["mon", "fri"]}) == "daily at 09:00 (mon, fri)"


def test_schedule_new_validates_and_computes_first_slot():
    s = Schedule.new(name="  triage  ", task="check inbox", spec={"every": "1h"}, now=1000.0)
    assert s.name == "triage" and s.next_run_at == 4600.0 and s.enabled
    assert s.is_due(4600.0) and not s.is_due(4599.0)
    with pytest.raises(ValueError, match="name"):
        Schedule.new(name="", task="x", spec={"every": "1h"})
    with pytest.raises(ValueError, match="task"):
        Schedule.new(name="n", task="  ", spec={"every": "1h"})
    with pytest.raises(ValueError, match="exceeds"):
        Schedule.new(name="n", task="x" * (config.AGENT_MAX_MESSAGE_CHARS + 1), spec={"every": "1h"})


def test_record_run_advances_from_now_and_bounds_history():
    """A slot missed while the server was down fires once, then advances from
    *now* — never a burst of N catch-up runs."""
    s = Schedule.new(name="n", task="t", spec={"every": "1h"}, now=0.0)
    late = 10 * 3600.0  # ten slots overdue
    assert s.is_due(late)
    s.record_run(late, "submitted", keep=3)
    assert s.next_run_at == late + 3600 and not s.is_due(late)
    for i in range(5):
        s.record_run(late + i, f"r{i}", keep=3)
    assert [r["status"] for r in s.runs] == ["r2", "r3", "r4"]
    assert s.last_status == "r4"


def test_store_persists_atomically_and_reloads(tmp_path):
    path = tmp_path / "schedules.json"
    store = ScheduleStore(path)
    assert store.count() == 0 and not path.exists()  # nothing written until a change
    now = _local(2026, 9, 21, 8, 0)
    s = store.add(Schedule.new(name="n", task="t", spec={"at": "09:00"}, now=now))
    s.record_run(now + 5, "submitted")
    store.save()
    assert path.exists() and not path.with_suffix(".json.tmp").exists()

    again = ScheduleStore(path)
    got = again.get(s.id)
    assert got is not None and got.task == "t" and got.last_status == "submitted"
    assert got.next_run_at == s.next_run_at
    assert again.remove(s.id) and not again.remove(s.id)
    assert json.loads(path.read_text())["schedules"] == []


def test_store_ignores_a_torn_file(tmp_path):
    path = tmp_path / "schedules.json"
    path.write_text("{not json")
    assert ScheduleStore(path).count() == 0


# =============================================================================
# Layer 2 — scheduler policy against a fake manager
# =============================================================================
class FakeSession:
    def __init__(self, workspace: Path, sid: str) -> None:
        self.id = sid
        self.workspace = workspace
        self.submitted: list[str] = []
        self.busy = False
        self.queue_full = False

    async def submit(self, text: str) -> None:
        from relife.server.session import TurnQueueFull

        if self.queue_full:
            raise TurnQueueFull("full")
        self.submitted.append(text)


class FakeManager:
    def __init__(self, limit: int = 8) -> None:
        self.sessions: dict[str, FakeSession] = {}
        self.limit = limit
        self.created: list[Path] = []

    async def create(self, workspace: Path) -> FakeSession:
        from relife.server.session import SessionLimitReached

        if len(self.sessions) >= self.limit:
            raise SessionLimitReached("at most N sessions")
        s = FakeSession(workspace, f"sess-{len(self.created)}")
        self.created.append(workspace)
        self.sessions[s.id] = s
        return s

    def get(self, sid: str):
        return self.sessions.get(sid)


@pytest.fixture
def sched(tmp_path):
    from relife.server.scheduler import Scheduler

    root = tmp_path / "ws"
    root.mkdir()
    store = ScheduleStore(tmp_path / "schedules.json")
    manager = FakeManager()
    return Scheduler(store, manager, workspace_root=root, tick=0.01), store, manager, root


def test_tick_fires_due_schedules_into_their_own_sessions(sched):
    scheduler, store, manager, root = sched
    a = store.add(Schedule.new(name="a", task="do A", spec={"every": "1h"}, workspace="proj-a", now=0.0))
    b = store.add(Schedule.new(name="b", task="do B", spec={"every": "2h"}, now=0.0))
    c = store.add(Schedule.new(name="c", task="do C", spec={"every": "1h"}, enabled=False, now=0.0))

    fired = anyio.run(scheduler.tick, 3600.0)
    assert fired == [a.id]  # b isn't due yet, c is disabled
    assert manager.created == [root / "proj-a"] and (root / "proj-a").is_dir()
    sess = manager.sessions[a.session_id]
    assert sess.submitted and sess.submitted[0].endswith("do A")
    assert sess.submitted[0].startswith("[Scheduled run: a]")
    assert a.last_status == "submitted" and a.next_run_at == 7200.0
    assert b.last_status is None and c.last_status is None

    # The record hit disk (a restart must not forget the run happened).
    assert ScheduleStore(store.path).get(a.id).last_status == "submitted"

    # Next slot: reuses the same session rather than spawning another.
    fired = anyio.run(scheduler.tick, 7200.0)
    assert set(fired) == {a.id, b.id}
    assert len(manager.created) == 2  # one new session for b only
    assert len(sess.submitted) == 2


def test_busy_session_skips_the_slot_instead_of_stacking_turns(sched):
    scheduler, store, manager, _ = sched
    a = store.add(Schedule.new(name="a", task="long job", spec={"every": "1h"}, now=0.0))
    anyio.run(scheduler.tick, 3600.0)
    sess = manager.sessions[a.session_id]
    sess.busy = True

    anyio.run(scheduler.tick, 7200.0)
    assert len(sess.submitted) == 1
    assert a.last_status == "skipped: previous run still in progress"
    assert a.next_run_at == 10800.0  # advanced anyway: no tight retry loop

    sess.busy = False
    anyio.run(scheduler.tick, 10800.0)
    assert len(sess.submitted) == 2 and a.last_status == "submitted"


def test_reaped_session_is_recreated_on_the_next_slot(sched):
    scheduler, store, manager, _ = sched
    a = store.add(Schedule.new(name="a", task="t", spec={"every": "1h"}, now=0.0))
    anyio.run(scheduler.tick, 3600.0)
    first = a.session_id
    manager.sessions.pop(first)  # the idle reaper closed it

    anyio.run(scheduler.tick, 7200.0)
    assert a.session_id != first and a.last_status == "submitted"
    assert a.session_id in manager.sessions


def test_session_ceiling_and_full_queue_are_recorded_not_raised(sched):
    scheduler, store, manager, _ = sched
    manager.limit = 0
    a = store.add(Schedule.new(name="a", task="t", spec={"every": "1h"}, now=0.0))
    anyio.run(scheduler.tick, 3600.0)
    assert a.last_status.startswith("skipped: at most")
    assert a.session_id is None and a.next_run_at == 7200.0

    manager.limit = 8
    anyio.run(scheduler.tick, 7200.0)
    manager.sessions[a.session_id].queue_full = True
    anyio.run(scheduler.tick, 10800.0)
    assert a.last_status == "skipped: full"


def test_workspace_outside_root_is_an_error_not_a_session(sched):
    scheduler, store, manager, root = sched
    a = store.add(Schedule.new(name="a", task="t", spec={"every": "1h"}, workspace="../escape", now=0.0))
    anyio.run(scheduler.tick, 3600.0)
    assert a.last_status.startswith("error: workspace must be inside")
    assert manager.created == []


def test_fire_runs_a_schedule_that_is_not_due(sched):
    scheduler, store, manager, _ = sched
    a = store.add(Schedule.new(name="a", task="t", spec={"at": "09:00"}, now=_local(2026, 9, 21, 8, 0)))
    status = anyio.run(lambda: scheduler.fire(a, now=_local(2026, 9, 21, 8, 5)))
    assert status == "submitted" and a.session_id in manager.sessions
    # Firing by hand doesn't steal the regular slot: 09:00 today still stands.
    assert a.next_run_at == _local(2026, 9, 21, 9, 0)


def test_run_loop_ticks_and_survives_a_failing_tick(sched):
    scheduler, store, manager, _ = sched
    store.add(Schedule.new(name="a", task="t", spec={"every": "1h"}, now=0.0))
    calls = {"n": 0}
    real_tick = scheduler.tick

    async def flaky(now=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return await real_tick(now)

    scheduler.tick = flaky  # type: ignore[method-assign]

    async def flow():
        task = asyncio.create_task(scheduler.run())
        for _ in range(200):
            await asyncio.sleep(0.01)
            if calls["n"] >= 3:
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    anyio.run(flow)
    assert calls["n"] >= 3  # kept ticking after the exception


def test_agent_session_reports_busy_while_a_turn_runs():
    """`busy` is what the scheduler consults; it must cover a queued turn *and*
    one in flight, and clear once the turn ends."""
    from relife.server.session import AgentSession

    class _Client:
        def __init__(self):
            self.release = asyncio.Event()

        async def query(self, text):
            pass

        async def receive_response(self):
            await self.release.wait()
            yield object()

    async def flow():
        s = AgentSession(Path("."))
        client = _Client()
        s._client = client  # type: ignore[assignment]
        assert s.busy is False
        await s.submit("hello")
        assert s.busy is True  # queued
        s._worker = asyncio.create_task(s._run())
        await asyncio.sleep(0.02)
        assert s.busy is True  # in flight
        client.release.set()
        await asyncio.sleep(0.05)
        assert s.busy is False
        await s.aclose()

    anyio.run(flow)


# =============================================================================
# Layer 3 — HTTP routes
# =============================================================================
class RecordingSession:
    def __init__(self, workspace: Path, sid: str) -> None:
        self.id = sid
        self.workspace = workspace
        self.submitted: list[str] = []
        self.busy = False

    async def start(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    async def submit(self, text: str) -> None:
        self.submitted.append(text)

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        return True

    def subscribe(self, last_id=None):
        return asyncio.Queue(), []

    def unsubscribe(self, q) -> None:
        pass


@pytest.fixture
def http(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from relife.server.app import create_app

    monkeypatch.setattr(config, "AGENT_SCHEDULE_MIN_INTERVAL", 0.0)
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
        return TestClient(app), created, app

    return make


def test_schedule_crud_over_http(http):
    tc, _created, _app = http()
    assert tc.get("/schedules").json() == {"schedules": []}

    r = tc.post("/schedules", json={"name": "inbox", "task": "triage my inbox", "every": "1h", "workspace": "mail"})
    assert r.status_code == 201, r.text
    rec = r.json()
    assert rec["spec"] == {"every": "1h"} and rec["spec_text"] == "every 1h"
    assert rec["enabled"] is True and rec["next_run_at"] > rec["created_at"]
    sid = rec["id"]

    # Nested spec form + weekdays.
    r2 = tc.post("/schedules", json={"name": "standup", "task": "summarize", "spec": {"at": "09:00", "days": ["mon"]}})
    assert r2.status_code == 201 and r2.json()["spec_text"] == "daily at 09:00 (mon)"

    assert [s["name"] for s in tc.get("/schedules").json()["schedules"]] == ["inbox", "standup"]
    assert tc.get(f"/schedules/{sid}").json()["task"] == "triage my inbox"
    assert tc.get("/health").json()["schedules"] == 2

    p = tc.patch(f"/schedules/{sid}", json={"enabled": False, "task": "triage inbox, archive newsletters"})
    assert p.status_code == 200 and p.json()["enabled"] is False
    assert p.json()["task"] == "triage inbox, archive newsletters"
    before = p.json()["next_run_at"]
    p2 = tc.patch(f"/schedules/{sid}", json={"every": "2h"})
    assert p2.json()["spec"] == {"every": "2h"} and p2.json()["next_run_at"] != before

    assert tc.delete(f"/schedules/{sid}").json() == {"removed": True}
    assert tc.delete(f"/schedules/{sid}").status_code == 404
    assert tc.get(f"/schedules/{sid}").status_code == 404
    assert tc.patch(f"/schedules/{sid}", json={"enabled": True}).status_code == 404


def test_schedule_validation_over_http(http):
    tc, _, _ = http()
    bad = [
        {"task": "t", "every": "1h"},                       # no name
        {"name": "n", "every": "1h"},                       # no task
        {"name": "n", "task": "t"},                         # no cadence
        {"name": "n", "task": "t", "every": "1h", "at": "09:00"},
        {"name": "n", "task": "t", "every": "sometimes"},
        {"name": "n", "task": "t", "at": "9pm"},
        {"name": "n", "task": "t", "every": "1h", "workspace": "../outside"},
    ]
    for body in bad:
        assert tc.post("/schedules", json=body).status_code == 400, body
    sid = tc.post("/schedules", json={"name": "n", "task": "t", "every": "1h"}).json()["id"]
    assert tc.patch(f"/schedules/{sid}", json={"every": "never"}).status_code == 400
    assert tc.patch(f"/schedules/{sid}", json={"workspace": "/"}).status_code == 400
    assert tc.patch(f"/schedules/{sid}", json={"name": ""}).status_code == 400


def test_schedule_ceiling(http, monkeypatch):
    monkeypatch.setattr(config, "AGENT_MAX_SCHEDULES", 2)
    tc, _, _ = http()
    for i in range(2):
        assert tc.post("/schedules", json={"name": f"n{i}", "task": "t", "every": "1h"}).status_code == 201
    assert tc.post("/schedules", json={"name": "n2", "task": "t", "every": "1h"}).status_code == 429


def test_run_now_submits_to_a_session_and_reports_it(http):
    tc, created, _ = http()
    sid = tc.post("/schedules", json={"name": "n", "task": "ship it", "every": "1h", "workspace": "proj"}).json()["id"]
    r = tc.post(f"/schedules/{sid}/run")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "submitted" and body["session_id"] == "sess-0"
    assert created[0].submitted[0].endswith("ship it")
    assert created[0].workspace.name == "proj"
    assert body["schedule"]["last_status"] == "submitted"
    # The console can attach to that session through the ordinary probe.
    assert tc.get("/sessions/sess-0").status_code == 200
    assert tc.post("/schedules/nope/run").status_code == 404


def test_scheduler_tick_uses_the_apps_manager(http):
    """A due schedule fires through the same SessionManager the console uses."""
    tc, created, app = http()
    rec = tc.post("/schedules", json={"name": "n", "task": "t", "every": "1h"}).json()
    fired = anyio.run(app.state.scheduler.tick, rec["next_run_at"] + 1)
    assert fired == [rec["id"]]
    assert created and created[0].submitted[0].endswith("\n\nt")
    assert tc.get(f"/schedules/{rec['id']}").json()["session_id"] == created[0].id


def test_schedules_require_auth_and_same_origin(http):
    tc, _, _ = http(token="secret")
    assert tc.get("/schedules").status_code == 401
    assert tc.post("/schedules", json={"name": "n", "task": "t", "every": "1h"}).status_code == 401
    h = {"Authorization": "Bearer secret"}
    assert tc.get("/schedules", headers=h).status_code == 200
    r = tc.post("/schedules", json={"name": "n", "task": "t", "every": "1h"}, headers=h)
    assert r.status_code == 201
    sid = r.json()["id"]
    cross = {**h, "Origin": "http://evil.example"}
    assert tc.post(f"/schedules/{sid}/run", headers=cross).status_code == 403
    assert tc.delete(f"/schedules/{sid}", headers=cross).status_code == 403
    assert tc.patch(f"/schedules/{sid}", json={"enabled": False}, headers=cross).status_code == 403


def test_schedules_survive_an_app_rebuild(http, tmp_path):
    """The store is the source of truth: a restart sees the same schedules."""
    tc, _, _ = http()
    tc.post("/schedules", json={"name": "keep", "task": "t", "every": "1h"})
    tc2, _, _ = http()
    assert [s["name"] for s in tc2.get("/schedules").json()["schedules"]] == ["keep"]
