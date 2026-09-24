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
    """Enough of AgentSession for the scheduler: submit + a pubsub the test
    drives by hand with :meth:`emit`."""

    def __init__(self, workspace: Path, sid: str) -> None:
        self.id = sid
        self.workspace = workspace
        self.submitted: list[str] = []
        self.grants: list = []
        self.busy = False
        self.queue_full = False
        self.subs: set[asyncio.Queue] = set()
        self.seq = 0

    async def submit(self, text: str, *, grants=None) -> None:
        from relife.server.session import TurnQueueFull

        if self.queue_full:
            raise TurnQueueFull("full")
        self.submitted.append(text)
        self.grants.append(grants)

    def subscribe(self, last_id=None):
        q: asyncio.Queue = asyncio.Queue()
        self.subs.add(q)
        return q, []

    def unsubscribe(self, q) -> None:
        self.subs.discard(q)

    def emit(self, *events: dict) -> None:
        for ev in events:
            self.seq += 1
            for q in list(self.subs):
                q.put_nowait((self.seq, ev))


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
    from relife.server.runs import RunStore
    from relife.server.scheduler import Scheduler

    root = tmp_path / "ws"
    root.mkdir()
    store = ScheduleStore(tmp_path / "schedules.json")
    manager = FakeManager()
    scheduler = Scheduler(
        store, manager, workspace_root=root, tick=0.01, runs=RunStore(tmp_path / "runs"), run_timeout=2.0
    )
    return scheduler, store, manager, root


def _turn(prompt: str, *middle: dict, result: dict | None = None) -> list[dict]:
    """A scripted turn: the session's `user` echo, whatever happened, and the end."""
    return [{"type": "user", "text": prompt}, *middle, result or {"type": "result", "cost_usd": 0.02}]


async def _settle(scheduler) -> None:
    for _ in range(100):
        if not scheduler._recorders:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("recorder never finished")


async def _complete(scheduler, manager, schedule, text="ok") -> None:
    """Play the scheduled turn to its end so the recorder finishes."""
    sess = manager.sessions[schedule.session_id]
    sess.emit(*_turn(sess.submitted[-1], {"type": "text", "text": text}))
    await _settle(scheduler)


def test_tick_fires_due_schedules_into_their_own_sessions(sched):
    scheduler, store, manager, root = sched
    a = store.add(Schedule.new(name="a", task="do A", spec={"every": "1h"}, workspace="proj-a", now=0.0))
    b = store.add(Schedule.new(name="b", task="do B", spec={"every": "2h"}, now=0.0))
    c = store.add(Schedule.new(name="c", task="do C", spec={"every": "1h"}, enabled=False, now=0.0))

    async def flow():
        fired = await scheduler.tick(3600.0)
        assert fired == [a.id]  # b isn't due yet, c is disabled
        assert manager.created == [root / "proj-a"] and (root / "proj-a").is_dir()
        sess = manager.sessions[a.session_id]
        assert sess.submitted and sess.submitted[0].endswith("do A")
        assert sess.submitted[0].startswith("[Scheduled run: a]")
        assert a.last_status == "submitted" and a.next_run_at == 7200.0
        assert b.last_status is None and c.last_status is None

        # The record hit disk (a restart must not forget the run happened).
        assert ScheduleStore(store.path).get(a.id).last_status == "submitted"
        await _complete(scheduler, manager, a)
        assert a.last_status == "done"

        # Next slot: reuses the same session rather than spawning another.
        fired = await scheduler.tick(7200.0)
        assert set(fired) == {a.id, b.id}
        assert len(manager.created) == 2  # one new session for b only
        assert len(sess.submitted) == 2
        # Both runs are being recorded at once; end both, then let them settle.
        sess.emit(*_turn(sess.submitted[-1], {"type": "text", "text": "A again"}))
        sb = manager.sessions[b.session_id]
        sb.emit(*_turn(sb.submitted[-1], {"type": "text", "text": "B"}))
        await _settle(scheduler)
        assert a.runs[-1]["summary"] == "A again" and b.runs[-1]["summary"] == "B"

    anyio.run(flow)


def test_busy_session_skips_the_slot_instead_of_stacking_turns(sched):
    scheduler, store, manager, _ = sched
    a = store.add(Schedule.new(name="a", task="long job", spec={"every": "1h"}, now=0.0))

    async def flow():
        await scheduler.tick(3600.0)
        sess = manager.sessions[a.session_id]
        sess.busy = True  # the first run is still going

        await scheduler.tick(7200.0)
        assert len(sess.submitted) == 1
        assert a.last_status == "skipped: previous run still in progress"
        assert a.next_run_at == 10800.0  # advanced anyway: no tight retry loop
        assert a.runs[-1].get("run_id") is None  # nothing to record for a skip

        await _complete(scheduler, manager, a)
        sess.busy = False
        await scheduler.tick(10800.0)
        assert len(sess.submitted) == 2 and a.last_status == "submitted"
        await _complete(scheduler, manager, a)

    anyio.run(flow)


def test_reaped_session_is_recreated_on_the_next_slot(sched):
    scheduler, store, manager, _ = sched
    a = store.add(Schedule.new(name="a", task="t", spec={"every": "1h"}, now=0.0))

    async def flow():
        await scheduler.tick(3600.0)
        first = a.session_id
        await _complete(scheduler, manager, a)
        manager.sessions.pop(first)  # the idle reaper closed it

        await scheduler.tick(7200.0)
        assert a.session_id != first and a.last_status == "submitted"
        assert a.session_id in manager.sessions
        await _complete(scheduler, manager, a)

    anyio.run(flow)


def test_session_ceiling_and_full_queue_are_recorded_not_raised(sched):
    scheduler, store, manager, _ = sched
    manager.limit = 0
    a = store.add(Schedule.new(name="a", task="t", spec={"every": "1h"}, now=0.0))
    anyio.run(scheduler.tick, 3600.0)
    assert a.last_status.startswith("skipped: at most")
    assert a.session_id is None and a.next_run_at == 7200.0

    manager.limit = 8

    async def flow():
        await scheduler.tick(7200.0)
        await _complete(scheduler, manager, a)
        sess = manager.sessions[a.session_id]
        sess.queue_full = True
        await scheduler.tick(10800.0)
        assert a.last_status == "skipped: full"
        assert not sess.subs  # the recorder let go when the submit failed

    anyio.run(flow)


def test_run_with_no_free_stream_slot_still_runs_but_says_it_is_unrecorded(sched):
    """When every SSE slot is held by watchers the turn still goes out, but the
    history must not show a `submitted` that no recorder will ever upgrade."""
    scheduler, store, manager, _ = sched
    a = store.add(Schedule.new(name="a", task="t", spec={"every": "1h"}, now=0.0))

    async def flow():
        await scheduler.tick(3600.0)
        await _complete(scheduler, manager, a)
        sess = manager.sessions[a.session_id]

        def full(last_id=None):
            from relife.server.session import TooManySubscribers

            raise TooManySubscribers("too many event streams open for this session")

        sess.subscribe = full
        await scheduler.tick(7200.0)
        assert len(sess.submitted) == 2  # the task was still submitted
        assert a.last_status.startswith("submitted (unrecorded: too many event streams")
        assert "run_id" not in a.runs[-1]  # nothing will ever upgrade this entry
        assert not scheduler._recorders and a.next_run_at == 10800.0

    anyio.run(flow)


def test_workspace_outside_root_is_an_error_not_a_session(sched):
    scheduler, store, manager, root = sched
    a = store.add(Schedule.new(name="a", task="t", spec={"every": "1h"}, workspace="../escape", now=0.0))
    anyio.run(scheduler.tick, 3600.0)
    assert a.last_status.startswith("error: workspace must be inside")
    assert manager.created == []


def test_fire_runs_a_schedule_that_is_not_due(sched):
    scheduler, store, manager, _ = sched
    a = store.add(Schedule.new(name="a", task="t", spec={"at": "09:00"}, now=_local(2026, 9, 21, 8, 0)))
    async def flow():
        status = await scheduler.fire(a, now=_local(2026, 9, 21, 8, 5))
        assert status == "submitted" and a.session_id in manager.sessions
        await _complete(scheduler, manager, a)

    anyio.run(flow)
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


def test_recorder_persists_the_outcome_and_upgrades_the_history(sched):
    """`submitted` only means queued. Once the turn ends the run record holds
    what actually happened — and the entry the panel shows says so too."""
    scheduler, store, manager, _ = sched
    a = store.add(Schedule.new(name="inbox", task="triage", spec={"every": "1h"}, now=0.0))

    async def flow():
        await scheduler.tick(3600.0)
        sess = manager.sessions[a.session_id]
        assert a.last_status == "submitted" and a.runs[-1]["run_id"]
        assert len(sess.subs) == 1  # recorder attached before the turn ran
        sess.emit(*_turn(
            sess.submitted[0],
            {"type": "text", "text": "Looking… "},
            {"type": "tool_use", "name": "mcp__claude_ai_Gmail__search", "brief": "is:unread"},
            {"type": "tool_result", "brief": "3 threads"},
            {"type": "tool_use", "name": "Bash", "brief": "gh pr create"},
            {"type": "approval_request", "approval_id": "ap1", "tool": "Bash",
             "reason": "outward-facing", "brief": "gh pr create -t x"},
            {"type": "approval_resolved", "approval_id": "ap1", "approved": False},
            {"type": "tool_result", "brief": "denied"},
            {"type": "text", "text": "Two urgent mails from the bank. "},
            {"type": "text", "text": "I could not open the PR (approval denied)."},
        ))
        await _settle(scheduler)
        assert not sess.subs  # detached: the idle reaper may have the session back

    anyio.run(flow)
    run_id = a.runs[-1]["run_id"]
    rec = scheduler.runs.get(a.id, run_id)
    assert rec is not None and rec.status == "done" and rec.session_id == a.session_id
    assert rec.summary == "Two urgent mails from the bank. I could not open the PR (approval denied)."
    assert rec.tool_calls == 2 and rec.cost_usd == 0.02
    assert rec.denied == [{"tool": "Bash", "brief": "gh pr create -t x", "reason": "outward-facing"}]
    assert rec.events[0]["type"] == "user" and rec.events[-1]["type"] == "result"
    # The inline history entry the panel renders was upgraded in place …
    entry = a.runs[-1]
    assert entry["status"] == "done" and entry["denied"] == 1 and entry["tool_calls"] == 2
    assert entry["summary"].startswith("Two urgent")
    assert a.last_status == "done"
    # … and both are on disk.
    assert ScheduleStore(store.path).get(a.id).last_status == "done"
    assert scheduler.runs.list(a.id)[0].run_id == run_id


def test_recorder_only_takes_its_own_turn(sched):
    """A turn the user typed into the schedule's session (or one still draining)
    must not be recorded as the scheduled run's outcome."""
    scheduler, store, manager, _ = sched
    a = store.add(Schedule.new(name="n", task="t", spec={"every": "1h"}, now=0.0))

    async def flow():
        await scheduler.tick(3600.0)
        sess = manager.sessions[a.session_id]
        sess.emit(*_turn("something the user typed", {"type": "text", "text": "user answer"}))
        sess.emit(*_turn(sess.submitted[0], {"type": "text", "text": "scheduled answer"}))
        await _settle(scheduler)

    anyio.run(flow)
    rec = scheduler.runs.list(a.id)[0]
    assert rec.summary == "scheduled answer"
    assert all(e.get("text") != "user answer" for e in rec.events)


def test_recorder_records_errors_and_timeouts(sched):
    scheduler, store, manager, _ = sched
    scheduler.run_timeout = 0.05
    a = store.add(Schedule.new(name="a", task="t", spec={"every": "1h"}, now=0.0))
    b = store.add(Schedule.new(name="b", task="t", spec={"every": "1h"}, now=0.0))

    async def flow():
        await scheduler.tick(3600.0)
        sa = manager.sessions[a.session_id]
        sa.emit(*_turn(sa.submitted[0], result={"type": "error", "message": "boom"}))
        # b's session never finishes its turn.
        await _settle(scheduler)

    anyio.run(flow)
    assert scheduler.runs.list(a.id)[0].status == "error"
    assert scheduler.runs.list(a.id)[0].error == "boom"
    assert a.runs[-1]["status"] == "error"
    assert scheduler.runs.list(b.id)[0].status == "timeout"
    assert b.last_status == "timeout"


def test_aclose_interrupts_recorders_but_keeps_the_record(sched):
    scheduler, store, manager, _ = sched
    a = store.add(Schedule.new(name="a", task="t", spec={"every": "1h"}, now=0.0))

    async def flow():
        await scheduler.tick(3600.0)
        assert scheduler._recorders
        await scheduler.aclose()
        assert not scheduler._recorders

    anyio.run(flow)
    rec = scheduler.runs.list(a.id)[0]
    assert rec.status == "interrupted"
    assert a.last_status == "interrupted"


def test_summarize_events_without_tools_uses_all_text():
    from relife.server.runs import summarize_events

    info = summarize_events([
        {"type": "user", "text": "p"},
        {"type": "text", "text": "Nothing to do. "},
        {"type": "thinking"},
        {"type": "text", "text": "All quiet."},
        {"type": "result", "cost_usd": None},
    ])
    assert info == {
        "summary": "Nothing to do. All quiet.", "tool_calls": 0, "cost_usd": None, "denied": [], "acted": [],
        "error": None,
    }
    assert summarize_events([])["summary"] == ""


def test_run_store_is_bounded_and_removable(tmp_path):
    from relife.server.runs import RunRecord, RunStore

    rs = RunStore(tmp_path / "runs", keep=3)

    def rid(i: int) -> str:
        return f"20260922-0900{i:02d}-000"  # the real id shape, time-ordered

    for i in range(5):
        rec = RunRecord(schedule_id="s1", run_id=rid(i), started_at=float(i))
        rec.finish("done", [{"type": "text", "text": f"run {i}"}], now=float(i) + 1)
        rs.save(rec)
    ids = [r.run_id for r in rs.list("s1")]
    assert ids == [rid(4), rid(3), rid(2)]  # newest first, oldest pruned
    assert rs.list("s1", limit=1)[0].summary == "run 4"
    assert rs.get("s1", rid(0)) is None and rs.get("nope", rid(4)) is None
    d = rs.get("s1", rid(4)).to_dict()
    assert "events" not in d and d["event_count"] == 1
    assert "events" in rs.get("s1", rid(4)).to_dict(with_events=True)
    rs.remove_all("s1")
    assert rs.list("s1") == [] and not (tmp_path / "runs" / "s1").exists()
    rs.remove_all("s1")  # idempotent


def test_run_store_refuses_ids_that_are_not_run_ids(tmp_path):
    """A run id is a URL path parameter that names a file: only the exact shape
    ``new_id`` produces may reach the filesystem."""
    from relife.server.runs import RunRecord, RunStore

    rs = RunStore(tmp_path / "runs")
    (tmp_path / "secret.json").write_text("{}", encoding="utf-8")
    for bad in ("../../secret", "..", "", "x", "20260922-090000-000.json", "20260922-090000"):
        assert rs.get("s1", bad) is None
    assert RunRecord.new_id(0.0) and rs.get("s1", RunRecord.new_id(0.0)) is None  # valid shape, absent


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

    async def submit(self, text: str, *, grants=None) -> None:
        self.submitted.append(text)

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        return True

    def subscribe(self, last_id=None):
        self.q: asyncio.Queue = asyncio.Queue()
        return self.q, []

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
            runs_dir=tmp_path / "runs",
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


def test_run_outcomes_over_http(http, tmp_path):
    tc, created, app = http()
    sid = tc.post("/schedules", json={"name": "n", "task": "t", "every": "1h"}).json()["id"]
    assert tc.get(f"/schedules/{sid}/runs").json() == {"runs": []}
    assert tc.get("/schedules/nope/runs").status_code == 404
    assert tc.get(f"/schedules/{sid}/runs/nope").status_code == 404

    # Feed the turn's events to the recorder on the app's loop and let it finish.
    async def finish():
        sess = created[0]
        sess.q.put_nowait((1, {"type": "user", "text": sess.submitted[0]}))
        sess.q.put_nowait((2, {"type": "text", "text": "did the thing"}))
        sess.q.put_nowait((3, {"type": "result", "cost_usd": 0.5}))
        for _ in range(100):
            if not app.state.scheduler._recorders:
                return
            await asyncio.sleep(0.01)
        raise AssertionError("recorder still running")

    with tc:  # one loop for the run + the recorder (the portal lives inside the context)
        tc.post(f"/schedules/{sid}/run")
        tc.portal.call(finish)

    runs = tc.get(f"/schedules/{sid}/runs").json()["runs"]
    assert len(runs) == 1 and runs[0]["status"] == "done" and runs[0]["summary"] == "did the thing"
    assert "events" not in runs[0]
    full = tc.get(f"/schedules/{sid}/runs/{runs[0]['run_id']}").json()
    assert [e["type"] for e in full["events"]] == ["user", "text", "result"]
    sched = tc.get(f"/schedules/{sid}").json()
    assert sched["last_status"] == "done" and sched["runs"][-1]["summary"] == "did the thing"
    assert (tmp_path / "runs" / sid).is_dir()

    tc.delete(f"/schedules/{sid}")
    assert not (tmp_path / "runs" / sid).exists()


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
