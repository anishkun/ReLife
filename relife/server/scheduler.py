"""The scheduler: fires due schedules into agent sessions and records outcomes.

Runs as one background task under the app lifespan (like the idle reaper) and
ticks every ``AGENT_SCHEDULER_TICK`` seconds. A firing is just a *turn
submitted to a session*: each schedule owns a session (created in the
schedule's confined workspace, reused while alive, recreated after the idle
reaper closes it), so a scheduled run streams, journals and asks for approvals
exactly like a turn typed in the console — the UI can attach to it with
"watch". Nobody watching means an ask-case times out to **deny**, the same safe
default as the non-interactive CLI; the prompt tells the agent so.

Because nobody is watching, the run must also *deliver* unattended: for each
firing a **recorder** subscribes to the session for exactly that turn (from
its ``user`` echo to ``result``/``error``) and persists a
:class:`~.runs.RunRecord` — closing summary, tool count, cost, every approval
denied in absentia, the events — then upgrades the schedule's history entry
from ``submitted`` to the real outcome. The session's ring buffer is thus
never the only copy of what a scheduled run did.

Policy, all deterministic and unit-tested with a fake manager:
  * due = ``enabled and next_run_at <= now``; after any attempt the schedule
    advances from *now*, so a slot missed during downtime fires once, not N×;
  * a schedule whose previous run is still in progress is **skipped** for this
    slot (an hourly task that takes ninety minutes must not pile up turns);
  * a run that can't start (session ceiling, full turn queue, bad workspace)
    is recorded as ``skipped: …`` / ``error: …`` and the schedule still
    advances — the failure is visible in its history, and nothing retries in a
    tight loop against the same wall.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from .. import config
from .runs import RunRecord, RunStore
from ..permissions import describe_grant
from .schedules import Schedule, ScheduleStore
from .security import resolve_workspace
from .session import SessionLimitReached, SessionManager, TooManySubscribers, TurnQueueFull


def scheduled_prompt(schedule: Schedule) -> str:
    """The turn text a firing submits. Tells the agent the run is automatic and
    possibly unattended, so a denied approval is reported, not fought."""
    return (
        f"[Scheduled run: {schedule.name}] This turn was triggered automatically by a "
        "schedule, not typed by the user, and may be unattended — an action that needs "
        "approval can be denied by timeout; if that happens, say so and stop rather than "
        "retrying. Do the task, then finish with a short summary of what you did and "
        f"anything that needs the user.{_grants_note(schedule)}\n\n"
        f"{schedule.task}"
    )


def _grants_note(schedule: Schedule) -> str:
    """One sentence (same paragraph — the UI splits the preamble at the first
    blank line) telling the agent what it may do without an approval."""
    if not schedule.grants:
        return ""
    allowed = "; ".join(describe_grant(g) for g in schedule.grants)
    return (
        f" The user pre-approved these actions for this run, so they need no approval: "
        f"{allowed} (at most {config.AGENT_GRANT_MAX_USES} uses). Anything else still "
        "needs approval."
    )


class Scheduler:
    def __init__(
        self,
        store: ScheduleStore,
        manager: SessionManager,
        *,
        workspace_root: Path | None = None,
        tick: float | None = None,
        runs: RunStore | None = None,
        run_timeout: float | None = None,
    ) -> None:
        self.store = store
        self.manager = manager
        self.runs = runs if runs is not None else RunStore()
        self.root = Path(workspace_root) if workspace_root is not None else config.AGENT_WORKSPACE_ROOT
        self.tick_seconds = config.AGENT_SCHEDULER_TICK if tick is None else tick
        self.run_timeout = config.AGENT_SCHEDULE_RUN_TIMEOUT if run_timeout is None else run_timeout
        self._recorders: set[asyncio.Task[None]] = set()

    async def run(self) -> None:
        """Background loop: tick forever. Upkeep must never kill the server."""
        while True:
            await asyncio.sleep(self.tick_seconds)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                pass

    async def aclose(self) -> None:
        """Stop in-flight recorders (each writes an ``interrupted`` record)."""
        await asyncio.sleep(0)  # a recorder created this tick must start before it can be cancelled
        for t in list(self._recorders):
            t.cancel()
        for t in list(self._recorders):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._recorders.clear()

    async def tick(self, now: float | None = None) -> list[str]:
        """Fire every due schedule once. Returns the ids fired."""
        t = time.time() if now is None else now
        fired = []
        for schedule in self.store.due(t):
            await self.fire(schedule, now=t)
            fired.append(schedule.id)
        return fired

    async def fire(self, schedule: Schedule, *, now: float | None = None) -> str:
        """Run one schedule now (due or not) and record the outcome."""
        t = time.time() if now is None else now
        run_id = RunRecord.new_id(t)
        status = await self._submit(schedule, run_id, t)
        schedule.record_run(t, status, run_id=run_id if status == "submitted" else None)
        self.store.save()
        return status

    async def _submit(self, schedule: Schedule, run_id: str, now: float) -> str:
        session: Any = self.manager.get(schedule.session_id) if schedule.session_id else None
        if session is not None and getattr(session, "busy", False):
            return "skipped: previous run still in progress"
        if session is None:
            try:
                ws = resolve_workspace(schedule.workspace, self.root)
            except ValueError as e:
                return f"error: {e}"
            try:
                ws.mkdir(parents=True, exist_ok=True)
                session = await self.manager.create(ws)
            except SessionLimitReached as e:
                return f"skipped: {e}"
            except Exception as e:  # noqa: BLE001 - a bad workspace/subprocess must not kill the tick
                return f"error: {e}"
            schedule.session_id = session.id
        prompt = scheduled_prompt(schedule)
        # Subscribe *before* submitting so the turn's first event can't be missed.
        try:
            queue, _backlog = session.subscribe()
        except TooManySubscribers as e:
            # Every stream slot is taken by watchers, so the run can still go
            # out (someone is clearly looking) but no recorder can follow it.
            # Say so in the history rather than leave a `submitted` that would
            # never be upgraded to an outcome.
            queue = None
            unrecorded = f"submitted (unrecorded: {e})"
        try:
            await session.submit(prompt, grants=schedule.grants)
        except TurnQueueFull as e:
            if queue is not None:
                session.unsubscribe(queue)
            return f"skipped: {e}"
        if queue is None:
            return unrecorded
        record = RunRecord(
            schedule_id=schedule.id, run_id=run_id, started_at=now, session_id=session.id
        )
        task = asyncio.create_task(self._record(schedule, record, session, queue, prompt))
        self._recorders.add(task)
        task.add_done_callback(self._recorders.discard)
        return "submitted"

    async def _record(
        self, schedule: Schedule, record: RunRecord, session: Any, queue: asyncio.Queue, prompt: str
    ) -> None:
        """Collect exactly this turn's events, then persist the outcome."""
        events: list[dict[str, Any]] = []
        started = False
        status = "timeout"
        deadline = time.monotonic() + self.run_timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    _sid, ev = await asyncio.wait_for(queue.get(), remaining)
                except asyncio.TimeoutError:
                    break
                if not started:
                    # The session echoes the turn as a `user` event when it
                    # dequeues it; anything before that belongs to another turn.
                    if ev.get("type") == "user" and ev.get("text") == prompt:
                        started = True
                        events.append(ev)
                    continue
                events.append(ev)
                if ev.get("type") == "result":
                    status = "done"
                    break
                if ev.get("type") == "error":
                    status = "error"
                    break
        except asyncio.CancelledError:
            status = "interrupted"
            self._finish(schedule, record, status, events)
            raise
        finally:
            session.unsubscribe(queue)
        self._finish(schedule, record, status, events)

    def _finish(
        self, schedule: Schedule, record: RunRecord, status: str, events: list[dict[str, Any]]
    ) -> None:
        record.finish(status, events)
        try:
            self.runs.save(record)
        except OSError:  # pragma: no cover - a full disk must not crash the loop
            pass
        fields: dict[str, Any] = {"status": status, "tool_calls": record.tool_calls}
        if record.cost_usd is not None:
            fields["cost_usd"] = record.cost_usd
        if record.denied:
            fields["denied"] = len(record.denied)
        if record.acted:
            fields["acted"] = len(record.acted)
        if record.summary:
            fields["summary"] = record.summary[:200]
        schedule.update_run(record.run_id, **fields)
        try:
            self.store.save()
        except OSError:  # pragma: no cover
            pass
