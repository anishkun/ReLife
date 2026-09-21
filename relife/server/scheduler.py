"""The scheduler: fires due schedules into agent sessions.

Runs as one background task under the app lifespan (like the idle reaper) and
ticks every ``AGENT_SCHEDULER_TICK`` seconds. A firing is just a *turn
submitted to a session*: each schedule owns a session (created in the
schedule's confined workspace, reused while alive, recreated after the idle
reaper closes it), so a scheduled run streams, journals and asks for approvals
exactly like a turn typed in the console — the UI can attach to it with
"watch". Nobody watching means an ask-case times out to **deny**, the same safe
default as the non-interactive CLI; the prompt tells the agent so.

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
from .schedules import Schedule, ScheduleStore
from .security import resolve_workspace
from .session import SessionLimitReached, SessionManager, TurnQueueFull


def scheduled_prompt(schedule: Schedule) -> str:
    """The turn text a firing submits. Tells the agent the run is automatic and
    possibly unattended, so a denied approval is reported, not fought."""
    return (
        f"[Scheduled run: {schedule.name}] This turn was triggered automatically by a "
        "schedule, not typed by the user, and may be unattended — an action that needs "
        "approval can be denied by timeout; if that happens, say so and stop rather than "
        "retrying. Do the task, then finish with a short summary of what you did and "
        "anything that needs the user.\n\n"
        f"{schedule.task}"
    )


class Scheduler:
    def __init__(
        self,
        store: ScheduleStore,
        manager: SessionManager,
        *,
        workspace_root: Path | None = None,
        tick: float | None = None,
    ) -> None:
        self.store = store
        self.manager = manager
        self.root = Path(workspace_root) if workspace_root is not None else config.AGENT_WORKSPACE_ROOT
        self.tick_seconds = config.AGENT_SCHEDULER_TICK if tick is None else tick

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
        status = await self._submit(schedule)
        schedule.record_run(t, status)
        self.store.save()
        return status

    async def _submit(self, schedule: Schedule) -> str:
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
        try:
            await session.submit(scheduled_prompt(schedule))
        except TurnQueueFull as e:
            return f"skipped: {e}"
        return "submitted"
