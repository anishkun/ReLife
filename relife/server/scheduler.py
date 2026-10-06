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
from .. import workitems
from .runs import RunRecord, RunStore
from ..permissions import describe_grant
from .schedules import Schedule, ScheduleStore
from .security import resolve_workspace
from .session import SessionLimitReached, SessionManager, TooManySubscribers, TurnQueueFull


def scheduled_prompt(
    schedule: Schedule, body: str | None = None, grants: list[dict[str, Any]] | None = None
) -> str:
    """The turn text a firing submits. Tells the agent the run is automatic and
    possibly unattended, so a denied approval is reported, not fought.
    ``body`` replaces the schedule's task (a work schedule's issue prompt)."""
    return (
        f"[Scheduled run: {schedule.name}] This turn was triggered automatically by a "
        "schedule, not typed by the user, and may be unattended — an action that needs "
        "approval can be denied by timeout; if that happens, say so and stop rather than "
        "retrying. Do the task, then finish with a short summary of what you did and "
        f"anything that needs the user.{_grants_note(schedule, grants)}\n\n"
        f"{schedule.task if body is None else body}"
    )


def pick_issue(schedule: Schedule, items: list[workitems.WorkItem]) -> workitems.WorkItem | None:
    """The next issue a work schedule should take: newest-updated first, not
    attempted before, carrying the schedule's label if it names one."""
    label = (schedule.work or {}).get("label")
    for item in items:
        if item.ref in schedule.worked:
            continue
        if label and label not in item.labels:
            continue
        return item
    return None


def _grants_note(schedule: Schedule, grants: list[dict[str, Any]] | None = None) -> str:
    """One sentence (same paragraph — the UI splits the preamble at the first
    blank line) telling the agent what it may do without an approval."""
    grants = schedule.grants if grants is None else grants
    if not grants:
        return ""
    allowed = "; ".join(describe_grant(g) for g in grants)
    pr = next((g for g in grants if g.get("kind") == "pull_request" and g.get("repo")), None)
    how = (
        f" For the pull request that means one plain `gh pr create --repo {pr['repo']} "
        f"--head {pr['branch']} --title \"…\" --body \"…\"` call (optionally --base/--draft) "
        "with no `$` or backticks in it and nothing chained — any other shape asks."
        if pr else ""
    )
    return (
        f" The user pre-approved these actions for this run, so they need no approval: "
        f"{allowed} (at most {config.AGENT_GRANT_MAX_USES} uses).{how} Anything else still "
        "needs approval."
    )


def bind_grants(
    grants: list[dict[str, Any]], repo: str | None, branch: str | None
) -> list[dict[str, Any]]:
    """The grants one turn runs with: a stored ``pull_request`` grant is bound
    to this issue's repo + branch, or dropped when there's no issue to bind."""
    out = []
    for g in grants:
        if g.get("kind") == "pull_request":
            if repo and branch:
                out.append({"kind": "pull_request", "repo": repo, "branch": branch})
            continue
        out.append(g)
    return out


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
        self._item: dict[str, str] = {}  # schedule id → issue ref being started

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
        status, item = await self._submit(schedule, run_id, t)
        schedule.record_run(t, status, run_id=run_id if status == "submitted" else None)
        if item:
            schedule.runs[-1]["item"] = item  # which issue a work schedule took
        self.store.save()
        return status

    async def _prepare_work(self, schedule: Schedule) -> tuple[str, Path, str, str] | str:
        """Pick, fetch and check out the next issue → (body, checkout, ref, branch),
        or a status string when there's nothing to do. All ``gh``/git work is a
        blocking subprocess, so it runs on a worker thread, never on the loop
        every session shares."""
        try:
            base = resolve_workspace(schedule.workspace, self.root)
        except ValueError as e:
            return f"error: {e}"
        work = schedule.work or {}
        try:
            items = await asyncio.to_thread(
                workitems.list_assigned, repo=work.get("repo"), limit=50
            )
            item = pick_issue(schedule, items)
            if item is None:
                return "skipped: no new assigned issues"
            full = await asyncio.to_thread(workitems.fetch, item.repo, item.number)
            if full.state != "OPEN":
                schedule.mark_worked(item.ref)
                return f"skipped: {item.ref} is {full.state.lower()}"
            checkout, _cloned = await asyncio.to_thread(
                workitems.ensure_checkout, base, item.repo
            )
        except workitems.WorkItemError as e:
            return f"error: {e}"
        branch = workitems.branch_name(full)
        body = workitems.task_prompt(full, branch)
        if schedule.task:
            body += f"\nAdditional instructions from the user for these runs:\n{schedule.task}\n"
        return body, checkout, full.ref, branch

    async def _submit(self, schedule: Schedule, run_id: str, now: float) -> tuple[str, str | None]:
        status = await self._start(schedule, run_id, now)
        item = self._item.pop(schedule.id, None)
        if item and status.startswith("submitted"):
            # Attempted, whatever the run's outcome: an unattended schedule must
            # not grind on one issue every slot. `relife work REF` is how the
            # user sends it back in.
            schedule.mark_worked(item)
        return status, item

    async def _start(self, schedule: Schedule, run_id: str, now: float) -> str:
        session: Any = self.manager.get(schedule.session_id) if schedule.session_id else None
        if session is not None and getattr(session, "busy", False):
            return "skipped: previous run still in progress"
        body: str | None = None
        repo = branch = None
        if schedule.work is not None:
            prepared = await self._prepare_work(schedule)
            if isinstance(prepared, str):
                return prepared
            body, checkout, ref, branch = prepared
            self._item[schedule.id] = ref
            repo = ref.split("#", 1)[0]
            # Each issue runs in its own checkout, and a session's cwd is fixed
            # at creation — so a work schedule gets a fresh session per firing
            # (the previous one is idle: `busy` was checked above).
            if session is not None:
                await self.manager.close(session.id)
                session = None
            try:
                session = await self.manager.create(checkout)
            except SessionLimitReached as e:
                return f"skipped: {e}"
            except Exception as e:  # noqa: BLE001
                return f"error: {e}"
            schedule.session_id = session.id
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
        grants = bind_grants(schedule.grants, repo, branch)
        prompt = scheduled_prompt(schedule, body, grants)
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
            await session.submit(prompt, grants=grants)
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
