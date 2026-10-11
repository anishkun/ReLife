"""Crew runs inside the always-on server.

``relife crew`` on a terminal plans, confirms and runs a crew in the
foreground. Here the same two halves (``crew.runner.prepare_crew`` /
``execute_crew``) run under ``relife serve``, so a crew is planned from the web
UI (or fired by a schedule), its members' work streams to the browser, and an
outward action a member wants is an **approval card**, not a denial.

A :class:`CrewHost` is a session-shaped object hosted by the
:class:`~.session.SessionManager` (``adopt``): it publishes through the same
:class:`~.session.EventStream`, so ``GET /sessions/{id}/events``, the approval
route, the UI's *watch* and the scheduler's run recorder all work on a crew
exactly as on a chat — and it counts against the same session ceiling, reaper
and shutdown. It takes no typed turns (:class:`~.session.SessionReadOnly`).

**Threads.** CrewAI's ``kickoff()`` is synchronous and long, so it runs on a
daemon worker thread; each ReLife member's turn runs on that thread's own
event loop (``crew.turns.run_sync``). Two bridges cross back to the server
loop, where every SSE stream and approval future lives:

* events: the worker calls :meth:`CrewHost._emit`, which schedules
  ``_publish`` with ``run_coroutine_threadsafe`` (FIFO, so order is kept);
* approvals: a member's ``can_use_tool`` is the ordinary
  ``make_approval_callback`` — the pure ``classify()`` decides — over a
  :class:`LoopBroker` that hops each ask onto the server loop's
  :class:`~.session.ApprovalBroker` and awaits the browser's answer (or the
  timeout's deny) from the worker loop.

Stopping is cooperative: pending approvals are denied at once and no further
task starts, whichever kind of member it belongs to (a ReLife member's turn
refuses to begin, and the crew's ``task_callback`` raises after the task in
flight — CrewAI has no cancel, so that task finishes first). Members' own ``result``/``error`` events are renamed
(``member_done``/``member_error``) so a watcher — or the scheduler's recorder,
which ends a run on ``result`` — sees one ``result`` for the whole crew.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import threading
from pathlib import Path
from typing import Any, Awaitable, Callable

from .. import config
from ..agents import AgentStore
from ..crew.record import CrewRunRecord, CrewRunStore
from ..crew.spec import CrewSpec
from ..permissions import make_approval_callback
from .session import ApprovalBroker, EventStream, SessionLimitReached, SessionReadOnly

TurnFn = Callable[..., Awaitable[Any]]


class CrewUnavailable(Exception):
    """CrewAI isn't importable in this interpreter (it needs Python ≤ 3.13)."""


class CrewLimitReached(Exception):
    """``AGENT_MAX_CREWS`` crews are already running."""


def crewai_available() -> bool:
    return importlib.util.find_spec("crewai") is not None


CREWAI_HINT = (
    "CrewAI isn't installed for this Python (it needs Python <= 3.13): run the "
    'server from a 3.12 venv with `pip install -e ".[crewai,server]"`'
)


class LoopBroker:
    """An approval broker for a crew member's turn, which runs on the crew's
    worker thread: each request is handed to the server loop's broker (where
    the browser's decision resolves it) and awaited from the worker loop."""

    def __init__(self, broker: ApprovalBroker, loop: asyncio.AbstractEventLoop, extra: dict[str, Any]) -> None:
        self._broker = broker
        self._loop = loop
        self._extra = extra

    async def request(
        self, tool_name: str, tool_input: dict[str, Any], reason: str, *, timeout: float
    ) -> bool:
        cf = asyncio.run_coroutine_threadsafe(
            self._broker.request(tool_name, tool_input, reason, timeout=timeout, extra=self._extra),
            self._loop,
        )
        try:
            return bool(await asyncio.wrap_future(cf))
        except Exception:  # noqa: BLE001 — a broken bridge must deny, never allow
            return False


# Member events whose type means "the whole run ended" to a watcher.
_RENAMED = {"result": "member_done", "error": "member_error", "user": "member_prompt"}


def member_event(agent: str, ev: dict[str, Any]) -> dict[str, Any]:
    out = {**ev, "agent": agent}
    if ev.get("type") in _RENAMED:
        out["type"] = _RENAMED[ev["type"]]
    return out


class CrewHost(EventStream):
    """One crew run, hosted like a session (see module doc)."""

    kind = "crew"

    def __init__(
        self,
        record: CrewRunRecord,
        spec: CrewSpec,
        *,
        records: CrewRunStore,
        agents_path: Path | None,
        client: Any,
        prompt: str | None = None,
        turn: TurnFn | None = None,
        llm_factory: Any = None,
        manager_llm: Any = None,
        approval_timeout: float | None = None,
        on_finish: Callable[[CrewRunRecord], None] | None = None,
    ) -> None:
        super().__init__()
        self.record = record
        self.spec = spec
        self.workspace = Path(record.workspace)
        self.prompt = prompt or f"[Crew {record.id}] {record.task}"
        self._records = records
        self._agents_path = agents_path
        self._client = client
        self._turn = turn
        self._llm_factory = llm_factory
        self._manager_llm = manager_llm
        self._timeout = config.AGENT_APPROVAL_TIMEOUT if approval_timeout is None else approval_timeout
        self._on_finish = on_finish
        self._broker = ApprovalBroker(self._publish)
        self._stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None

    # --- session protocol -----------------------------------------------------
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._run())

    async def aclose(self) -> None:
        """Stop the crew and stop streaming. The worker thread is a daemon: it
        finishes the member turn in flight on its own and records the outcome."""
        self.stop()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def submit(self, text: str, **_: Any) -> None:
        raise SessionReadOnly("this is a crew run: it follows its plan and takes no messages")

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        return self._broker.resolve(approval_id, approved)

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> int:
        """Ask the crew to stop: deny what's pending, start nothing new."""
        self._stop.set()
        return self._broker.deny_all()

    # --- the run ------------------------------------------------------------------
    def _emit(self, ev: dict[str, Any]) -> None:
        """Publish from the worker thread (scheduled on the server loop)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._publish(ev), loop)
        except RuntimeError:  # pragma: no cover - loop shutting down
            pass

    async def _run(self) -> None:
        await self._publish({"type": "user", "text": self.prompt})
        team = ", ".join(
            f"{a.name} ({'ReLife' if a.runtime == 'relife' else a.llm})" for a in self.spec.agents
        )
        await self._publish({"type": "note", "text": f"crew {self.record.id}: {team}"})
        fut: concurrent.futures.Future[CrewRunRecord] = concurrent.futures.Future()
        threading.Thread(
            target=self._work, args=(fut,), name=f"relife-crew-{self.record.id}", daemon=True
        ).start()
        try:
            record = await asyncio.wrap_future(fut)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            await self._publish({"type": "error", "message": f"{type(e).__name__}: {e}"})
            return
        self.record = record
        if record.final_output:
            await self._publish({"type": "text", "text": record.final_output})
        if record.status == "done":
            await self._publish({"type": "result", "cost_usd": record.cost_usd})
        else:
            await self._publish({"type": "error", "message": record.error or record.status})
        if self._on_finish is not None:
            try:
                self._on_finish(record)
            except Exception:  # noqa: BLE001 — bookkeeping never breaks the stream
                pass

    def _work(self, fut: concurrent.futures.Future) -> None:
        from ..crew.runner import execute_crew

        try:
            rec = execute_crew(
                self.record,
                self.spec,
                echo=lambda line: self._emit({"type": "note", "text": line.strip()}),
                runner=self._member_turn,
                llm_factory=self._llm_factory,
                manager_llm=self._manager_llm,
                store=AgentStore(self._agents_path),
                client=self._client,
                records=self._records,
                on_event=lambda name, ev: self._emit(member_event(name, ev)),
                on_step=self._on_step,
                on_task=self._on_task,
                should_stop=self._stop.is_set,
            )
        except BaseException as e:  # noqa: BLE001 — surfaced on the loop
            if not fut.cancelled():  # the host was closed while we worked
                fut.set_exception(e if isinstance(e, Exception) else RuntimeError(str(e)))
            return
        if not fut.cancelled():
            fut.set_result(rec)

    def _on_step(self, agent: str, step: Any) -> None:
        tool = getattr(step, "tool", None)
        if tool:
            brief = str(getattr(step, "tool_input", ""))[:160]
            self._emit({"type": "tool_use", "name": str(tool), "brief": brief, "agent": agent})

    def _on_task(self, task: str, agent: str, result: Any) -> None:
        cost = f" · ${result.cost_usd:.3f}" if result is not None and result.cost_usd else ""
        self._emit({"type": "note", "text": f"task {task} finished by {agent}{cost}"})

    async def _member_turn(self, agent: Any, prompt: str, extra_mcp: dict[str, Any]) -> Any:
        """A ReLife member's task (runs on the worker thread's loop): the same
        fresh turn ``relife crew`` runs, with approvals routed to the browser."""
        if self._stop.is_set():
            raise RuntimeError("crew stopped by the user")
        from ..crew.turns import run_relife_turn

        assert self._loop is not None
        ws = Path(agent.workspace)
        self._emit({"type": "note", "text": f"{agent.agent_name} starts a task"})
        can_use_tool = make_approval_callback(
            ws,
            LoopBroker(self._broker, self._loop, {"agent": agent.agent_name}),
            timeout=self._timeout,
        )
        turn = self._turn or run_relife_turn
        return await turn(
            prompt,
            workspace=ws,
            memory_client=agent.memory_client,
            can_use_tool=can_use_tool,
            extra_mcp=extra_mcp,
            on_event=agent.on_event,
        )


class CrewService:
    """What the server's crew routes and the scheduler share: planning,
    starting a hosted run, and finding a live one. Model calls, the member
    turn and the CrewAI LLMs are injectable (tests run all of it model-free)."""

    def __init__(
        self,
        manager: Any,
        *,
        workspace_root: Path,
        records: CrewRunStore | None = None,
        agents_path: Path | None = None,
        client: Any = None,
        ask_model: Any = None,
        turn: TurnFn | None = None,
        llm_factory: Any = None,
        manager_llm: Any = None,
        approval_timeout: float | None = None,
        max_running: int | None = None,
    ) -> None:
        self.manager = manager
        self.root = Path(workspace_root)
        self.records = records or CrewRunStore()
        self.agents_path = agents_path
        self._client = client
        self.ask_model = ask_model
        self.turn = turn
        self.llm_factory = llm_factory
        self.manager_llm = manager_llm
        self.approval_timeout = approval_timeout
        self.max_running = config.AGENT_MAX_CREWS if max_running is None else max_running
        self._hosts: dict[str, str] = {}  # crew id → session id
        # Crews between "checked" and "hosted": adopt() awaits (it reaps idle
        # sessions first), so without a reservation two quick requests could
        # both pass the checks and start one crew twice, or overshoot the cap.
        self._pending: set[str] = set()

    @property
    def client(self) -> Any:
        if self._client is not None:
            return self._client
        from ..memory.client import default_client

        return default_client()

    def agents(self) -> AgentStore:
        # Re-read per call: the CLI may have changed the registry since.
        return AgentStore(self.agents_path)

    # --- plan ---------------------------------------------------------------
    async def plan(self, *, task: str | None = None, spec_raw: Any = None) -> tuple[CrewRunRecord, CrewSpec]:
        """Plan a task (one tool-less model call) or validate a spec; record it
        as ``planned``. Off the loop: planning blocks on the model and SQLite."""
        from ..crew.runner import prepare_crew

        def work() -> tuple[CrewRunRecord, CrewSpec]:
            return prepare_crew(
                task,
                workspace=self.root,
                spec_raw=spec_raw,
                # Over HTTP a spec may only name the configured non-Claude models.
                allowed_llms=config.CREW_LLMS,
                echo=lambda _line: None,
                ask_model=self.ask_model,
                store=self.agents(),
                client=self.client,
                records=self.records,
            )

        # Not asyncio.to_thread: it copies context variables, and the planner's
        # anyio.run would then believe it is already inside this loop.
        return await asyncio.get_running_loop().run_in_executor(None, work)

    # --- run ----------------------------------------------------------------
    def running(self) -> int:
        live = sum(
            1 for s in self.manager.all() if getattr(s, "kind", "") == "crew" and getattr(s, "busy", False)
        )
        return live + len(self._pending)

    def host(self, crew_id: str) -> CrewHost | None:
        sid = self._hosts.get(crew_id)
        s = self.manager.get(sid) if sid else None
        return s if isinstance(s, CrewHost) else None

    def starting(self, crew_id: str) -> bool:
        """Checked and being hosted, not yet registered (see ``_pending``)."""
        return crew_id in self._pending

    def session_id(self, crew_id: str) -> str | None:
        h = self.host(crew_id)
        return h.id if h is not None else None

    def build_host(
        self,
        record: CrewRunRecord,
        *,
        prompt: str | None = None,
        on_finish: Callable[[CrewRunRecord], None] | None = None,
    ) -> CrewHost:
        """A host for a ``planned`` record, not yet started — so a caller (the
        scheduler's recorder) can subscribe before its first event."""
        if not crewai_available():
            raise CrewUnavailable(CREWAI_HINT)
        # The live host first: the record only says "running" once the worker
        # thread has saved it, so a quick second request would still read
        # "planned" and start the same crew twice.
        if record.id in self._hosts or record.id in self._pending:
            raise ValueError(f"crew {record.id} has already been started")
        if record.status != "planned":
            raise ValueError(f"crew {record.id} is {record.status}; only a planned crew can run")
        if self.max_running and self.running() >= self.max_running:
            raise CrewLimitReached(
                f"at most {self.max_running} crew(s) at a time (raise RELIFE_AGENT_MAX_CREWS)"
            )
        return CrewHost(
            record,
            CrewSpec.from_dict(record.spec),
            records=self.records,
            agents_path=self.agents_path,
            client=self.client,
            prompt=prompt,
            turn=self.turn,
            llm_factory=self.llm_factory,
            manager_llm=self.manager_llm,
            approval_timeout=self.approval_timeout,
            on_finish=on_finish,
        )

    async def launch(self, host: CrewHost) -> CrewHost:
        """Start a built host under the session manager (ceiling, reaper, shutdown)."""
        crew_id = host.record.id
        if crew_id in self._hosts or crew_id in self._pending:
            raise ValueError(f"crew {crew_id} has already been started")
        self._pending.add(crew_id)  # synchronously, before the first await
        try:
            await self.manager.adopt(host)  # raises SessionLimitReached
        finally:
            self._pending.discard(crew_id)
        self._hosts[crew_id] = host.id
        return host

    async def start(self, crew_id: str) -> CrewHost:
        record = self.records.get(crew_id)
        if record is None:
            raise LookupError("unknown crew")
        return await self.launch(self.build_host(record))


__all__ = [
    "CREWAI_HINT",
    "CrewHost",
    "CrewLimitReached",
    "CrewService",
    "CrewUnavailable",
    "LoopBroker",
    "SessionLimitReached",
    "crewai_available",
    "member_event",
]
