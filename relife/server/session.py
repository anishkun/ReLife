"""Persistent agent sessions for the always-on server.

A ``relife serve`` process holds one long-lived ``ClaudeSDKClient`` per session
(exactly as ``run_chat`` proves works across many turns) and streams its work as
structured events to the web UI over SSE. Outward-action approvals are routed to
the browser via an :class:`ApprovalBroker` and block the run until the user
decides (or a timeout denies).

Concurrency model mirrors the memory daemon: everything runs on the single server
event loop. The permission callback awaits a future that a *different* request
handler (``POST …/approvals/…``) resolves — both run on the same loop, so the
suspended worker and the approving request interleave cleanly.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable

from claude_agent_sdk import ClaudeSDKClient

from .. import config
from ..agent import _tool_brief, build_options, maybe_consolidate_off_loop, to_event
from ..hooks import memory_hooks
from ..permissions import make_approval_callback

# How many recent events to keep per session for SSE reconnect/replay.
_RING_MAX = 500
# Per-subscriber outbound depth. A subscriber that stops reading (a wedged
# browser tab, a paused debugger) must not grow the server's memory without
# bound, so the oldest event is dropped to make room instead.
_SUB_QUEUE_MAX = 1000

Publish = Callable[[dict[str, Any]], Awaitable[None]]


class SessionLimitReached(Exception):
    """Raised when the process already hosts AGENT_MAX_SESSIONS sessions."""


class TooManySubscribers(Exception):
    """Raised when a session already has AGENT_MAX_SUBSCRIBERS SSE streams."""


class TurnQueueFull(Exception):
    """Raised when a session's inbound turn queue is saturated."""


class ApprovalBroker:
    """Routes ask-case tool approvals to the UI and awaits the user's decision.

    ``request`` emits an ``approval_request`` event (so the UI can render an
    approve/deny card) and blocks on a future until :meth:`resolve` is called or
    ``timeout`` seconds elapse (→ deny). One broker per session.
    """

    def __init__(self, publish: Publish) -> None:
        self._publish = publish
        self._pending: dict[str, asyncio.Future[bool]] = {}

    async def request(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        reason: str,
        *,
        timeout: float,
    ) -> bool:
        approval_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        self._pending[approval_id] = fut
        await self._publish(
            {
                "type": "approval_request",
                "approval_id": approval_id,
                "tool": tool_name,
                "reason": reason,
                # Roomier than the transcript hint: the card is where the user
                # decides whether an email/event/etc. leaves the machine.
                "brief": _tool_brief(tool_input, limit=400),
            }
        )
        try:
            approved = await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            approved = False
        finally:
            self._pending.pop(approval_id, None)
        await self._publish(
            {
                "type": "approval_resolved",
                "approval_id": approval_id,
                "approved": approved,
            }
        )
        return approved

    def resolve(self, approval_id: str, approved: bool) -> bool:
        """Settle a pending approval. Returns False if it's unknown/already done."""
        fut = self._pending.get(approval_id)
        if fut is None or fut.done():
            return False
        fut.set_result(approved)
        return True


class AgentSession:
    """One long-lived agent conversation bound to a workspace."""

    def __init__(self, workspace: Path, *, session_id: str | None = None) -> None:
        self.id = session_id or uuid.uuid4().hex
        self.workspace = workspace
        self._inbound: asyncio.Queue[str] = asyncio.Queue(maxsize=config.AGENT_MAX_QUEUED_TURNS)
        self._subscribers: set[asyncio.Queue[tuple[int, dict[str, Any]]]] = set()
        self._ring: deque[tuple[int, dict[str, Any]]] = deque(maxlen=_RING_MAX)
        self._seq = 0
        self._broker = ApprovalBroker(self._publish)
        self._client: ClaudeSDKClient | None = None
        self._worker: asyncio.Task[None] | None = None
        self.last_active = time.monotonic()

    # --- lifecycle ----------------------------------------------------------
    async def start(self) -> None:
        options = build_options(
            cwd=self.workspace,
            can_use_tool=make_approval_callback(
                self.workspace, self._broker, timeout=config.AGENT_APPROVAL_TIMEOUT
            ),
            mcp_servers=config.default_mcp_servers(),
            hooks=memory_hooks(),
        )
        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()
        self._worker = asyncio.create_task(self._run())

    async def aclose(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass

    # --- inbound / outbound -------------------------------------------------
    async def submit(self, text: str) -> None:
        """Queue a turn. Raises :class:`TurnQueueFull` rather than blocking the
        request handler (which would pin an event-loop task per pending turn)."""
        try:
            self._inbound.put_nowait(text)
        except asyncio.QueueFull as e:
            raise TurnQueueFull("too many turns queued for this session") from e
        self.touch()

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        return self._broker.resolve(approval_id, approved)

    def subscribe(
        self, last_id: int | None = None
    ) -> tuple[asyncio.Queue[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]]]:
        """Register an SSE subscriber. Returns its live queue plus a backlog of
        buffered events with id > ``last_id`` (for reconnect/replay).

        Snapshot + registration happen with no ``await`` between them, so on the
        single event loop no event can slip between the backlog and the queue.
        """
        if len(self._subscribers) >= config.AGENT_MAX_SUBSCRIBERS:
            raise TooManySubscribers("too many event streams open for this session")
        q: asyncio.Queue[tuple[int, dict[str, Any]]] = asyncio.Queue(maxsize=_SUB_QUEUE_MAX)
        backlog = [(sid, ev) for sid, ev in self._ring if last_id is None or sid > last_id]
        self._subscribers.add(q)
        return q, backlog

    def unsubscribe(self, q: asyncio.Queue[tuple[int, dict[str, Any]]]) -> None:
        self._subscribers.discard(q)

    async def _publish(self, ev: dict[str, Any]) -> None:
        # Anything the agent emits is activity: a turn that streams for longer
        # than the idle timeout with no browser attached must not be reaped
        # out from under itself.
        self.touch()
        self._seq += 1
        item = (self._seq, ev)
        self._ring.append(item)
        for q in list(self._subscribers):
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                # Drop this subscriber's oldest event; the ring + Last-Event-ID
                # replay is how a lagging client catches back up.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover - racy, harmless
                    pass
                q.put_nowait(item)

    def touch(self) -> None:
        """Mark the session active (defers idle reaping)."""
        self.last_active = time.monotonic()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    # --- worker -------------------------------------------------------------
    async def _run(self) -> None:
        assert self._client is not None
        while True:
            turn = await self._inbound.get()
            await self._publish({"type": "user", "text": turn})
            try:
                await self._client.query(turn)
                async for msg in self._client.receive_response():
                    for ev in to_event(msg):
                        await self._publish(ev)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — never let one turn kill the session
                await self._publish({"type": "error", "message": str(e)})
                continue
            finally:
                self.touch()
            # Brain upkeep after each completed turn (deterministic, fail-safe).
            # Off the loop: this process hosts every session's stream and every
            # pending approval on one loop, and a pass can take seconds on a
            # large store — inline, all of them froze for the duration.
            note = await maybe_consolidate_off_loop()
            if note:
                await self._publish({"type": "note", "text": note})


# Factory used by SessionManager; overridable in tests.
SessionFactory = Callable[[Path], AgentSession]


def _default_session_factory(workspace: Path) -> AgentSession:
    return AgentSession(workspace)


class SessionManager:
    """Owns the live sessions for one server process (single-user, in-memory).

    Hardened with the two ceilings a long-lived process needs: a cap on how many
    sessions may exist at once (each holds a ``ClaudeSDKClient`` subprocess) and
    an idle reaper so an abandoned browser tab doesn't leak one forever. Each
    session stamps its own ``last_active``; the manager only delegates
    :meth:`touch` so any HTTP route can keep a session alive.
    """

    def __init__(
        self,
        session_factory: SessionFactory | None = None,
        *,
        max_sessions: int | None = None,
        idle_timeout: float | None = None,
    ) -> None:
        self._factory = session_factory or _default_session_factory
        self._sessions: dict[str, AgentSession] = {}
        self.max_sessions = config.AGENT_MAX_SESSIONS if max_sessions is None else max_sessions
        self.idle_timeout = (
            config.AGENT_SESSION_IDLE_TIMEOUT if idle_timeout is None else idle_timeout
        )

    def count(self) -> int:
        return len(self._sessions)

    async def create(self, workspace: Path) -> AgentSession:
        await self.reap_idle()  # make room before refusing
        if self.max_sessions and len(self._sessions) >= self.max_sessions:
            raise SessionLimitReached(
                f"at most {self.max_sessions} concurrent sessions "
                "(close one, or raise RELIFE_AGENT_MAX_SESSIONS)"
            )
        session = self._factory(workspace)
        await session.start()
        self.touch(session)
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> AgentSession | None:
        return self._sessions.get(session_id)

    def touch(self, session: AgentSession | str | None) -> None:
        """Defer idle reaping for a session (accepts the object or its id)."""
        if isinstance(session, str):
            session = self._sessions.get(session)
        if session is not None and hasattr(session, "touch"):
            session.touch()

    async def close(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        await session.aclose()
        return True

    def _idle_ids(self, now: float) -> list[str]:
        """Sessions past the idle timeout with no live event stream attached.

        An open SSE connection counts as activity (the stream heartbeat touches
        the session), so a watching browser never has its agent reaped mid-run.
        """
        if not self.idle_timeout:
            return []
        stale = []
        for sid, session in self._sessions.items():
            if getattr(session, "subscriber_count", 0):
                continue
            last = getattr(session, "last_active", now)
            if now - last >= self.idle_timeout:
                stale.append(sid)
        return stale

    async def reap_idle(self) -> list[str]:
        """Close sessions idle beyond the timeout. Returns the ids closed."""
        now = time.monotonic()
        closed = []
        for sid in self._idle_ids(now):
            if await self.close(sid):
                closed.append(sid)
        return closed

    async def run_reaper(self, interval: float | None = None) -> None:
        """Background loop that periodically reaps idle sessions."""
        every = config.AGENT_REAP_INTERVAL if interval is None else interval
        while True:
            await asyncio.sleep(every)
            try:
                await self.reap_idle()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - upkeep must never kill the server
                pass

    async def aclose(self) -> None:
        for session in list(self._sessions.values()):
            await session.aclose()
        self._sessions.clear()
