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
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable

from claude_agent_sdk import ClaudeSDKClient

from .. import config
from ..agent import _maybe_consolidate, _tool_brief, build_options, to_event
from ..hooks import memory_hooks
from ..permissions import make_approval_callback

# How many recent events to keep per session for SSE reconnect/replay.
_RING_MAX = 500

Publish = Callable[[dict[str, Any]], Awaitable[None]]


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
                "brief": _tool_brief(tool_input),
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
        self._inbound: asyncio.Queue[str] = asyncio.Queue()
        self._subscribers: set[asyncio.Queue[tuple[int, dict[str, Any]]]] = set()
        self._ring: deque[tuple[int, dict[str, Any]]] = deque(maxlen=_RING_MAX)
        self._seq = 0
        self._broker = ApprovalBroker(self._publish)
        self._client: ClaudeSDKClient | None = None
        self._worker: asyncio.Task[None] | None = None

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
        await self._inbound.put(text)

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
        q: asyncio.Queue[tuple[int, dict[str, Any]]] = asyncio.Queue()
        backlog = [(sid, ev) for sid, ev in self._ring if last_id is None or sid > last_id]
        self._subscribers.add(q)
        return q, backlog

    def unsubscribe(self, q: asyncio.Queue[tuple[int, dict[str, Any]]]) -> None:
        self._subscribers.discard(q)

    async def _publish(self, ev: dict[str, Any]) -> None:
        self._seq += 1
        item = (self._seq, ev)
        self._ring.append(item)
        for q in list(self._subscribers):
            q.put_nowait(item)

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
            # Brain upkeep after each completed turn (deterministic, fail-safe).
            _maybe_consolidate()


# Factory used by SessionManager; overridable in tests.
SessionFactory = Callable[[Path], AgentSession]


def _default_session_factory(workspace: Path) -> AgentSession:
    return AgentSession(workspace)


class SessionManager:
    """Owns the live sessions for one server process (single-user, in-memory)."""

    def __init__(self, session_factory: SessionFactory | None = None) -> None:
        self._factory = session_factory or _default_session_factory
        self._sessions: dict[str, AgentSession] = {}

    async def create(self, workspace: Path) -> AgentSession:
        session = self._factory(workspace)
        await session.start()
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> AgentSession | None:
        return self._sessions.get(session_id)

    async def aclose(self) -> None:
        for session in list(self._sessions.values()):
            await session.aclose()
        self._sessions.clear()
