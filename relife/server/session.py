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

**Sessions sleep instead of dying.** A session's ``claude`` subprocess is the
expensive part, so it is started only when a turn arrives, and the idle reaper
*puts a session to sleep*: the subprocess goes, but a small record stays in
``data/sessions/<id>.json`` (workspace, the SDK's conversation id, the last
events). The next request for that id (the browser reattaching, a new message,
the server restarting) restores it: the transcript replays, and the next turn
resumes the same conversation (``resume=``). A server restart used to lose
every session, and an idle tab held a subprocess for an hour.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable

from claude_agent_sdk import ClaudeSDKClient, ResultMessage

from .. import _atomic, budget, config
from ..agent import _tool_brief, build_options, maybe_consolidate_off_loop, to_event
from ..hooks import memory_hooks
from ..permissions import grant_allows, make_approval_callback

# How many recent events to keep per session for SSE reconnect/replay.
_RING_MAX = 500
# Per-subscriber outbound depth. A subscriber that stops reading (a wedged
# browser tab, a paused debugger) must not grow the server's memory without
# bound, so the oldest event is dropped to make room instead.
_SUB_QUEUE_MAX = 1000

# How many events a sleeping session keeps for its transcript replay.
_PERSIST_EVENTS = 300

Publish = Callable[[dict[str, Any]], Awaitable[None]]


class SessionLimitReached(Exception):
    """Raised when the process already hosts AGENT_MAX_SESSIONS sessions."""


class TooManySubscribers(Exception):
    """Raised when a session already has AGENT_MAX_SUBSCRIBERS SSE streams."""


class TurnQueueFull(Exception):
    """Raised when a session's inbound turn queue is saturated."""


class SessionReadOnly(Exception):
    """Raised when a turn is submitted to a session that takes none (a crew run
    streams and asks for approvals like a chat, but its work is its plan)."""


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
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """``extra`` fields ride on both events (a crew names the member asking)."""
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
                **(extra or {}),
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
                **(extra or {}),
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

    def deny_all(self) -> int:
        """Settle every pending approval as denied (a stopped run). Returns how many."""
        n = 0
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_result(False)
                n += 1
        return n


class EventStream:
    """The streaming half of a session: numbered events, a replay ring, bounded
    SSE subscribers, and the activity clock the idle reaper reads. A chat
    session and a crew run both publish through it, so the same SSE and
    approval routes (and the UI's watch path) serve either."""

    def __init__(self, session_id: str | None = None) -> None:
        self.id = session_id or uuid.uuid4().hex
        self._subscribers: set[asyncio.Queue[tuple[int, dict[str, Any]]]] = set()
        self._ring: deque[tuple[int, dict[str, Any]]] = deque(maxlen=_RING_MAX)
        self._seq = 0
        self.last_active = time.monotonic()

    def preload(self, history: list[tuple[int, dict[str, Any]]]) -> None:
        """Seed the replay ring with a restored session's events, keeping their
        numbers so a browser's ``Last-Event-ID`` still means the same event."""
        for seq, ev in history:
            self._ring.append((seq, ev))
            self._seq = max(self._seq, seq)

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


class AgentSession(EventStream):
    """One long-lived agent conversation bound to a workspace."""

    def __init__(
        self,
        workspace: Path,
        *,
        session_id: str | None = None,
        resume: str | None = None,
        history: list[tuple[int, dict[str, Any]]] | None = None,
        created_at: float | None = None,
    ) -> None:
        super().__init__(session_id)
        self.workspace = workspace
        # The SDK's own conversation id (from each result): what a restored
        # session resumes. Distinct from ``self.id``, the server's handle.
        self.sdk_session_id = resume
        self.created_at = time.time() if created_at is None else created_at
        # Called after each turn and on close with this session (the manager
        # persists it); None = not persisted (tests, a crew host).
        self.on_save: Callable[[AgentSession], None] | None = None
        if history:
            self.preload(history)
        # Each queued turn carries the grants that apply to it (a scheduled turn's
        # pre-authorizations; ``[]`` for anything typed).
        self._inbound: asyncio.Queue[tuple[str, list[dict[str, Any]]]] = asyncio.Queue(
            maxsize=config.AGENT_MAX_QUEUED_TURNS
        )
        self._broker = ApprovalBroker(self._publish)
        self._client: ClaudeSDKClient | None = None
        self._worker: asyncio.Task[None] | None = None
        self._turn_active = False
        self._turn_grants: list[dict[str, Any]] = []
        self._grant_uses = 0
        # The client lives across turns, so its result cost is a running
        # total: the meter turns it into each turn's own cost (budget.py).
        self._meter = budget.CostMeter("server", self.id)

    # --- lifecycle ----------------------------------------------------------
    async def start(self) -> None:
        # No subprocess yet: the client connects when the first turn arrives,
        # so a session opened and never used (or restored and only looked at)
        # costs nothing.
        self._worker = asyncio.create_task(self._run())

    def _options(self, resume: str | None):
        return build_options(
            cwd=self.workspace,
            can_use_tool=make_approval_callback(
                self.workspace,
                self._broker,
                timeout=config.AGENT_APPROVAL_TIMEOUT,
                preauthorize=self._preauthorize,
            ),
            mcp_servers=config.default_mcp_servers(),
            hooks=memory_hooks(),
            resume=resume,
        )

    async def _ensure_client(self) -> None:
        if self._client is not None:
            return
        resume = self.sdk_session_id
        client = ClaudeSDKClient(options=self._options(resume))
        try:
            await client.connect()
        except Exception:
            if resume is None:
                raise
            # The CLI no longer has that conversation (cleared, or another
            # machine): carry on in a fresh one rather than fail the turn.
            await self._publish(
                {"type": "note", "text": "the earlier conversation could not be resumed; starting fresh"}
            )
            self.sdk_session_id = None
            client = ClaudeSDKClient(options=self._options(None))
            await client.connect()
        self._client = client

    def snapshot(self) -> dict[str, Any]:
        """What a sleeping session keeps (see module doc)."""
        events = list(self._ring)[-_PERSIST_EVENTS:]
        return {
            "id": self.id,
            "workspace": str(self.workspace),
            "sdk_session_id": self.sdk_session_id,
            "created_at": self.created_at,
            "updated_at": time.time(),
            "events": [[seq, ev] for seq, ev in events],
        }

    def _save(self) -> None:
        if self.on_save is not None:
            try:
                self.on_save(self)
            except Exception:  # noqa: BLE001 - persistence never breaks a turn
                pass

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
            self._client = None
        self._save()

    # --- inbound / outbound -------------------------------------------------
    async def submit(self, text: str, *, grants: list[dict[str, Any]] | None = None) -> None:
        """Queue a turn. Raises :class:`TurnQueueFull` rather than blocking the
        request handler (which would pin an event-loop task per pending turn).

        ``grants`` pre-authorize specific outward actions for *this turn only*
        (the scheduler passes a schedule's; a typed turn gets none)."""
        try:
            self._inbound.put_nowait((text, list(grants or [])))
        except asyncio.QueueFull as e:
            raise TurnQueueFull("too many turns queued for this session") from e
        self.touch()

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        return self._broker.resolve(approval_id, approved)

    async def _preauthorize(self, tool_name: str, tool_input: dict[str, Any]) -> str | None:
        """Apply the running turn's grants to an ask-case. Capped per turn at
        ``AGENT_GRANT_MAX_USES``; every use is published as ``approval_auto`` so
        the transcript and the run record show what was done on the user's behalf."""
        if not self._turn_grants or self._grant_uses >= config.AGENT_GRANT_MAX_USES:
            return None
        grant = grant_allows(self._turn_grants, tool_name, tool_input)
        if grant is None:
            return None
        self._grant_uses += 1
        await self._publish(
            {
                "type": "approval_auto",
                "tool": tool_name,
                "grant": grant,
                "brief": _tool_brief(tool_input, limit=400),
            }
        )
        return grant

    @property
    def busy(self) -> bool:
        """A turn is running or queued. The scheduler skips a slot rather than
        stacking another turn behind one still in progress."""
        return self._turn_active or not self._inbound.empty()

    # --- worker -------------------------------------------------------------
    async def _run(self) -> None:
        while True:
            turn, self._turn_grants = await self._inbound.get()
            self._grant_uses = 0
            self._turn_active = True
            await self._publish({"type": "user", "text": turn})
            try:
                await self._ensure_client()
                assert self._client is not None
                await self._client.query(turn)
                async for msg in self._client.receive_response():
                    if isinstance(msg, ResultMessage) and getattr(msg, "session_id", None):
                        self.sdk_session_id = msg.session_id
                    for ev in to_event(msg):
                        if ev["type"] == "result":
                            # Publish what this turn cost, not the session's
                            # running total (which every turn's record and the
                            # scheduler's run cost used to show).
                            ev = {**ev, "cost_usd": self._meter.observe(ev.get("cost_usd"))}
                        await self._publish(ev)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — never let one turn kill the session
                await self._publish({"type": "error", "message": str(e)})
                continue
            finally:
                self._turn_active = False
                self._turn_grants = []
                self.touch()
                self._save()
            # Brain upkeep after each completed turn (deterministic, fail-safe).
            # Off the loop: this process hosts every session's stream and every
            # pending approval on one loop, and a pass can take seconds on a
            # large store — inline, all of them froze for the duration.
            note = await maybe_consolidate_off_loop()
            if note:
                await self._publish({"type": "note", "text": note})


# Factory used by SessionManager; overridable in tests. A restore calls it with
# the keywords AgentSession takes (session_id, resume, history, created_at).
SessionFactory = Callable[..., AgentSession]


def _default_session_factory(workspace: Path, **kw: Any) -> AgentSession:
    return AgentSession(workspace, **kw)


_SESSION_ID = re.compile(r"^[0-9a-f]{32}$")


class SessionStore:
    """Sleeping sessions, one JSON file each (``data/sessions/<id>.json``).

    Holds a transcript, so it lives under ``data/`` (never in a workspace) and
    only ids shaped like the server's own (32 hex) ever name a file. Bounded:
    the newest ``AGENT_SESSION_KEEP`` are kept, and none older than
    ``AGENT_SESSION_RETENTION_DAYS``.
    """

    def __init__(self, root: Path, *, keep: int | None = None, retention_days: float | None = None) -> None:
        self.root = Path(root)
        self.keep = config.AGENT_SESSION_KEEP if keep is None else keep
        self.retention = 86400 * (
            config.AGENT_SESSION_RETENTION_DAYS if retention_days is None else retention_days
        )

    def _path(self, session_id: str) -> Path | None:
        if not isinstance(session_id, str) or not _SESSION_ID.match(session_id):
            return None
        return self.root / f"{session_id}.json"

    def save(self, snap: dict[str, Any]) -> None:
        path = self._path(snap.get("id", ""))
        if path is None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
        _atomic.replace(tmp, path)
        self.prune()

    def load(self, session_id: str) -> dict[str, Any] | None:
        path = self._path(session_id)
        if path is None or not path.exists():
            return None
        try:
            rec = json.loads(_atomic.read_text(path))
        except (OSError, ValueError):
            return None
        if not isinstance(rec, dict) or rec.get("id") != session_id:
            return None
        if time.time() - float(rec.get("updated_at") or 0) > self.retention:
            self.delete(session_id)
            return None
        return rec

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path is None or not path.exists():
            return False
        try:
            path.unlink()
        except OSError:
            return False
        return True

    def ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return [p.stem for p in self.root.glob("*.json") if _SESSION_ID.match(p.stem)]

    def prune(self) -> None:
        if not self.root.is_dir():
            return
        files = sorted(
            (p for p in self.root.glob("*.json") if _SESSION_ID.match(p.stem)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        cutoff = time.time() - self.retention
        for i, p in enumerate(files):
            try:
                if (self.keep and i >= self.keep) or p.stat().st_mtime < cutoff:
                    p.unlink()
            except OSError:
                pass


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
        store: SessionStore | None = None,
        workspace_ok: Callable[[str], Path] | None = None,
    ) -> None:
        self._factory = session_factory or _default_session_factory
        self._sessions: dict[str, AgentSession] = {}
        # Sleeping sessions (None: sessions just end when closed, as in tests).
        self.store = store
        # Re-checks a restored session's workspace (the server passes its
        # root-confinement rule): a record on disk is not trusted further than
        # a request body would be.
        self._workspace_ok = workspace_ok
        self._restoring = asyncio.Lock()
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
        return await self.adopt(self._factory(workspace))

    async def adopt(self, session: Any) -> Any:
        """Start and host a session built elsewhere (a crew run), under the same
        ceiling, reaper and shutdown as a chat session."""
        await self.reap_idle()
        if self.max_sessions and len(self._sessions) >= self.max_sessions:
            raise SessionLimitReached(
                f"at most {self.max_sessions} concurrent sessions "
                "(close one, or raise RELIFE_AGENT_MAX_SESSIONS)"
            )
        await session.start()
        self.touch(session)
        self._sessions[session.id] = session
        if self.store is not None and hasattr(session, "snapshot") and hasattr(session, "on_save"):
            store = self.store
            session.on_save = lambda s: store.save(s.snapshot())
        return session

    async def restore(self, session_id: str) -> AgentSession | None:
        """The live session, or a sleeping one woken from its record (see the
        module doc); None if there is neither. Raises SessionLimitReached when
        waking it would exceed the ceiling."""
        live = self._sessions.get(session_id)
        if live is not None or self.store is None:
            return live
        async with self._restoring:  # two requests for one id wake it once
            live = self._sessions.get(session_id)
            if live is not None:
                return live
            rec = self.store.load(session_id)
            if rec is None:
                return None
            try:
                ws = (
                    self._workspace_ok(rec.get("workspace", ""))
                    if self._workspace_ok is not None
                    else Path(rec["workspace"])
                )
            except (ValueError, KeyError, TypeError):
                return None  # its workspace is no longer allowed: it stays asleep
            history = [
                (int(item[0]), item[1])
                for item in rec.get("events") or []
                if isinstance(item, list) and len(item) == 2 and isinstance(item[1], dict)
            ]
            session = self._factory(
                ws,
                session_id=session_id,
                resume=rec.get("sdk_session_id") or None,
                history=history,
                created_at=rec.get("created_at"),
            )
            return await self.adopt(session)

    def all(self) -> list[Any]:
        return list(self._sessions.values())

    def get(self, session_id: str) -> AgentSession | None:
        return self._sessions.get(session_id)

    def touch(self, session: AgentSession | str | None) -> None:
        """Defer idle reaping for a session (accepts the object or its id)."""
        if isinstance(session, str):
            session = self._sessions.get(session)
        if session is not None and hasattr(session, "touch"):
            session.touch()

    async def close(self, session_id: str, *, forget: bool = False) -> bool:
        """End a live session: it goes to sleep (its record stays, so it can be
        restored), or with ``forget`` is discarded for good ("new session")."""
        session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.aclose()
        dropped = forget and self.store is not None and self.store.delete(session_id)
        return session is not None or bool(dropped)

    def _idle_ids(self, now: float) -> list[str]:
        """Sessions past the idle timeout with no live event stream attached.

        An open SSE connection counts as activity (the stream heartbeat touches
        the session), so a watching browser never has its agent reaped mid-run.
        Nor is a running crew: closing its host can't stop the worker thread,
        only orphan the run from its stream and approvals.
        """
        if not self.idle_timeout:
            return []
        stale = []
        for sid, session in self._sessions.items():
            if getattr(session, "subscriber_count", 0):
                continue
            if getattr(session, "kind", "") == "crew" and getattr(session, "busy", False):
                continue
            last = getattr(session, "last_active", now)
            if now - last >= self.idle_timeout:
                stale.append(sid)
        return stale

    async def reap_idle(self) -> list[str]:
        """Put sessions idle beyond the timeout to sleep (their subprocess
        goes; a persisted session can be restored). Returns the ids closed."""
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
