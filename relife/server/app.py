"""FastAPI app + ``serve()`` for the always-on agent server.

Mirrors the memory daemon (``relife/memory/remote/daemon.py``): a side-effect-free
:func:`create_app` factory (so tests can drive it via ``httpx.ASGITransport`` /
``TestClient`` with an injected fake session factory — no model calls), plus a
``serve()`` that lazily imports uvicorn.

Security posture (every policy decision is a pure function in ``security.py``):
  * auth accepts a bearer header **or** a cookie minted by ``POST /auth`` — the
    browser can't set headers on an ``EventSource``, so without the cookie the
    SSE stream (which carries the whole transcript, approval prompts included)
    would be the one route the UI could never authenticate;
  * mutating routes additionally require a same-origin ``Origin`` (CSRF), on top
    of the ``SameSite=Strict`` cookie;
  * a session's workspace is confined to ``AGENT_WORKSPACE_ROOT`` — the
    permission policy auto-allows writes inside a session's workspace, so an
    unconstrained path in the request body would let the caller choose the
    auto-allow blast radius;
  * ``serve()`` refuses a non-loopback bind without a token (fail closed);
  * sessions, queued turns, event streams, and message size are all bounded.

Endpoints:
  GET    /                                    → the self-contained web UI
  GET    /health                              → status (no auth)
  GET    /auth/status                         → whether auth is on / satisfied
  POST   /auth                                → exchange a token for a cookie
  POST   /auth/logout                         → clear the cookie
  POST   /sessions                            → create a session → {session_id}
  GET    /sessions/{id}                       → is this session still live?
  DELETE /sessions/{id}                       → close a session
  POST   /sessions/{id}/messages              → submit a turn
  GET    /sessions/{id}/events                → SSE stream of agent events
  POST   /sessions/{id}/approvals/{approval}  → resolve an approval (allow|deny)
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from .. import config
from .security import (
    AttemptLimiter,
    guard_bind,
    presented_token,
    resolve_workspace,
    same_origin,
    token_matches,
)
from .session import (
    SessionFactory,
    SessionLimitReached,
    SessionManager,
    TooManySubscribers,
    TurnQueueFull,
)

# Seconds between SSE keepalive comments when no events flow (keeps proxies and
# some clients from dropping an idle connection).
_SSE_HEARTBEAT = 15.0


def _load_ui() -> str:
    path = config.WEB_DIR / "index.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "<!doctype html><title>ReLife</title><p>UI not found.</p>"


def create_app(
    *,
    token: str | None = None,
    session_factory: SessionFactory | None = None,
    workspace_root: Path | None = None,
    reap: bool = True,
) -> FastAPI:
    """Build the agent server app.

    ``session_factory`` builds an :class:`AgentSession` for a workspace; the
    default creates real (model-backed) sessions. Tests inject a fake factory
    that emits scripted events so the whole HTTP + SSE + approval flow runs with
    no model calls. Nothing here touches the network or spawns a task at build
    time — the idle reaper starts under the app lifespan.
    """
    manager = SessionManager(session_factory=session_factory)
    root = Path(workspace_root) if workspace_root is not None else config.AGENT_WORKSPACE_ROOT
    ui_html = _load_ui()
    auth_limiter = AttemptLimiter(config.AGENT_AUTH_MAX_ATTEMPTS, config.AGENT_AUTH_WINDOW)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        reaper = asyncio.create_task(manager.run_reaper()) if reap else None
        try:
            yield
        finally:
            if reaper is not None:
                reaper.cancel()
                try:
                    await reaper
                except (asyncio.CancelledError, Exception):
                    pass
            # Every session owns a ClaudeSDKClient subprocess; shutting the
            # server down must not orphan them.
            await manager.aclose()

    app = FastAPI(title="ReLife agent server", version="0.1.0", lifespan=lifespan)
    app.state.manager = manager
    app.state.workspace_root = root

    def _authenticated(request: Request) -> bool:
        return token_matches(
            token,
            presented_token(
                request.headers.get("authorization"), request.cookies.get(config.AGENT_COOKIE)
            ),
        )

    def require_token(request: Request) -> None:
        if not _authenticated(request):
            raise HTTPException(status_code=401, detail="invalid or missing token")

    def require_same_origin(request: Request) -> None:
        """CSRF guard for state-changing routes (cookie auth is ambient)."""
        if not same_origin(request.headers.get("origin"), request.headers.get("host")):
            raise HTTPException(status_code=403, detail="cross-origin request refused")

    auth = [Depends(require_token)]
    mutate = [Depends(require_same_origin), Depends(require_token)]

    def _session_or_404(session_id: str):
        session = manager.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        manager.touch(session)
        return session

    # --- UI + status --------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return ui_html

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "sessions": manager.count(), "auth_required": token is not None}

    # --- auth ---------------------------------------------------------------
    @app.get("/auth/status")
    async def auth_status(request: Request) -> dict[str, Any]:
        return {"auth_required": token is not None, "authenticated": _authenticated(request)}

    @app.post("/auth", dependencies=[Depends(require_same_origin)])
    async def auth_login(
        request: Request, response: Response, body: dict[str, Any]
    ) -> dict[str, Any]:
        if token is None:
            return {"ok": True, "auth_required": False}
        client = request.client.host if request.client else "unknown"
        if not auth_limiter.allow(client):
            raise HTTPException(status_code=429, detail="too many attempts; wait a minute")
        if not token_matches(token, (body or {}).get("token")):
            raise HTTPException(status_code=401, detail="invalid token")
        auth_limiter.reset(client)
        response.set_cookie(
            config.AGENT_COOKIE,
            token,
            max_age=config.AGENT_COOKIE_MAX_AGE,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            path="/",
        )
        return {"ok": True, "auth_required": True}

    @app.post("/auth/logout", dependencies=[Depends(require_same_origin)])
    async def auth_logout(response: Response) -> dict[str, Any]:
        response.delete_cookie(config.AGENT_COOKIE, path="/")
        return {"ok": True}

    # --- sessions -----------------------------------------------------------
    @app.post("/sessions", dependencies=mutate)
    async def create_session(body: dict[str, Any] | None = None) -> dict[str, Any]:
        body = body or {}
        try:
            ws = resolve_workspace(body.get("workspace"), root)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        ws.mkdir(parents=True, exist_ok=True)
        try:
            session = await manager.create(ws)
        except SessionLimitReached as e:
            raise HTTPException(status_code=429, detail=str(e)) from e
        return {"session_id": session.id, "workspace": str(ws)}

    @app.get("/sessions/{session_id}", dependencies=auth)
    async def get_session(session_id: str) -> dict[str, Any]:
        """Does this session still exist? Lets a reloading UI reattach to its
        own agent instead of spawning a second one and orphaning the first."""
        session = _session_or_404(session_id)
        return {
            "session_id": session.id,
            "workspace": str(getattr(session, "workspace", "")),
        }

    @app.delete("/sessions/{session_id}", dependencies=mutate)
    async def close_session(session_id: str) -> dict[str, Any]:
        if not await manager.close(session_id):
            raise HTTPException(status_code=404, detail="unknown session")
        return {"closed": True}

    @app.post("/sessions/{session_id}/messages", dependencies=mutate)
    async def post_message(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        session = _session_or_404(session_id)
        text = (body.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty message")
        if len(text) > config.AGENT_MAX_MESSAGE_CHARS:
            raise HTTPException(
                status_code=413,
                detail=f"message exceeds {config.AGENT_MAX_MESSAGE_CHARS} characters",
            )
        try:
            await session.submit(text)
        except TurnQueueFull as e:
            raise HTTPException(status_code=429, detail=str(e)) from e
        return {"ok": True}

    @app.post("/sessions/{session_id}/approvals/{approval_id}", dependencies=mutate)
    async def resolve_approval(
        session_id: str, approval_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        session = _session_or_404(session_id)
        decision = (body.get("decision") or "").lower()
        if decision not in {"allow", "deny"}:
            raise HTTPException(status_code=400, detail="decision must be allow|deny")
        resolved = session.resolve_approval(approval_id, decision == "allow")
        return {"resolved": resolved}

    @app.get("/sessions/{session_id}/events", dependencies=auth)
    async def events(session_id: str, request: Request) -> StreamingResponse:
        session = _session_or_404(session_id)
        last_raw = request.headers.get("last-event-id") or request.query_params.get("last_id")
        last_id = int(last_raw) if last_raw and last_raw.isdigit() else None
        try:
            queue, backlog = session.subscribe(last_id)
        except TooManySubscribers as e:
            raise HTTPException(status_code=429, detail=str(e)) from e

        async def gen():
            try:
                for sid, ev in backlog:
                    yield _sse(sid, ev)
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        sid, ev = await asyncio.wait_for(queue.get(), _SSE_HEARTBEAT)
                    except asyncio.TimeoutError:
                        # A watching browser is activity: keep the session off
                        # the idle reaper's list for as long as it is streaming.
                        manager.touch(session)
                        yield ": ping\n\n"
                        continue
                    yield _sse(sid, ev)
            finally:
                session.unsubscribe(queue)
                manager.touch(session)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _sse(seq: int, ev: dict[str, Any]) -> str:
    return f"id: {seq}\ndata: {json.dumps(ev)}\n\n"


def serve(
    host: str | None = None,
    port: int | None = None,
    *,
    token: str | None = None,
) -> None:
    """Run the agent server (blocking).

    Refuses to bind a non-loopback address without a token: this process can run
    shell commands and edit files, so an unauthenticated reachable bind is a
    mistake to fail on, not to warn about.
    """
    import uvicorn

    h = host or config.AGENT_HOST
    p = port or config.AGENT_PORT
    tok = token if token is not None else config.AGENT_TOKEN
    guard_bind(h, tok)
    app = create_app(token=tok)
    uvicorn.run(app, host=h, port=p)
