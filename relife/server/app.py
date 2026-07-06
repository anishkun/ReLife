"""FastAPI app + ``serve()`` for the always-on agent server.

Mirrors the memory daemon (``relife/memory/remote/daemon.py``): a side-effect-free
:func:`create_app` factory (so tests can drive it via ``httpx.ASGITransport`` /
``TestClient`` with an injected fake session factory — no model calls), plus a
``serve()`` that lazily imports uvicorn.

Endpoints:
  GET  /                                    → the self-contained web UI
  GET  /health                              → status (no auth)
  POST /sessions                            → create a session → {session_id}
  POST /sessions/{id}/messages              → submit a turn
  GET  /sessions/{id}/events                → SSE stream of agent events
  POST /sessions/{id}/approvals/{approval}  → resolve an approval (allow|deny)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from .. import config
from .session import SessionFactory, SessionManager

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
) -> FastAPI:
    """Build the agent server app.

    ``session_factory`` builds an :class:`AgentSession` for a workspace; the
    default creates real (model-backed) sessions. Tests inject a fake factory
    that emits scripted events so the whole HTTP + SSE + approval flow runs with
    no model calls.
    """
    manager = SessionManager(session_factory=session_factory)
    ui_html = _load_ui()
    app = FastAPI(title="ReLife agent server", version="0.1.0")
    app.state.manager = manager

    def require_token(authorization: str | None = Header(default=None)) -> None:
        if token is None:
            return
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="invalid or missing token")

    auth = [Depends(require_token)]

    def _session_or_404(session_id: str):
        session = manager.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return session

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return ui_html

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "sessions": len(manager._sessions)}

    @app.post("/sessions", dependencies=auth)
    async def create_session(body: dict[str, Any] | None = None) -> dict[str, Any]:
        body = body or {}
        ws = Path(body["workspace"]).resolve() if body.get("workspace") else config.DEFAULT_WORKSPACE
        ws.mkdir(parents=True, exist_ok=True)
        session = await manager.create(ws)
        return {"session_id": session.id, "workspace": str(ws)}

    @app.post("/sessions/{session_id}/messages", dependencies=auth)
    async def post_message(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        session = _session_or_404(session_id)
        text = (body.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty message")
        await session.submit(text)
        return {"ok": True}

    @app.post("/sessions/{session_id}/approvals/{approval_id}", dependencies=auth)
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
        queue, backlog = session.subscribe(last_id)

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
                        yield ": ping\n\n"
                        continue
                    yield _sse(sid, ev)
            finally:
                session.unsubscribe(queue)

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
    """Run the agent server (blocking)."""
    import uvicorn

    h = host or config.AGENT_HOST
    p = port or config.AGENT_PORT
    app = create_app(token=token if token is not None else config.AGENT_TOKEN)
    uvicorn.run(app, host=h, port=p)
