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
  * sessions, queued turns, event streams, message size — and schedules — are
    all bounded.

Schedules (autonomous triggers) live in ``schedules.py``/``scheduler.py``: a
background task under the lifespan fires each due schedule as a turn in its own
session, so an unattended run goes through the very same approval path (and
times out to deny when no one is watching).

Agents, memory spaces and crews (the platform pass) are here too: the agent
registry and its handoff operations (``relife/agents.py``) over HTTP, and
crews planned and run under the server (``crews.py``) — a running crew is
hosted like a session, so it streams on ``/sessions/{id}/events`` and its
members' approvals resolve on the same approval route.

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
  GET    /schedules                           → list schedules
  POST   /schedules                           → create one → the record
  GET    /schedules/{id}                      → one schedule
  PATCH  /schedules/{id}                      → edit / enable / disable
  DELETE /schedules/{id}                      → remove
  POST   /schedules/{id}/run                  → fire it now
  GET    /schedules/{id}/runs                 → recorded outcomes, newest first
  GET    /schedules/{id}/runs/{run_id}        → one outcome, with its event stream
  GET    /spaces                              → memory spaces and their counts
  GET    /agents                              → registered agents
  POST   /agents                              → register one (inherit / fork)
  GET    /agents/{name}                       → one agent
  DELETE /agents/{name}                       → unregister (archives its memory)
  POST   /agents/{name}/attach|detach         → read another agent's memory, or stop
  POST   /agents/{name}/promote               → copy its memories into yours (default)
  GET    /crews                               → recent crew runs
  POST   /crews                               → plan a task (one model call) or record a spec
  GET    /crews/{id}                          → one crew run, with its plan
  POST   /crews/{id}/run                      → run a planned crew → {session_id}
  POST   /crews/{id}/stop                     → stop a running crew
  DELETE /crews/{id}                          → discard a planned crew
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from .. import agents as agents_mod
from .. import config
from ..crew.record import CrewRunStore
from ..crew.spec import CrewSpec
from ..memory.client import off_loop
from .crews import CREWAI_HINT, CrewLimitReached, CrewService, CrewUnavailable, crewai_available
from .runs import RunStore
from ..permissions import normalize_grants
from .scheduler import Scheduler
from .schedules import (
    Schedule,
    ScheduleStore,
    check_grants_fit,
    next_run,
    normalize_crew,
    normalize_spec,
    normalize_work,
    validate_name,
    validate_task,
)
from .security import (
    AttemptLimiter,
    guard_bind,
    host_allowed,
    presented_token,
    resolve_workspace,
    same_origin,
    token_matches,
)
from .session import (
    SessionFactory,
    SessionLimitReached,
    SessionManager,
    SessionReadOnly,
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
    schedules_path: Path | None = None,
    runs_dir: Path | None = None,
    run_scheduler: bool | None = None,
    allowed_hosts: frozenset[str] | None = None,
    agents_path: Path | None = None,
    crews_dir: Path | None = None,
    memory_client: Any = None,
    crew_ask_model: Any = None,
    crew_turn: Any = None,
    crew_llm_factory: Any = None,
    crew_approval_timeout: float | None = None,
) -> FastAPI:
    """Build the agent server app.

    ``session_factory`` builds an :class:`AgentSession` for a workspace; the
    default creates real (model-backed) sessions. Tests inject a fake factory
    that emits scripted events so the whole HTTP + SSE + approval flow runs with
    no model calls. Nothing here touches the network or spawns a task at build
    time — the idle reaper and the scheduler start under the app lifespan.
    ``schedules_path`` is where schedules persist and ``runs_dir`` where run
    outcomes go (tests point both at tmp); ``run_scheduler=False`` keeps the
    tick loop off so tests drive ``app.state.scheduler.tick(now)`` by hand.
    ``allowed_hosts`` are non-loopback ``Host`` names a tokenless server may
    answer to (default ``RELIFE_AGENT_ALLOWED_HOSTS``) — see ``host_allowed``.
    ``agents_path``/``crews_dir``/``memory_client`` and the ``crew_*`` hooks
    point the platform routes at tmp state and stub models in tests.
    """
    manager = SessionManager(session_factory=session_factory)
    root = Path(workspace_root) if workspace_root is not None else config.AGENT_WORKSPACE_ROOT
    ui_html = _load_ui()
    auth_limiter = AttemptLimiter(config.AGENT_AUTH_MAX_ATTEMPTS, config.AGENT_AUTH_WINDOW)
    store = ScheduleStore(schedules_path)
    runs = RunStore(runs_dir)
    crews = CrewService(
        manager,
        workspace_root=root,
        records=CrewRunStore(crews_dir),
        agents_path=agents_path,
        client=memory_client,
        ask_model=crew_ask_model,
        turn=crew_turn,
        llm_factory=crew_llm_factory,
        approval_timeout=crew_approval_timeout,
    )
    scheduler = Scheduler(store, manager, workspace_root=root, runs=runs, crews=crews)
    tick = config.AGENT_SCHEDULER if run_scheduler is None else run_scheduler

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks = []
        if reap:
            tasks.append(asyncio.create_task(manager.run_reaper()))
        if tick:
            tasks.append(asyncio.create_task(scheduler.run()))
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            # In-flight run recorders write an `interrupted` outcome.
            await scheduler.aclose()
            # Every session owns a ClaudeSDKClient subprocess; shutting the
            # server down must not orphan them.
            await manager.aclose()

    extra_hosts = frozenset(
        h.lower() for h in (config.AGENT_ALLOWED_HOSTS if allowed_hosts is None else allowed_hosts)
    )

    def require_known_host(request: Request) -> None:
        """DNS-rebinding guard, on every route (the UI and SSE included)."""
        if not host_allowed(request.headers.get("host"), token, extra_hosts):
            raise HTTPException(status_code=403, detail="unrecognized Host header")

    app = FastAPI(
        title="ReLife agent server",
        version="1.0.0",
        lifespan=lifespan,
        dependencies=[Depends(require_known_host)],
    )
    app.state.manager = manager
    app.state.workspace_root = root
    app.state.schedules = store
    app.state.scheduler = scheduler
    app.state.runs = runs
    app.state.crews = crews

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
        return {
            "status": "ok",
            "sessions": manager.count(),
            "schedules": store.count(),
            "auth_required": token is not None,
        }

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
            "kind": getattr(session, "kind", "chat"),
            "busy": bool(getattr(session, "busy", False)),
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
        except SessionReadOnly as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
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

    # --- schedules ----------------------------------------------------------
    def _schedule_or_404(schedule_id: str) -> Schedule:
        schedule = store.get(schedule_id)
        if schedule is None:
            raise HTTPException(status_code=404, detail="unknown schedule")
        return schedule

    def _spec_from(body: dict[str, Any]) -> dict[str, Any] | None:
        """Accept the spec nested (``{"spec": {...}}``) or flat (``every``/``at``/``days``)."""
        if isinstance(body.get("spec"), dict):
            return body["spec"]
        flat = {k: body[k] for k in ("every", "at", "days") if k in body}
        return flat or None

    def _check_workspace(raw: Any) -> str:
        # Same containment as POST /sessions: a schedule's workspace is where the
        # permission policy auto-allows writes, so it must live under the root.
        try:
            resolve_workspace(raw, root)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return str(raw or "").strip()

    @app.get("/schedules", dependencies=auth)
    async def list_schedules() -> dict[str, Any]:
        return {"schedules": [s.to_dict() for s in store.list()]}

    @app.post("/schedules", dependencies=mutate, status_code=201)
    async def create_schedule(body: dict[str, Any]) -> dict[str, Any]:
        if config.AGENT_MAX_SCHEDULES and store.count() >= config.AGENT_MAX_SCHEDULES:
            raise HTTPException(
                status_code=429,
                detail=f"at most {config.AGENT_MAX_SCHEDULES} schedules "
                "(remove one, or raise RELIFE_AGENT_MAX_SCHEDULES)",
            )
        ws = _check_workspace(body.get("workspace"))
        try:
            schedule = Schedule.new(
                name=body.get("name"),
                task=body.get("task"),
                spec=_spec_from(body),
                workspace=ws,
                enabled=body.get("enabled", True),
                grants=body.get("grants"),
                work=body.get("work"),
                crew=body.get("crew"),
                agents=_agent_names(),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        store.add(schedule)
        return schedule.to_dict()

    @app.get("/schedules/{schedule_id}", dependencies=auth)
    async def get_schedule(schedule_id: str) -> dict[str, Any]:
        return _schedule_or_404(schedule_id).to_dict()

    @app.patch("/schedules/{schedule_id}", dependencies=mutate)
    async def update_schedule(schedule_id: str, body: dict[str, Any]) -> dict[str, Any]:
        schedule = _schedule_or_404(schedule_id)
        try:
            if "name" in body:
                schedule.name = validate_name(body["name"])
            # Validate work + task together before assigning either: turning
            # work off must leave a task behind, never a blank schedule.
            work = normalize_work(body["work"]) if "work" in body else schedule.work
            crew = (
                normalize_crew(body["crew"], agents=_agent_names()) if "crew" in body else schedule.crew
            )
            task = validate_task(
                body.get("task", schedule.task), required=work is None and crew is None
            )
            grants = normalize_grants(body["grants"]) if "grants" in body else schedule.grants
            check_grants_fit(grants, work, crew)
            schedule.work, schedule.crew, schedule.task, schedule.grants = work, crew, task, grants
            if "workspace" in body:
                schedule.workspace = _check_workspace(body["workspace"])
            spec = _spec_from(body)
            if spec is not None:
                schedule.spec = normalize_spec(spec)
                schedule.next_run_at = next_run(schedule.spec, time.time())
            if "enabled" in body:
                schedule.enabled = bool(body["enabled"])
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        store.save()
        return schedule.to_dict()

    @app.delete("/schedules/{schedule_id}", dependencies=mutate)
    async def delete_schedule(schedule_id: str) -> dict[str, Any]:
        if not store.remove(schedule_id):
            raise HTTPException(status_code=404, detail="unknown schedule")
        runs.remove_all(schedule_id)
        return {"removed": True}

    @app.post("/schedules/{schedule_id}/run", dependencies=mutate)
    async def run_schedule(schedule_id: str) -> dict[str, Any]:
        schedule = _schedule_or_404(schedule_id)
        status = await scheduler.fire(schedule)
        return {"status": status, "session_id": schedule.session_id, "schedule": schedule.to_dict()}

    @app.get("/schedules/{schedule_id}/runs", dependencies=auth)
    async def list_runs(schedule_id: str, limit: int = 20) -> dict[str, Any]:
        _schedule_or_404(schedule_id)
        limit = max(1, min(limit, config.AGENT_RUN_HISTORY or limit))
        return {"runs": [r.to_dict() for r in runs.list(schedule_id, limit)]}

    @app.get("/schedules/{schedule_id}/runs/{run_id}", dependencies=auth)
    async def get_run(schedule_id: str, run_id: str) -> dict[str, Any]:
        _schedule_or_404(schedule_id)
        rec = runs.get(schedule_id, run_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="unknown run")
        return rec.to_dict(with_events=True)

    # --- agents + memory spaces ---------------------------------------------
    # Same operations as `relife agent …`: the registry is re-read per request
    # (the CLI may change it), and every memory touch runs off the loop. Token
    # minting stays CLI-only — this server's token holder is already the user,
    # but a credential for the memory daemon is better printed once to a
    # terminal than carried through a browser.
    def _agent_store() -> agents_mod.AgentStore:
        return agents_mod.AgentStore(agents_path)

    def _agent_names() -> set[str]:
        return {a.name for a in _agent_store().list()}

    def _agent_view(a: agents_mod.AgentProfile, counts: dict[str, dict[str, int]]) -> dict[str, Any]:
        d = a.public()
        d["memories"] = counts.get(a.own_space, {}).get("memories", 0)
        d["reads"] = list(a.scope().read)
        return d

    def _bad(e: Exception) -> HTTPException:
        if isinstance(e, LookupError):
            return HTTPException(status_code=404, detail=str(e).strip("'\""))
        return HTTPException(status_code=400, detail=str(e))

    @app.get("/spaces", dependencies=auth)
    async def list_spaces() -> dict[str, Any]:
        return {"spaces": await off_loop(crews.client.spaces)}

    @app.get("/agents", dependencies=auth)
    async def list_agents() -> dict[str, Any]:
        st = _agent_store()
        counts = await off_loop(crews.client.spaces)
        return {"agents": [_agent_view(a, counts) for a in st.list()], "problem": st.problem}

    @app.post("/agents", dependencies=mutate, status_code=201)
    async def create_agent_route(body: dict[str, Any]) -> dict[str, Any]:
        inherit = body.get("inherit") or []
        if not isinstance(inherit, list) or not all(isinstance(x, str) for x in inherit):
            raise HTTPException(status_code=400, detail="inherit must be a list of agent names")

        def make() -> tuple[agents_mod.AgentProfile, Any]:
            return agents_mod.create_agent(
                _agent_store(), crews.client, str(body.get("name") or ""),
                runtime=str(body.get("runtime") or "relife"),
                model=str(body.get("model") or ""),
                description=str(body.get("description") or ""),
                inherit=inherit,
                fork=body.get("fork") or None,
                isolated=bool(body.get("isolated", False)),
            )

        try:
            profile, copied = await off_loop(make)
        except (ValueError, LookupError, TypeError) as e:
            raise _bad(e) from e
        out = _agent_view(profile, await off_loop(crews.client.spaces))
        out["copied"] = copied
        return out

    @app.get("/agents/{name}", dependencies=auth)
    async def get_agent(name: str) -> dict[str, Any]:
        try:
            a = _agent_store().require(name)
        except LookupError as e:
            raise _bad(e) from e
        return _agent_view(a, await off_loop(crews.client.spaces))

    @app.delete("/agents/{name}", dependencies=mutate)
    async def delete_agent_route(name: str, keep_memory: bool = False) -> dict[str, Any]:
        try:
            archived = await off_loop(
                agents_mod.delete_agent, _agent_store(), crews.client, name, keep_memory=keep_memory
            )
        except LookupError as e:
            raise _bad(e) from e
        return {"removed": True, "archived": archived}

    @app.post("/agents/{name}/attach", dependencies=mutate)
    async def attach_route(name: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            a = agents_mod.attach(_agent_store(), name, str(body.get("other") or ""))
        except (ValueError, LookupError) as e:
            raise _bad(e) from e
        return _agent_view(a, await off_loop(crews.client.spaces))

    @app.post("/agents/{name}/detach", dependencies=mutate)
    async def detach_route(name: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            a = agents_mod.detach(_agent_store(), name, str(body.get("other") or ""))
        except (ValueError, LookupError) as e:
            raise _bad(e) from e
        return _agent_view(a, await off_loop(crews.client.spaces))

    @app.post("/agents/{name}/promote", dependencies=mutate)
    async def promote_route(name: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        ids = (body or {}).get("ids")
        if ids is not None and (
            not isinstance(ids, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids)
        ):
            raise HTTPException(status_code=400, detail="ids must be a list of memory ids")
        try:
            copied = await off_loop(agents_mod.promote, _agent_store(), crews.client, name, ids=ids)
        except (ValueError, LookupError) as e:
            raise _bad(e) from e
        return {"copied": copied}

    # --- crews ------------------------------------------------------------------
    def _crew_view(rec: Any, *, full: bool = False) -> dict[str, Any]:
        from ..crew.runner import describe_plan

        d = rec.to_dict()
        try:
            d["plan"] = describe_plan(CrewSpec.from_dict(rec.spec))
        except (KeyError, TypeError):
            d["plan"] = []
        d["session_id"] = crews.session_id(rec.id)
        host = crews.host(rec.id)
        if host is not None and host.busy and d["status"] == "planned":
            # Started, but the worker thread hasn't saved "running" yet: the
            # host is the truth (otherwise the UI offers "run" on a live crew).
            d["status"] = "running"
        if not full:
            for t in d["tasks"]:
                t["output"] = t["output"][:600]
            d["final_output"] = d["final_output"][:1200]
        return d

    def _crew_or_404(crew_id: str) -> Any:
        rec = crews.records.get(crew_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="unknown crew")
        return rec

    @app.get("/crews", dependencies=auth)
    async def list_crews(limit: int = 20) -> dict[str, Any]:
        recs = await off_loop(crews.records.list, max(1, min(limit, 100)))
        return {
            "crews": [_crew_view(r) for r in recs],
            "crewai": crewai_available(),
            "hint": None if crewai_available() else CREWAI_HINT,
            "llms": [config.CREW_CLAUDE_LLM, *config.CREW_LLMS],
        }

    @app.post("/crews", dependencies=mutate, status_code=201)
    async def plan_crew_route(body: dict[str, Any]) -> dict[str, Any]:
        """Plan a task (spends one tool-less model call) or record a given spec
        (no model call). Nothing runs until ``POST /crews/{id}/run``."""
        task = body.get("task")
        spec = body.get("spec")
        if spec is None:
            task = str(task or "").strip()
            if not task:
                raise HTTPException(status_code=400, detail="give a task or a spec")
            if len(task) > config.AGENT_MAX_MESSAGE_CHARS:
                raise HTTPException(status_code=413, detail="task is too long")
        try:
            rec, _spec = await crews.plan(task=task or None, spec_raw=spec)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return _crew_view(rec, full=True)

    @app.get("/crews/{crew_id}", dependencies=auth)
    async def get_crew(crew_id: str) -> dict[str, Any]:
        return _crew_view(_crew_or_404(crew_id), full=True)

    @app.post("/crews/{crew_id}/run", dependencies=mutate)
    async def run_crew_route(crew_id: str) -> dict[str, Any]:
        _crew_or_404(crew_id)
        try:
            host = await crews.start(crew_id)
        except CrewUnavailable as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except (CrewLimitReached, SessionLimitReached) as e:
            raise HTTPException(status_code=429, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return {"session_id": host.id, "crew_id": crew_id}

    @app.post("/crews/{crew_id}/stop", dependencies=mutate)
    async def stop_crew_route(crew_id: str) -> dict[str, Any]:
        _crew_or_404(crew_id)
        host = crews.host(crew_id)
        if host is None or not host.busy:
            raise HTTPException(status_code=409, detail="that crew isn't running here")
        denied = host.stop()
        return {"stopping": True, "denied": denied}

    @app.delete("/crews/{crew_id}", dependencies=mutate)
    async def discard_crew_route(crew_id: str) -> dict[str, Any]:
        rec = _crew_or_404(crew_id)
        if crews.host(crew_id) is not None or crews.starting(crew_id):
            # Started, though the record may still say "planned" for an
            # instant: discarding it would mark a live crew cancelled.
            raise HTTPException(status_code=409, detail="that crew has been started; stop it instead")
        if rec.status != "planned":
            raise HTTPException(status_code=409, detail=f"crew is {rec.status}; only a plan can be discarded")
        rec.status = "cancelled"
        rec.finished_at = time.time()
        crews.records.save(rec)
        return {"discarded": True}

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
