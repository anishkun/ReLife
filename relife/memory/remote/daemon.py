"""The memory daemon — a FastAPI service core over the same ``MemoryService`` the
in-process path uses.

Design notes (see ``.claude/plans`` for the full rationale):

- **DB binding.** ``save``/``recall``/… honour an injected store, but
  ``consolidate()`` / ``dream()`` mine the module-level store (``store._DB_PATH``)
  and the event log (``events._DB_PATH``). So the daemon does **not** inject a
  store: it points both ``_DB_PATH`` globals at its configured DB and uses the
  default ``MemoryService()``. Then every operation — including upkeep — targets
  the same database.
- **Concurrency.** Endpoints are ``async def`` calling the *sync* service
  directly, so all DB + embedding access is serialized on the event loop: no
  threadpool writers, no SQLite lock contention. Fine for a single-user daemon;
  per-request offloading is a later refinement.
- **Eager schema init** (not a lifespan) so ``httpx.ASGITransport`` /
  ``TestClient`` work without startup events.

``fastapi`` is an optional (``[daemon]``) dependency; importing this module
requires it. ``wire`` stays dependency-free and is imported by both sides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException

from .. import events as _events
from .. import skills as _skills
from .. import store as _store_mod
from .. import workflows as _workflows
from ..service import MemoryService
from . import wire


def _bind_db(db_path: Path | str) -> None:
    """Point the module-level store + event log at ``db_path`` and init schema.

    Both globals must move together: consolidate/dream read the event log via
    ``events._DB_PATH`` while recall/save use ``store._DB_PATH``.
    """
    p = Path(db_path)
    _store_mod._DB_PATH = p
    _events._DB_PATH = p
    # Eager init + WAL so ASGI/TestClient work with no lifespan and a stray
    # external reader (e.g. `relife memory stats` without the env var) is safe.
    store = _store_mod._store()
    store.init_db()
    with store._connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")


def _bind_dirs(
    skills_dir: Path | str | None, workflows_dir: Path | str | None
) -> None:
    """Point the module-level skills/workflows stores at the daemon's dirs.

    Mirrors ``_bind_db`` for procedural memory: ``consolidate()`` writes
    workflows via the ``workflows`` module functions (never the client — that
    would deadlock the daemon's event loop), so those functions must resolve to
    the daemon's dirs, not the client's. ``None`` leaves the module default
    (``config.SKILLS_DIR`` / ``config.WORKFLOWS_DIR``) untouched — deliberate, so
    tests that build an app against a tmp DB without dirs don't clobber the real
    dirs for the rest of the process.
    """
    if skills_dir is not None:
        _skills._SKILLS_DIR = Path(skills_dir)
    if workflows_dir is not None:
        _workflows._WORKFLOWS_DIR = Path(workflows_dir)


def create_app(
    db_path: Path | str,
    token: str | None = None,
    *,
    skills_dir: Path | str | None = None,
    workflows_dir: Path | str | None = None,
) -> FastAPI:
    """Build the memory daemon app bound to ``db_path``.

    Constructing the app does not touch the sidecar file or the network — that
    is ``serve()``'s job — so tests can build the app against a ``tmp_path`` DB
    and drive it via ``httpx.ASGITransport`` with no side effects.
    """
    _bind_db(db_path)
    _bind_dirs(skills_dir, workflows_dir)
    svc = MemoryService()  # default: follows the _DB_PATH we just bound
    app = FastAPI(title="ReLife memory daemon", version="0.1.0")

    def require_token(authorization: str | None = Header(default=None)) -> None:
        if token is None:
            return
        expected = f"Bearer {token}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="invalid or missing token")

    auth = [Depends(require_token)]

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "count": svc.count(include_archived=True),
            "skills": svc.skill_count(),
            "workflows": svc.workflow_count(),
            "events": svc.event_count(),
        }

    @app.post("/save", dependencies=auth)
    async def save(body: dict[str, Any]) -> dict[str, Any]:
        mid = svc.save(
            body["text"],
            kind=body.get("kind", "fact"),
            tags=body.get("tags", ""),
            importance=body.get("importance"),
        )
        return {"id": mid}

    @app.post("/recall", dependencies=auth)
    async def recall(body: dict[str, Any]) -> dict[str, Any]:
        hits = svc.recall(
            body["query"],
            k=int(body.get("k", 5)),
            reinforce=bool(body.get("reinforce", False)),
            include_archived=bool(body.get("include_archived", False)),
        )
        return {"memories": wire.memories_to_list(hits)}

    @app.post("/forget", dependencies=auth)
    async def forget(body: dict[str, Any]) -> dict[str, Any]:
        gone = svc.forget(body["query"])
        return {"memory": wire.memory_or_none_to_dict(gone)}

    @app.get("/memories", dependencies=auth)
    async def memories(include_archived: bool = True) -> dict[str, Any]:
        ms = svc.all_memories(include_archived=include_archived)
        return {"memories": wire.memories_to_list(ms)}

    @app.get("/count", dependencies=auth)
    async def count(include_archived: bool = True) -> dict[str, Any]:
        return {"count": svc.count(include_archived=include_archived)}

    @app.post("/consolidate", dependencies=auth)
    async def consolidate() -> dict[str, Any]:
        return wire.consolidation_to_dict(svc.consolidate())

    @app.post("/consolidate/maybe", dependencies=auth)
    async def consolidate_maybe() -> dict[str, Any] | None:
        # The throttle decision runs here, server-side, against the daemon's own
        # event log + watermark — never split to a client-side gate.
        report = svc.maybe_consolidate()
        return None if report is None else wire.consolidation_to_dict(report)

    @app.post("/dream", dependencies=auth)
    async def dream() -> dict[str, Any]:
        # Runs the REM pass server-side using the daemon's default ask_model
        # (needs a logged-in `claude` CLI / Max budget). The client cannot send
        # its ask_model callable over the wire, so the daemon owns it.
        report = await svc.dream()
        return wire.rem_to_dict(report)

    # --- procedural memory: skills ------------------------------------------
    @app.post("/skills/write", dependencies=auth)
    async def skill_write(body: dict[str, Any]) -> dict[str, Any]:
        try:
            slug = svc.skill_write(
                body["name"], body.get("when_to_use", ""), body["steps"]
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"slug": slug}

    @app.post("/skills/find", dependencies=auth)
    async def skill_find(body: dict[str, Any]) -> dict[str, Any]:
        hits = svc.skill_find(body["query"], k=int(body.get("k", 3)))
        return {"skills": wire.skills_to_list(hits)}

    @app.get("/skills/count", dependencies=auth)
    async def skill_count() -> dict[str, Any]:
        return {"count": svc.skill_count()}

    # --- procedural memory: workflows ---------------------------------------
    @app.post("/workflows/write", dependencies=auth)
    async def workflow_write(body: dict[str, Any]) -> dict[str, Any]:
        try:
            slug = svc.workflow_write(
                body["name"],
                body.get("when_to_use", ""),
                body["steps"],
                trigger=body.get("trigger", ""),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"slug": slug}

    @app.post("/workflows/find", dependencies=auth)
    async def workflow_find(body: dict[str, Any]) -> dict[str, Any]:
        hits = svc.workflow_find(body["query"], k=int(body.get("k", 3)))
        return {"workflows": wire.workflows_to_list(hits)}

    @app.get("/workflows/count", dependencies=auth)
    async def workflow_count() -> dict[str, Any]:
        return {"count": svc.workflow_count()}

    # --- tool-event log -----------------------------------------------------
    @app.post("/events/log", dependencies=auth)
    async def event_log(body: dict[str, Any]) -> dict[str, Any]:
        mid = svc.log_event(
            body["tool"], body.get("brief", ""), body.get("task_id", "")
        )
        return {"id": mid}

    @app.get("/events/by-task", dependencies=auth)
    async def events_by_task(task_id: str, limit: int = 500) -> dict[str, Any]:
        evs = svc.events_for_task(task_id, limit=limit)
        return {"events": wire.events_to_list(evs)}

    @app.get("/events/count", dependencies=auth)
    async def event_count() -> dict[str, Any]:
        return {"count": svc.event_count()}

    return app


def _write_sidecar(db_path: Path, url: str) -> Path:
    sidecar = Path(str(db_path) + ".daemon")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(f"pid={os.getpid()}\nurl={url}\n", encoding="utf-8")
    return sidecar


def serve(
    db_path: Path | str,
    host: str = "127.0.0.1",
    port: int = 8787,
    token: str | None = None,
    *,
    skills_dir: Path | str | None = None,
    workflows_dir: Path | str | None = None,
) -> None:
    """Run the daemon (blocking). Writes a sidecar file so a local in-process
    write can warn, and removes it on shutdown.

    ``skills_dir`` / ``workflows_dir`` default to the module defaults
    (``config.SKILLS_DIR`` / ``config.WORKFLOWS_DIR``) — i.e. the daemon serves
    skills/workflows from its own ``data/`` dirs, alongside its DB."""
    import atexit

    import uvicorn

    db_path = Path(db_path)
    app = create_app(
        db_path, token=token, skills_dir=skills_dir, workflows_dir=workflows_dir
    )
    sidecar = _write_sidecar(db_path, f"http://{host}:{port}")

    def _cleanup() -> None:
        try:
            sidecar.unlink()
        except OSError:
            pass

    # Belt-and-suspenders: uvicorn's SIGINT/SIGTERM handling returns normally,
    # so both the finally and atexit run; atexit also covers sys.exit paths. A
    # hard SIGKILL/crash can still orphan the sidecar — hence it is advisory only
    # (default_client warns, never refuses).
    atexit.register(_cleanup)
    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        _cleanup()
