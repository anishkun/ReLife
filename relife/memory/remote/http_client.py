"""``HttpMemoryClient`` — the out-of-process transport for long-term memory.

Implements the ``MemoryClient`` protocol (``memory/client.py``) against a running
``relife memory serve`` daemon, so it drops in for ``LocalMemoryClient`` with no
consumer change. It rebuilds real ``Memory`` / report objects from the wire dicts
so callers get the identical types they'd get in-process.

- **Pooled client.** One keep-alive ``httpx.Client`` is reused across calls, so
  recall overhead stays ~1-3 ms (a fresh TCP handshake per call would dwarf
  that). Tests inject an ``ASGITransport``-backed client to talk to the app with
  no real socket.
- **dream.** The ``ask_model`` callable can't cross the wire, so it is ignored
  (the daemon uses its own default). REM runs for minutes, so its POST uses
  ``timeout=None``, and — since ``dream`` is the one ``async`` protocol method —
  it is dispatched via ``anyio.to_thread`` to avoid blocking the caller's loop.
  The short, frequent ``save``/``recall`` calls stay synchronous by design (the
  protocol is sync; wrapping them would force awaits into consumers).

``httpx`` is an optional (``[daemon]``) dependency; importing this module needs
it.
"""

from __future__ import annotations

from typing import Any

import anyio
import httpx

from ..events import Event
from ..skills import Skill
from ..spaces import DEFAULT_SPACE
from ..store import Memory
from ..workflows import Workflow
from . import wire

# Long enough that a warm recall (embedding + DB) never trips it, short enough to
# fail fast if the daemon is down. dream overrides this with timeout=None.
_DEFAULT_TIMEOUT = 30.0


class HttpMemoryClient:
    """Talks to the memory daemon over pooled httpx. Mirrors ``LocalMemoryClient``."""

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        client: httpx.Client | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        # An injected client (tests use ASGITransport) already carries base_url;
        # otherwise build a pooled keep-alive client bound to the daemon.
        self._client = client or httpx.Client(
            base_url=self._base_url, headers=headers, timeout=_DEFAULT_TIMEOUT
        )
        # If a client was injected without our auth header, still send it.
        if client is not None and token:
            self._client.headers.setdefault("Authorization", f"Bearer {token}")

    # --- helpers ------------------------------------------------------------
    @staticmethod
    def _check(r) -> None:
        # The daemon answers invalid input (a bad space name, an empty memory)
        # with 400; surface it as the ValueError the in-process path raises.
        if r.status_code == 400:
            raise ValueError(r.json().get("detail", "invalid request"))
        r.raise_for_status()

    def _post(self, path: str, json: dict[str, Any] | None = None, **kw) -> dict[str, Any]:
        r = self._client.post(path, json=json or {}, **kw)
        self._check(r)
        return r.json()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        r = self._client.get(path, params=params or {})
        self._check(r)
        return r.json()

    @staticmethod
    def _space_params(params: dict[str, Any], spaces) -> dict[str, Any]:
        if spaces is not None:
            params["spaces"] = list(spaces)
        return params

    def _post_write(self, path: str, json: dict[str, Any]) -> dict[str, Any]:
        """POST a write, translating the daemon's 400 (invalid input) back into
        the ``ValueError`` the in-process path raises, so both transports fail
        identically for consumers (and the conformance suite stays parametrizable)."""
        # Checked on the response, not by catching ``httpx.HTTPStatusError``: an
        # injected client (Starlette's TestClient now rides ``httpx2``) raises
        # its own library's exception class.
        r = self._client.post(path, json=json)
        if r.status_code == 400:
            raise ValueError(r.json().get("detail", "invalid request"))
        r.raise_for_status()
        return r.json()

    # --- MemoryClient protocol ---------------------------------------------
    def save(
        self, text, kind="fact", tags="", importance=None, *, space=DEFAULT_SPACE, source=""
    ) -> int:
        data = self._post(
            "/save",
            {
                "text": text,
                "kind": kind,
                "tags": tags,
                "importance": importance,
                "space": space,
                "source": source,
            },
        )
        return int(data["id"])

    def recall(
        self, query, k=5, *, reinforce=False, include_archived=False, spaces=None
    ) -> list[Memory]:
        body = {
            "query": query,
            "k": k,
            "reinforce": reinforce,
            "include_archived": include_archived,
        }
        data = self._post("/recall", self._space_params(body, spaces))
        return wire.memories_from_list(data["memories"])

    def forget(self, query, *, space=DEFAULT_SPACE) -> Memory | None:
        data = self._post("/forget", {"query": query, "space": space})
        return wire.memory_or_none_from_dict(data["memory"])

    def archive(self, mem_id, *, space=None) -> bool:
        return bool(self._post("/archive", {"id": int(mem_id), "space": space})["archived"])

    def get(self, mem_id) -> Memory | None:
        data = self._get(f"/memories/{int(mem_id)}")
        return wire.memory_or_none_from_dict(data["memory"])

    def all_memories(self, include_archived=True, *, spaces=None) -> list[Memory]:
        params = self._space_params({"include_archived": include_archived}, spaces)
        data = self._get("/memories", params)
        return wire.memories_from_list(data["memories"])

    def count(self, include_archived=True, *, spaces=None) -> int:
        params = self._space_params({"include_archived": include_archived}, spaces)
        data = self._get("/count", params)
        return int(data["count"])

    # --- memory spaces --------------------------------------------------------
    def spaces(self) -> dict[str, dict[str, int]]:
        return self._get("/spaces")["spaces"]

    def copy_space(self, src, dst, *, ids=None, source=None, procedural=True) -> dict[str, int]:
        return self._post(
            "/spaces/copy",
            {
                "src": src,
                "dst": dst,
                "ids": None if ids is None else [int(i) for i in ids],
                "source": source,
                "procedural": procedural,
            },
        )

    def archive_space(self, space) -> int:
        return int(self._post("/spaces/archive", {"space": space})["archived"])

    def export_space(self, space) -> dict[str, Any]:
        return self._get(f"/spaces/{space}/export")

    def import_pack(self, pack, space) -> dict[str, int]:
        return self._post("/spaces/import", {"pack": pack, "space": space})

    def consolidate(self):
        return wire.consolidation_from_dict(self._post("/consolidate"))

    def maybe_consolidate(self):
        data = self._post("/consolidate/maybe")
        return wire.consolidation_from_dict(data) if data is not None else None

    async def dream(self, ask_model=None):
        # ask_model can't be serialized — the daemon uses its own default. REM
        # runs for minutes (timeout=None) and dream is async, so dispatch the
        # blocking POST to a worker thread to keep the caller's loop responsive.
        data = await anyio.to_thread.run_sync(
            lambda: self._post("/dream", timeout=None)
        )
        return wire.rem_from_dict(data)

    # --- procedural memory (skills / workflows) -----------------------------
    def skill_write(self, name, when_to_use, steps, *, space=DEFAULT_SPACE) -> str:
        data = self._post_write(
            "/skills/write",
            {"name": name, "when_to_use": when_to_use, "steps": steps, "space": space},
        )
        return str(data["slug"])

    def skill_find(self, query, k=3, *, spaces=None) -> list[Skill]:
        data = self._post("/skills/find", self._space_params({"query": query, "k": k}, spaces))
        return wire.skills_from_list(data["skills"])

    def skill_count(self, *, space=DEFAULT_SPACE) -> int:
        return int(self._get("/skills/count", {"space": space})["count"])

    def workflow_write(self, name, when_to_use, steps, trigger="", *, space=DEFAULT_SPACE) -> str:
        data = self._post_write(
            "/workflows/write",
            {
                "name": name,
                "when_to_use": when_to_use,
                "steps": steps,
                "trigger": trigger,
                "space": space,
            },
        )
        return str(data["slug"])

    def workflow_find(self, query, k=3, *, spaces=None) -> list[Workflow]:
        data = self._post(
            "/workflows/find", self._space_params({"query": query, "k": k}, spaces)
        )
        return wire.workflows_from_list(data["workflows"])

    def workflow_count(self, *, space=DEFAULT_SPACE) -> int:
        return int(self._get("/workflows/count", {"space": space})["count"])

    # --- tool-event log -----------------------------------------------------
    def log_event(self, tool, brief="", task_id="", *, space=DEFAULT_SPACE) -> int:
        data = self._post(
            "/events/log", {"tool": tool, "brief": brief, "task_id": task_id, "space": space}
        )
        return int(data["id"])

    def events_for_task(self, task_id, limit=500) -> list[Event]:
        data = self._get("/events/by-task", {"task_id": task_id, "limit": limit})
        return wire.events_from_list(data["events"])

    def event_count(self) -> int:
        return int(self._get("/events/count")["count"])

    def close(self) -> None:
        self._client.close()
