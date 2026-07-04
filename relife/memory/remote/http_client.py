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

from ..skills import Skill
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
    def _post(self, path: str, json: dict[str, Any] | None = None, **kw) -> dict[str, Any]:
        r = self._client.post(path, json=json or {}, **kw)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        r = self._client.get(path, params=params or {})
        r.raise_for_status()
        return r.json()

    def _post_write(self, path: str, json: dict[str, Any]) -> dict[str, Any]:
        """POST a write, translating the daemon's 400 (invalid input) back into
        the ``ValueError`` the in-process path raises, so both transports fail
        identically for consumers (and the conformance suite stays parametrizable)."""
        try:
            return self._post(path, json)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                detail = e.response.json().get("detail", str(e))
                raise ValueError(detail) from None
            raise

    # --- MemoryClient protocol ---------------------------------------------
    def save(self, text, kind="fact", tags="", importance=None) -> int:
        data = self._post(
            "/save",
            {"text": text, "kind": kind, "tags": tags, "importance": importance},
        )
        return int(data["id"])

    def recall(self, query, k=5, *, reinforce=False, include_archived=False) -> list[Memory]:
        data = self._post(
            "/recall",
            {
                "query": query,
                "k": k,
                "reinforce": reinforce,
                "include_archived": include_archived,
            },
        )
        return wire.memories_from_list(data["memories"])

    def forget(self, query) -> Memory | None:
        data = self._post("/forget", {"query": query})
        return wire.memory_or_none_from_dict(data["memory"])

    def all_memories(self, include_archived=True) -> list[Memory]:
        data = self._get("/memories", {"include_archived": include_archived})
        return wire.memories_from_list(data["memories"])

    def count(self, include_archived=True) -> int:
        data = self._get("/count", {"include_archived": include_archived})
        return int(data["count"])

    def consolidate(self):
        return wire.consolidation_from_dict(self._post("/consolidate"))

    async def dream(self, ask_model=None):
        # ask_model can't be serialized — the daemon uses its own default. REM
        # runs for minutes (timeout=None) and dream is async, so dispatch the
        # blocking POST to a worker thread to keep the caller's loop responsive.
        data = await anyio.to_thread.run_sync(
            lambda: self._post("/dream", timeout=None)
        )
        return wire.rem_from_dict(data)

    # --- procedural memory (skills / workflows) -----------------------------
    def skill_write(self, name, when_to_use, steps) -> str:
        data = self._post_write(
            "/skills/write",
            {"name": name, "when_to_use": when_to_use, "steps": steps},
        )
        return str(data["slug"])

    def skill_find(self, query, k=3) -> list[Skill]:
        data = self._post("/skills/find", {"query": query, "k": k})
        return wire.skills_from_list(data["skills"])

    def skill_count(self) -> int:
        return int(self._get("/skills/count")["count"])

    def workflow_write(self, name, when_to_use, steps, trigger="") -> str:
        data = self._post_write(
            "/workflows/write",
            {
                "name": name,
                "when_to_use": when_to_use,
                "steps": steps,
                "trigger": trigger,
            },
        )
        return str(data["slug"])

    def workflow_find(self, query, k=3) -> list[Workflow]:
        data = self._post("/workflows/find", {"query": query, "k": k})
        return wire.workflows_from_list(data["workflows"])

    def workflow_count(self) -> int:
        return int(self._get("/workflows/count")["count"])

    def close(self) -> None:
        self._client.close()
