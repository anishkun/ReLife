"""MemoryClient — the consumer-facing seam for long-term memory.

Consumers (the MCP tool layer, the recall/episode hooks, the CLI) depend on this
interface, never on the store or service internals. The default
``LocalMemoryClient`` makes direct in-process calls to a ``MemoryService`` — so
this seam is, today, a no-op indirection. Its whole purpose is forward
compatibility: when memory becomes a standalone process, an ``HttpMemoryClient``
implementing the same ``MemoryClient`` protocol drops in with no change to any
consumer.

``default_client()`` returns a process-wide default so call sites don't each
construct one; tests can still point it at an isolated store via the service.

``ScopedMemoryClient`` wraps any client in one agent's ``MemoryScope``: it is
what a non-default agent's hooks and memory tools are handed, and the place the
"an agent writes only its own space" rule is enforced on the consumer side.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Sequence
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable

import anyio

from .. import config
from .events import Event
from .service import MemoryService
from .skills import Skill
from .spaces import DEFAULT_SCOPE, DEFAULT_SPACE, MemoryScope
from .store import Memory
from .workflows import Workflow


_T = TypeVar("_T")


async def off_loop(fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
    """Run a sync client call on a worker thread, for callers on an event loop.

    The client is synchronous, but its async consumers — the memory hooks and
    the memory MCP tools — run on the loop that ``relife serve`` shares across
    *every* session. Inline, each recall (FTS scan, ONNX inference with
    embeddings on), each journaled tool call (a loopback POST in daemon mode)
    and an agent-requested consolidation froze every other session's stream
    and pending approval for its duration. Safe to thread: the store opens a
    fresh SQLite connection per call, the embedding model loads under a lock,
    and ``httpx.Client`` is thread-safe.
    """
    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


@runtime_checkable
class MemoryClient(Protocol):
    # Every agent-facing call takes the space(s) it acts on, defaulting to the
    # main agent's ``default`` space; admin listings default to every space.
    def save(self, text: str, kind: str = ..., tags: str = ..., importance: float | None = ..., *, space: str = ..., source: str = ...) -> int: ...
    def recall(self, query: str, k: int = ..., *, reinforce: bool = ..., include_archived: bool = ..., spaces: Sequence[str] | None = ...) -> list[Memory]: ...
    def forget(self, query: str, *, space: str = ...) -> Memory | None: ...
    def archive(self, mem_id: int, *, space: str | None = ...) -> bool: ...
    def get(self, mem_id: int) -> Memory | None: ...
    def all_memories(self, include_archived: bool = ..., *, spaces: Sequence[str] | None = ...) -> list[Memory]: ...
    def count(self, include_archived: bool = ..., *, spaces: Sequence[str] | None = ...) -> int: ...
    def consolidate(self): ...
    def maybe_consolidate(self): ...
    async def dream(self, ask_model=...): ...
    # Memory spaces (agent partitions): list, fork/promote, export/import.
    def spaces(self) -> dict[str, dict[str, int]]: ...
    def copy_space(self, src: str, dst: str, *, ids: Sequence[int] | None = ..., source: str | None = ..., procedural: bool = ...) -> dict[str, int]: ...
    def archive_space(self, space: str) -> int: ...
    def export_space(self, space: str) -> dict[str, Any]: ...
    def import_pack(self, pack: dict[str, Any], space: str) -> dict[str, int]: ...
    # Procedural memory (skills / workflows).
    def skill_write(self, name: str, when_to_use: str, steps: str, *, space: str = ...) -> str: ...
    def skill_find(self, query: str, k: int = ..., *, spaces: Sequence[str] | None = ...) -> list[Skill]: ...
    def skill_count(self, *, space: str = ...) -> int: ...
    def workflow_write(self, name: str, when_to_use: str, steps: str, trigger: str = ..., *, space: str = ...) -> str: ...
    def workflow_find(self, query: str, k: int = ..., *, spaces: Sequence[str] | None = ...) -> list[Workflow]: ...
    def workflow_count(self, *, space: str = ...) -> int: ...
    # Tool-event log.
    def log_event(self, tool: str, brief: str = ..., task_id: str = ..., *, space: str = ...) -> int: ...
    def events_for_task(self, task_id: str, limit: int = ...) -> list[Event]: ...
    def event_count(self) -> int: ...


class LocalMemoryClient:
    """In-process transport: direct calls to a ``MemoryService``."""

    def __init__(self, service: MemoryService | None = None):
        self._svc = service or MemoryService()

    def save(
        self, text, kind="fact", tags="", importance=None, *, space=DEFAULT_SPACE, source=""
    ) -> int:
        return self._svc.save(
            text, kind=kind, tags=tags, importance=importance, space=space, source=source
        )

    def recall(
        self, query, k=5, *, reinforce=False, include_archived=False, spaces=None
    ) -> list[Memory]:
        return self._svc.recall(
            query, k=k, reinforce=reinforce, include_archived=include_archived, spaces=spaces
        )

    def forget(self, query, *, space=DEFAULT_SPACE) -> Memory | None:
        return self._svc.forget(query, space=space)

    def archive(self, mem_id, *, space=None) -> bool:
        return self._svc.archive(mem_id, space=space)

    def get(self, mem_id) -> Memory | None:
        return self._svc.get(mem_id)

    def all_memories(self, include_archived=True, *, spaces=None) -> list[Memory]:
        return self._svc.all_memories(include_archived=include_archived, spaces=spaces)

    def count(self, include_archived=True, *, spaces=None) -> int:
        return self._svc.count(include_archived=include_archived, spaces=spaces)

    def spaces(self) -> dict[str, dict[str, int]]:
        return self._svc.spaces()

    def copy_space(self, src, dst, *, ids=None, source=None, procedural=True) -> dict[str, int]:
        return self._svc.copy_space(src, dst, ids=ids, source=source, procedural=procedural)

    def archive_space(self, space) -> int:
        return self._svc.archive_space(space)

    def export_space(self, space) -> dict[str, Any]:
        return self._svc.export_space(space)

    def import_pack(self, pack, space) -> dict[str, int]:
        return self._svc.import_pack(pack, space)

    def consolidate(self):
        return self._svc.consolidate()

    def maybe_consolidate(self):
        return self._svc.maybe_consolidate()

    async def dream(self, ask_model=None):
        return await self._svc.dream(ask_model)

    def skill_write(self, name, when_to_use, steps, *, space=DEFAULT_SPACE) -> str:
        return self._svc.skill_write(name, when_to_use, steps, space=space)

    def skill_find(self, query, k=3, *, spaces=None) -> list[Skill]:
        return self._svc.skill_find(query, k=k, spaces=spaces)

    def skill_count(self, *, space=DEFAULT_SPACE) -> int:
        return self._svc.skill_count(space=space)

    def workflow_write(self, name, when_to_use, steps, trigger="", *, space=DEFAULT_SPACE) -> str:
        return self._svc.workflow_write(name, when_to_use, steps, trigger=trigger, space=space)

    def workflow_find(self, query, k=3, *, spaces=None) -> list[Workflow]:
        return self._svc.workflow_find(query, k=k, spaces=spaces)

    def workflow_count(self, *, space=DEFAULT_SPACE) -> int:
        return self._svc.workflow_count(space=space)

    def log_event(self, tool, brief="", task_id="", *, space=DEFAULT_SPACE) -> int:
        return self._svc.log_event(tool, brief=brief, task_id=task_id, space=space)

    def events_for_task(self, task_id, limit=500) -> list[Event]:
        return self._svc.events_for_task(task_id, limit=limit)

    def event_count(self) -> int:
        return self._svc.event_count()


class ScopedMemoryClient:
    """A ``MemoryClient`` confined to one agent's :class:`MemoryScope`.

    Reads (recall, skill/workflow find, listings, ``get``) see only
    ``scope.read``; writes (save, skills, workflows, journaled events) land only
    in ``scope.write``, stamped with ``scope.source``; ``forget``/``archive``
    reach only ``scope.write`` — an agent can never archive what it inherited.

    The agent-facing methods deliberately take **no** space argument: passing
    one is a ``TypeError``, not a silent override, so no caller (and no model
    output routed through a caller) can widen the scope. Space administration
    (copy/export/import/archive_space) is the user's, through the unscoped
    client — a scoped agent gets ``PermissionError``.
    """

    def __init__(self, inner: MemoryClient, scope: MemoryScope = DEFAULT_SCOPE):
        self._inner = inner
        self.scope = scope

    # --- writes -------------------------------------------------------------
    def save(self, text, kind="fact", tags="", importance=None) -> int:
        return self._inner.save(
            text,
            kind=kind,
            tags=tags,
            importance=importance,
            space=self.scope.write,
            source=self.scope.source,
        )

    def forget(self, query) -> Memory | None:
        return self._inner.forget(query, space=self.scope.write)

    def archive(self, mem_id) -> bool:
        return self._inner.archive(mem_id, space=self.scope.write)

    # --- reads --------------------------------------------------------------
    def recall(self, query, k=5, *, reinforce=False, include_archived=False) -> list[Memory]:
        return self._inner.recall(
            query,
            k=k,
            reinforce=reinforce,
            include_archived=include_archived,
            spaces=self.scope.read,
        )

    def get(self, mem_id) -> Memory | None:
        m = self._inner.get(mem_id)
        return m if m is not None and m.space in self.scope.read else None

    def all_memories(self, include_archived=True) -> list[Memory]:
        return self._inner.all_memories(include_archived=include_archived, spaces=self.scope.read)

    def count(self, include_archived=True) -> int:
        return self._inner.count(include_archived=include_archived, spaces=self.scope.read)

    # --- upkeep -------------------------------------------------------------
    def consolidate(self):
        # Deterministic, and per-space inside (it never merges across spaces),
        # so an agent tidying memory can't touch another agent's.
        return self._inner.consolidate()

    def maybe_consolidate(self):
        return self._inner.maybe_consolidate()

    async def dream(self, ask_model=None):
        # REM reviews (and may archive) memory across the store and spends Max
        # budget; only the main agent may start it.
        if self.scope.write != DEFAULT_SPACE:
            raise PermissionError("the REM (dream) pass is run by the main agent only")
        return await self._inner.dream(ask_model)

    # --- spaces (administration is the user's, not an agent's) --------------
    def spaces(self) -> dict[str, dict[str, int]]:
        visible = self._inner.spaces()
        return {k: v for k, v in visible.items() if k in self.scope.read}

    def _admin(self, *_a, **_kw):
        raise PermissionError("memory spaces are administered by the user, not by an agent")

    copy_space = archive_space = export_space = import_pack = _admin

    # --- procedural memory --------------------------------------------------
    def skill_write(self, name, when_to_use, steps) -> str:
        return self._inner.skill_write(name, when_to_use, steps, space=self.scope.write)

    def skill_find(self, query, k=3) -> list[Skill]:
        return self._inner.skill_find(query, k=k, spaces=self.scope.read)

    def skill_count(self) -> int:
        return self._inner.skill_count(space=self.scope.write)

    def workflow_write(self, name, when_to_use, steps, trigger="") -> str:
        return self._inner.workflow_write(
            name, when_to_use, steps, trigger=trigger, space=self.scope.write
        )

    def workflow_find(self, query, k=3) -> list[Workflow]:
        return self._inner.workflow_find(query, k=k, spaces=self.scope.read)

    def workflow_count(self) -> int:
        return self._inner.workflow_count(space=self.scope.write)

    # --- tool-event log -----------------------------------------------------
    def log_event(self, tool, brief="", task_id="") -> int:
        return self._inner.log_event(tool, brief=brief, task_id=task_id, space=self.scope.write)

    def events_for_task(self, task_id, limit=500) -> list[Event]:
        return [
            e
            for e in self._inner.events_for_task(task_id, limit=limit)
            if e.space == self.scope.write
        ]

    def event_count(self) -> int:
        return self._inner.event_count()


_default: MemoryClient | None = None


def _warn_if_daemon_running() -> None:
    """Warn (don't refuse) if a daemon appears to own the DB but we're going
    in-process. A stale sidecar after a crash must not brick the default path,
    so this is advisory only."""
    if config.MEMORY_SIDECAR_PATH.exists():
        warnings.warn(
            "A memory daemon appears to be running "
            f"({config.MEMORY_SIDECAR_PATH}); this process is using in-process "
            "memory instead. Set RELIFE_MEMORY_URL to route through the daemon "
            "and avoid concurrent writers.",
            stacklevel=2,
        )


def default_client() -> MemoryClient:
    """Process-wide default memory client.

    Transport is chosen by environment: ``RELIFE_MEMORY_URL`` set → an
    ``HttpMemoryClient`` against a standalone daemon; unset → the in-process
    ``LocalMemoryClient`` (the default — no daemon required). The choice is
    cached, so it is fixed for the life of the process.
    """
    global _default
    if _default is None:
        if config.MEMORY_URL:
            # Lazy import: httpx/the remote transport is an optional extra.
            from .remote.http_client import HttpMemoryClient

            _default = HttpMemoryClient(config.MEMORY_URL, token=config.MEMORY_TOKEN)
        else:
            _warn_if_daemon_running()
            _default = LocalMemoryClient()
    return _default
