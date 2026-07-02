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
"""

from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable

from .. import config
from .service import MemoryService
from .store import Memory


@runtime_checkable
class MemoryClient(Protocol):
    def save(self, text: str, kind: str = ..., tags: str = ..., importance: float | None = ...) -> int: ...
    def recall(self, query: str, k: int = ..., *, reinforce: bool = ..., include_archived: bool = ...) -> list[Memory]: ...
    def forget(self, query: str) -> Memory | None: ...
    def all_memories(self, include_archived: bool = ...) -> list[Memory]: ...
    def count(self, include_archived: bool = ...) -> int: ...
    def consolidate(self): ...
    async def dream(self, ask_model=...): ...


class LocalMemoryClient:
    """In-process transport: direct calls to a ``MemoryService``."""

    def __init__(self, service: MemoryService | None = None):
        self._svc = service or MemoryService()

    def save(self, text, kind="fact", tags="", importance=None) -> int:
        return self._svc.save(text, kind=kind, tags=tags, importance=importance)

    def recall(self, query, k=5, *, reinforce=False, include_archived=False) -> list[Memory]:
        return self._svc.recall(
            query, k=k, reinforce=reinforce, include_archived=include_archived
        )

    def forget(self, query) -> Memory | None:
        return self._svc.forget(query)

    def all_memories(self, include_archived=True) -> list[Memory]:
        return self._svc.all_memories(include_archived=include_archived)

    def count(self, include_archived=True) -> int:
        return self._svc.count(include_archived=include_archived)

    def consolidate(self):
        return self._svc.consolidate()

    async def dream(self, ask_model=None):
        return await self._svc.dream(ask_model)


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
