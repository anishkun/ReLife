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

from typing import Protocol, runtime_checkable

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


_default: LocalMemoryClient | None = None


def default_client() -> LocalMemoryClient:
    """Process-wide default memory client (in-process transport)."""
    global _default
    if _default is None:
        _default = LocalMemoryClient()
    return _default
