"""MemoryService — the in-process application facade for long-term memory.

Every long-term-memory operation routes through this one object: save, recall,
forget, consolidate, and read-only stats. Today it calls the in-process
``MemoryStore`` directly; the point of the facade is that **transport** (a
standalone daemon later) becomes the only thing that changes — consumers depend
on a ``MemoryClient`` (see ``client.py``), never on the store internals.

Scope is deliberately **long-term memory only**. Skills, workflows, and the
event log stay in-process and are not part of this seam yet (consolidation is
included because it is memory upkeep, even though it currently also mines events
into workflows).

The default service resolves the module-level default store on each call, so it
honours ``store._DB_PATH`` reassignment (the test-isolation mechanism). Pass an
explicit ``MemoryStore`` to bind a service to a specific database.
"""

from __future__ import annotations

from . import consolidate as _consolidate
from . import store as _store_mod
from .store import Memory, MemoryStore


class MemoryService:
    def __init__(self, store: MemoryStore | None = None):
        self._store = store

    def _resolved(self) -> MemoryStore:
        # None → the module default (follows _DB_PATH); else the injected store.
        return self._store if self._store is not None else _store_mod._store()

    # --- writes -------------------------------------------------------------
    def save(
        self,
        text: str,
        kind: str = "fact",
        tags: str = "",
        importance: float | None = None,
    ) -> int:
        return self._resolved().save(text, kind=kind, tags=tags, importance=importance)

    def forget(self, query: str) -> Memory | None:
        """Archive the single memory best matching ``query``; return it (or None)."""
        s = self._resolved()
        hits = s.recall(query, k=1)
        if not hits:
            return None
        s.archive(hits[0].id)
        return hits[0]

    def archive(self, mem_id: int) -> None:
        self._resolved().archive(mem_id)

    # --- reads --------------------------------------------------------------
    def recall(
        self,
        query: str,
        k: int = 5,
        *,
        reinforce: bool = False,
        include_archived: bool = False,
    ) -> list[Memory]:
        return self._resolved().recall(
            query, k=k, reinforce=reinforce, include_archived=include_archived
        )

    def all_memories(self, include_archived: bool = True) -> list[Memory]:
        return self._resolved().all_memories(include_archived=include_archived)

    def count(self, include_archived: bool = True) -> int:
        return self._resolved().count(include_archived=include_archived)

    # --- maintenance --------------------------------------------------------
    def consolidate(self) -> _consolidate.ConsolidationReport:
        """Run the deterministic consolidation ('sleep') pass."""
        return _consolidate.run_consolidation()

    async def dream(self, ask_model=None):
        """Run the opt-in, LLM-driven REM ('dream') pass — an adversarial critic
        over recent memory. Like ``consolidate``, it operates on the module-level
        store. The model is an advisor; mutations stay deterministic/reversible."""
        from . import rem

        return await rem.run_rem(ask_model)
