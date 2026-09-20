"""MemoryService — the in-process application facade for long-term memory.

Every long-term-memory operation routes through this one object: save, recall,
forget, consolidate, and read-only stats. Today it calls the in-process
``MemoryStore`` directly; the point of the facade is that **transport** (a
standalone daemon later) becomes the only thing that changes — consumers depend
on a ``MemoryClient`` (see ``client.py``), never on the store internals.

Scope covers long-term memory, **procedural memory** (skills/workflows), **and
the tool-event log** — all route through this facade and the ``MemoryClient``
seam, so the daemon serves every consumer-facing operation. (Consolidation/dream
still mine the *module-level* defaults directly, which is exactly why the daemon
binds those globals rather than injecting a store.)

The default service resolves the module-level default store on each call, so it
honours ``store._DB_PATH`` reassignment (the test-isolation mechanism). Pass an
explicit ``MemoryStore`` to bind a service to a specific database.
"""

from __future__ import annotations

from . import consolidate as _consolidate
from . import events as _events
from . import skills as _skills
from . import store as _store_mod
from . import workflows as _workflows
from .events import Event
from .skills import Skill
from .store import Memory, MemoryStore
from .workflows import Workflow


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

    def archive(self, mem_id: int) -> bool:
        """Archive one memory by id (reversible — same tier REM uses). Returns
        False if there is no such memory, so a CLI can say so."""
        s = self._resolved()
        if s.get(mem_id) is None:
            return False
        s.archive(mem_id)
        return True

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

    def get(self, mem_id: int) -> Memory | None:
        return self._resolved().get(mem_id)

    def count(self, include_archived: bool = True) -> int:
        return self._resolved().count(include_archived=include_archived)

    # --- procedural memory (skills / workflows) -----------------------------
    # These delegate to the module functions at call time, so they honour
    # reassignment of ``skills._SKILLS_DIR`` / ``workflows._WORKFLOWS_DIR`` (the
    # test-isolation mechanism) and the daemon's dir binding, exactly like the
    # store methods honour ``store._DB_PATH``. There is no injected-store variant
    # because skills/workflows are filesystem-backed, not a ``MemoryStore``.
    def skill_write(self, name: str, when_to_use: str, steps: str) -> str:
        return _skills.write_skill(name, when_to_use, steps)

    def skill_find(self, query: str, k: int = 3) -> list[Skill]:
        return _skills.find_skills(query, k=k)

    def skill_count(self) -> int:
        return _skills.count()

    def workflow_write(
        self, name: str, when_to_use: str, steps: str, trigger: str = ""
    ) -> str:
        return _workflows.write_workflow(name, when_to_use, steps, trigger=trigger)

    def workflow_find(self, query: str, k: int = 3) -> list[Workflow]:
        return _workflows.find_workflows(query, k=k)

    def workflow_count(self) -> int:
        return _workflows.count()

    # --- tool-event log -----------------------------------------------------
    # Same call-time delegation as skills/workflows: honours ``events._DB_PATH``
    # reassignment and the daemon's ``_bind_db``. Consolidate mines the log
    # directly (module-level), so it is *not* routed through the client — but the
    # hooks that WRITE events and read one task's events do go through here, so a
    # daemon deployment sees them.
    def log_event(self, tool: str, brief: str = "", task_id: str = "") -> int:
        return _events.log_event(tool, brief=brief, task_id=task_id)

    def events_for_task(self, task_id: str, limit: int = 500) -> list[Event]:
        return _events.events_by_task(limit).get(task_id, [])

    def event_count(self) -> int:
        return _events.count()

    # --- maintenance --------------------------------------------------------
    def consolidate(self) -> _consolidate.ConsolidationReport:
        """Run the deterministic consolidation ('sleep') pass."""
        return _consolidate.run_consolidation()

    def maybe_consolidate(self) -> _consolidate.ConsolidationReport | None:
        """Run consolidation only if enough events have accrued since last time.

        The throttle (``should_auto_run``) and the run are decided **together,
        here**, so both read the same event log + watermark. This must not be
        split into a client-side gate + a remote run: under a daemon the gate
        would read the caller's (empty) local event log while the work executes
        server-side, so auto-consolidation would silently never fire."""
        if not _consolidate.should_auto_run():
            return None
        return _consolidate.run_consolidation()

    async def dream(self, ask_model=None):
        """Run the opt-in, LLM-driven REM ('dream') pass — an adversarial critic
        over recent memory. Like ``consolidate``, it operates on the module-level
        store. The model is an advisor; mutations stay deterministic/reversible."""
        from . import rem

        return await rem.run_rem(ask_model)
