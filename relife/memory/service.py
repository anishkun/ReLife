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

**Spaces.** Every operation takes the memory space(s) it acts on (see
``spaces.py``). Omitted, an agent-facing operation acts on the ``default`` space
-- the main agent's memory, i.e. exactly the pre-spaces behaviour -- while the
admin listings (``all_memories``/``count``) cover every space. Which spaces an
*agent* may touch is decided by its ``MemoryScope`` (``client.ScopedMemoryClient``),
never by this facade or by the model.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from . import consolidate as _consolidate
from . import events as _events
from . import skills as _skills
from . import spaces as _spaces_mod
from . import store as _store_mod
from . import workflows as _workflows
from .events import Event
from .skills import Skill
from .spaces import DEFAULT_SPACE, validate_space
from ._procedure import check_procedure
from .store import _VALID_KINDS, MAX_TAGS_CHARS, MAX_TEXT_CHARS, Memory, MemoryStore
from .workflows import Workflow

PACK_FORMAT = "relife-memory-pack"
PACK_VERSION = 1


def _read_spaces(spaces: Sequence[str] | None) -> tuple[str, ...]:
    """An agent-facing read with no explicit spaces reads the default space."""
    return (DEFAULT_SPACE,) if spaces is None else tuple(spaces)


MAX_PACK_ITEMS = 50_000


def validate_pack(pack: Any) -> dict[str, Any]:
    """Check an exported memory pack's shape before anything is written.

    Packs travel between installs, so they are untrusted input: wrong format or
    version, non-list sections, or entries without text are rejected up front
    (``ValueError``) rather than half-imported."""
    if not isinstance(pack, dict) or pack.get("format") != PACK_FORMAT:
        raise ValueError("not a ReLife memory pack")
    if pack.get("version") != PACK_VERSION:
        raise ValueError(f"unsupported memory pack version {pack.get('version')!r}")
    for section in ("memories", "skills", "workflows"):
        items = pack.get(section, [])
        if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
            raise ValueError(f"memory pack section {section!r} must be a list of objects")
    if sum(len(pack.get(s, [])) for s in ("memories", "skills", "workflows")) > MAX_PACK_ITEMS:
        raise ValueError(f"memory pack holds more than {MAX_PACK_ITEMS} items")
    for m in pack.get("memories", []):
        if not isinstance(m.get("text"), str) or not m["text"].strip():
            raise ValueError("memory pack entry without text")
        if len(m["text"].strip()) > MAX_TEXT_CHARS:
            raise ValueError(f"memory pack entry longer than {MAX_TEXT_CHARS} characters")
        if len(str(m.get("tags") or "")) > MAX_TAGS_CHARS:
            raise ValueError(f"memory pack entry with tags longer than {MAX_TAGS_CHARS} characters")
    for section, label in (("skills", "skill"), ("workflows", "workflow")):
        for item in pack.get(section, []):
            if not isinstance(item.get("name"), str) or not isinstance(item.get("body"), str):
                raise ValueError(f"memory pack {label} without a name and body")
            try:
                check_procedure(item["name"], item["body"])
            except ValueError as e:
                raise ValueError(f"memory pack {label}: {e}") from None
    return pack


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
        *,
        space: str = DEFAULT_SPACE,
        source: str = "",
    ) -> int:
        return self._resolved().save(
            text, kind=kind, tags=tags, importance=importance, space=space, source=source
        )

    def forget(self, query: str, *, space: str = DEFAULT_SPACE) -> Memory | None:
        """Archive the memory in ``space`` best matching ``query``; return it (or None).
        Only that one space is searched -- forgetting never reaches another's."""
        s = self._resolved()
        hits = s.recall(query, k=1, spaces=(space,))
        if not hits:
            return None
        s.archive(hits[0].id)
        return hits[0]

    def archive(self, mem_id: int, *, space: str | None = None) -> bool:
        """Archive one memory by id (reversible -- same tier REM uses). Returns
        False if there is no such memory -- or, when ``space`` is given, if it
        lives in a different space -- so a CLI (or a scoped agent) can say so."""
        s = self._resolved()
        m = s.get(mem_id)
        if m is None or (space is not None and m.space != space):
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
        spaces: Sequence[str] | None = None,
        reinforce_space: str | None = None,
    ) -> list[Memory]:
        return self._resolved().recall(
            query,
            k=k,
            reinforce=reinforce,
            include_archived=include_archived,
            spaces=_read_spaces(spaces),
            reinforce_space=reinforce_space,
        )

    def all_memories(
        self, include_archived: bool = True, *, spaces: Sequence[str] | None = None
    ) -> list[Memory]:
        return self._resolved().all_memories(include_archived=include_archived, spaces=spaces)

    def get(self, mem_id: int) -> Memory | None:
        return self._resolved().get(mem_id)

    def count(
        self, include_archived: bool = True, *, spaces: Sequence[str] | None = None
    ) -> int:
        return self._resolved().count(include_archived=include_archived, spaces=spaces)

    # --- spaces -------------------------------------------------------------
    def spaces(self) -> dict[str, dict[str, int]]:
        """Every space that holds anything: ``{space: {memories, skills, workflows}}``
        (active memories only)."""
        mem = self._resolved().space_counts(include_archived=False)
        names = set(mem) | {DEFAULT_SPACE}
        root = _spaces_mod._SPACES_DIR
        if root.is_dir():
            for d in root.iterdir():
                if d.is_dir():
                    try:
                        names.add(validate_space(d.name))
                    except ValueError:
                        continue  # a stray dir is not a space
        return {
            n: {
                "memories": mem.get(n, 0),
                "skills": _skills.count(n),
                "workflows": _workflows.count(n),
            }
            for n in sorted(names)
        }

    def copy_space(
        self,
        src: str,
        dst: str,
        *,
        ids: Sequence[int] | None = None,
        source: str | None = None,
        procedural: bool = True,
    ) -> dict[str, int]:
        """Copy ``src`` into ``dst`` -- the fork/promote primitive. Memories keep
        their strength and provenance; skills/workflows ``dst`` already has are
        never overwritten. ``ids`` limits it to those memories (and skips the
        procedural copy); ``procedural=False`` copies memories only."""
        n = self._resolved().copy_space(src, dst, ids=ids, source=source)
        copy_files = procedural and ids is None
        return {
            "memories": n,
            "skills": _skills.copy_space(src, dst) if copy_files else 0,
            "workflows": _workflows.copy_space(src, dst) if copy_files else 0,
        }

    def archive_space(self, space: str) -> int:
        """Archive every active memory in ``space`` (reversible)."""
        return self._resolved().archive_space(space)

    def export_space(self, space: str) -> dict[str, Any]:
        """A portable ``relife-memory-pack`` of ``space``'s active memories,
        skills and workflows (plain JSON-able dict)."""
        validate_space(space)
        mems = self._resolved().all_memories(include_archived=False, spaces=(space,))
        return {
            "format": PACK_FORMAT,
            "version": PACK_VERSION,
            "space": space,
            "exported_at": time.time(),
            "memories": [
                {
                    "kind": m.kind,
                    "text": m.text,
                    "tags": m.tags,
                    "importance": m.importance,
                    "source": m.source,
                }
                for m in mems
            ],
            "skills": [
                {"name": s.name, "when_to_use": s.when_to_use, "body": s.body}
                for s in _skills.list_skills(space)
            ],
            "workflows": [
                {"name": w.name, "when_to_use": w.when_to_use, "trigger": w.trigger, "body": w.body}
                for w in _workflows.list_workflows(space)
            ],
        }

    def import_pack(self, pack: dict[str, Any], space: str) -> dict[str, int]:
        """Load a pack into ``space``. Memories go through ``save`` (so a text
        already there is reinforced, not cloned) and are stamped
        ``import:<origin space>``; skills/workflows the space already has are
        kept, not overwritten. Validated in full before anything is written."""
        validate_space(space)
        validate_pack(pack)
        origin = str(pack.get("space") or "unknown")
        st = self._resolved()
        out = {"memories": 0, "skills": 0, "workflows": 0}
        for m in pack.get("memories", []):
            kind = m.get("kind") if m.get("kind") in _VALID_KINDS else "fact"
            imp = m.get("importance")
            st.save(
                m["text"],
                kind=kind,
                tags=str(m.get("tags") or ""),
                importance=imp,  # save() drops junk (non-numeric, NaN, inf)
                space=space,
                source=f"import:{origin}",
            )
            out["memories"] += 1
        for sk in pack.get("skills", []):
            if _skills.read_skill(sk["name"], space) is None:
                _skills.write_skill(
                    sk["name"], str(sk.get("when_to_use") or ""), sk["body"], space=space
                )
                out["skills"] += 1
        for wf in pack.get("workflows", []):
            if _workflows.read_workflow(wf["name"], space) is None:
                _workflows.write_workflow(
                    wf["name"],
                    str(wf.get("when_to_use") or ""),
                    wf["body"],
                    trigger=str(wf.get("trigger") or ""),
                    space=space,
                )
                out["workflows"] += 1
        return out

    # --- procedural memory (skills / workflows) -----------------------------
    # These delegate to the module functions at call time, so they honour
    # reassignment of ``skills._SKILLS_DIR`` / ``workflows._WORKFLOWS_DIR`` (the
    # test-isolation mechanism) and the daemon's dir binding, exactly like the
    # store methods honour ``store._DB_PATH``. There is no injected-store variant
    # because skills/workflows are filesystem-backed, not a ``MemoryStore``.
    def skill_write(
        self, name: str, when_to_use: str, steps: str, *, space: str = DEFAULT_SPACE
    ) -> str:
        return _skills.write_skill(name, when_to_use, steps, space=space)

    def skill_find(
        self, query: str, k: int = 3, *, spaces: Sequence[str] | None = None
    ) -> list[Skill]:
        return _skills.find_skills(query, k=k, spaces=_read_spaces(spaces))

    def skill_count(self, *, space: str = DEFAULT_SPACE) -> int:
        return _skills.count(space)

    def workflow_write(
        self,
        name: str,
        when_to_use: str,
        steps: str,
        trigger: str = "",
        *,
        space: str = DEFAULT_SPACE,
    ) -> str:
        return _workflows.write_workflow(name, when_to_use, steps, trigger=trigger, space=space)

    def workflow_find(
        self, query: str, k: int = 3, *, spaces: Sequence[str] | None = None
    ) -> list[Workflow]:
        return _workflows.find_workflows(query, k=k, spaces=_read_spaces(spaces))

    def workflow_count(self, *, space: str = DEFAULT_SPACE) -> int:
        return _workflows.count(space)

    # --- tool-event log -----------------------------------------------------
    # Same call-time delegation as skills/workflows: honours ``events._DB_PATH``
    # reassignment and the daemon's ``_bind_db``. Consolidate mines the log
    # directly (module-level), so it is *not* routed through the client — but the
    # hooks that WRITE events and read one task's events do go through here, so a
    # daemon deployment sees them.
    def log_event(
        self, tool: str, brief: str = "", task_id: str = "", *, space: str = DEFAULT_SPACE
    ) -> int:
        return _events.log_event(tool, brief=brief, task_id=task_id, space=space)

    def events_for_task(self, task_id: str, limit: int = 500) -> list[Event]:
        return _events.for_task(task_id, limit)

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
