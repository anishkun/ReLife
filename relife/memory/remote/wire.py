"""Wire format for the memory daemon — the single source of truth for how a
``Memory`` (and the upkeep reports) cross the HTTP seam.

Both sides import these helpers so the daemon and the ``HttpMemoryClient`` can
never drift. Deliberately dependency-free (no fastapi/httpx): the wire format is
just plain dicts, so this module is import-safe even without the ``[daemon]``
extra installed.

Every field of the ``Memory`` dataclass round-trips losslessly (see the contract
test in ``tests/test_memory_remote.py``); ``activation()`` is derived, not
serialized. Reports serialize their scalar counters + list fields so the client
can rebuild the identical dataclass and consumers see the same ``.summary()`` /
``.workflows_created`` / ``.patterns`` / ``.notes`` they get in-process.
"""

from __future__ import annotations

from typing import Any

from ..consolidate import ConsolidationReport
from ..events import Event
from ..rem import RemReport
from ..skills import Skill
from ..store import Memory
from ..workflows import Workflow

# --- Memory <-> dict --------------------------------------------------------
_MEMORY_FIELDS = (
    "id",
    "kind",
    "text",
    "tags",
    "created_at",
    "importance",
    "last_used_at",
    "use_count",
    "status",
)


def memory_to_dict(m: Memory) -> dict[str, Any]:
    return {f: getattr(m, f) for f in _MEMORY_FIELDS}


def memory_from_dict(d: dict[str, Any]) -> Memory:
    return Memory(**{f: d[f] for f in _MEMORY_FIELDS})


def memory_or_none_to_dict(m: Memory | None) -> dict[str, Any] | None:
    return memory_to_dict(m) if m is not None else None


def memory_or_none_from_dict(d: dict[str, Any] | None) -> Memory | None:
    return memory_from_dict(d) if d is not None else None


def memories_to_list(ms: list[Memory]) -> list[dict[str, Any]]:
    return [memory_to_dict(m) for m in ms]


def memories_from_list(items: list[dict[str, Any]]) -> list[Memory]:
    return [memory_from_dict(d) for d in items]


# --- Skill / Workflow <-> dict ----------------------------------------------
# Every field is a string, so JSON (UTF-8) round-trips them losslessly,
# including non-ASCII bodies. Slugs are ASCII-only by construction.
_SKILL_FIELDS = ("name", "when_to_use", "body", "slug")
_WORKFLOW_FIELDS = ("name", "when_to_use", "trigger", "body", "slug")


def skill_to_dict(s: Skill) -> dict[str, Any]:
    return {f: getattr(s, f) for f in _SKILL_FIELDS}


def skill_from_dict(d: dict[str, Any]) -> Skill:
    return Skill(**{f: d[f] for f in _SKILL_FIELDS})


def skills_to_list(ss: list[Skill]) -> list[dict[str, Any]]:
    return [skill_to_dict(s) for s in ss]


def skills_from_list(items: list[dict[str, Any]]) -> list[Skill]:
    return [skill_from_dict(d) for d in items]


def workflow_to_dict(w: Workflow) -> dict[str, Any]:
    return {f: getattr(w, f) for f in _WORKFLOW_FIELDS}


def workflow_from_dict(d: dict[str, Any]) -> Workflow:
    return Workflow(**{f: d[f] for f in _WORKFLOW_FIELDS})


def workflows_to_list(ws: list[Workflow]) -> list[dict[str, Any]]:
    return [workflow_to_dict(w) for w in ws]


def workflows_from_list(items: list[dict[str, Any]]) -> list[Workflow]:
    return [workflow_from_dict(d) for d in items]


# --- Event <-> dict ---------------------------------------------------------
_EVENT_FIELDS = ("id", "task_id", "tool", "brief", "created_at")


def event_to_dict(e: Event) -> dict[str, Any]:
    return {f: getattr(e, f) for f in _EVENT_FIELDS}


def event_from_dict(d: dict[str, Any]) -> Event:
    return Event(**{f: d[f] for f in _EVENT_FIELDS})


def events_to_list(es: list[Event]) -> list[dict[str, Any]]:
    return [event_to_dict(e) for e in es]


def events_from_list(items: list[dict[str, Any]]) -> list[Event]:
    return [event_from_dict(d) for d in items]


# --- ConsolidationReport <-> dict -------------------------------------------
def consolidation_to_dict(r: ConsolidationReport) -> dict[str, Any]:
    return {
        "archived": r.archived,
        "deleted": r.deleted,
        "merged": r.merged,
        "patterns": list(r.patterns),
        "workflows_created": list(r.workflows_created),
    }


def consolidation_from_dict(d: dict[str, Any]) -> ConsolidationReport:
    return ConsolidationReport(
        archived=d.get("archived", 0),
        deleted=d.get("deleted", 0),
        merged=d.get("merged", 0),
        patterns=list(d.get("patterns", [])),
        workflows_created=list(d.get("workflows_created", [])),
    )


# --- RemReport <-> dict -----------------------------------------------------
def rem_to_dict(r: RemReport) -> dict[str, Any]:
    return {
        "reviewed": r.reviewed,
        "pruned": r.pruned,
        "reweighted": r.reweighted,
        "contradictions": r.contradictions,
        "skipped_low_conf": r.skipped_low_conf,
        "skipped_cap": r.skipped_cap,
        "cost_usd": r.cost_usd,
        "notes": list(r.notes),
    }


def rem_from_dict(d: dict[str, Any]) -> RemReport:
    return RemReport(
        reviewed=d.get("reviewed", 0),
        pruned=d.get("pruned", 0),
        reweighted=d.get("reweighted", 0),
        contradictions=d.get("contradictions", 0),
        skipped_low_conf=d.get("skipped_low_conf", 0),
        skipped_cap=d.get("skipped_cap", 0),
        cost_usd=d.get("cost_usd"),
        notes=list(d.get("notes", [])),
    )
