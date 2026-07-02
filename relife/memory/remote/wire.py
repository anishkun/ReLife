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
from ..rem import RemReport
from ..store import Memory

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
