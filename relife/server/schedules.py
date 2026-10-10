"""Schedules: durable, model-free records of "run this task on this cadence".

Pure model + a small JSON-file store (same discipline as ``build/ledger.py``):
no framework, no model, no event loop. Everything time-related takes an explicit
``now`` (epoch seconds) so the cadence logic is unit-testable at any instant.

A schedule's *spec* is one of two shapes, kept in the user-facing form:

  {"every": "30m"}                       – interval: N s|m|h|d (floor: config)
  {"at": "09:00", "days": ["mon","fri"]} – daily at a local wall-clock time,
                                           optionally only on those weekdays

The scheduler (``scheduler.py``) fires a schedule when ``next_run_at <= now``
and then advances it from *now* — so a slot missed while the server was down
fires **once** on the next tick, never N times.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .. import config
from ..permissions import describe_grant, normalize_grants

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_EVERY_RE = re.compile(r"^\s*(\d+)\s*([smhd])\s*$")
_AT_RE = re.compile(r"^\s*([01]?\d|2[0-3]):([0-5]\d)\s*$")
_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

MAX_NAME_CHARS = 80
MAX_WORKED = 200  # issue refs a work schedule remembers having attempted
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


# --- spec -------------------------------------------------------------------
def parse_every(raw: Any) -> float:
    """``"30m"`` → seconds. Accepts a bare number of seconds too."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        seconds = float(raw)
    else:
        m = _EVERY_RE.match(str(raw or ""))
        if not m:
            raise ValueError('every must look like "30m", "2h" or "1d"')
        seconds = int(m.group(1)) * _UNITS[m.group(2)]
    if seconds <= 0:
        raise ValueError("every must be positive")
    return seconds


def parse_at(raw: Any) -> tuple[int, int]:
    m = _AT_RE.match(str(raw or ""))
    if not m:
        raise ValueError('at must be a 24h wall-clock time like "09:00"')
    return int(m.group(1)), int(m.group(2))


def parse_days(raw: Any) -> list[int]:
    """``["mon","fri"]`` → ``[0, 4]`` (Monday = 0). Empty = every day."""
    if raw in (None, "", []):
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("days must be a list of weekday names")
    out: list[int] = []
    for d in raw:
        key = str(d).strip().lower()[:3]
        if key not in _DAYS:
            raise ValueError(f"unknown weekday {d!r}")
        if _DAYS.index(key) not in out:
            out.append(_DAYS.index(key))
    return sorted(out)


def normalize_spec(spec: Any, *, min_interval: float | None = None) -> dict[str, Any]:
    """Validate a spec and return its canonical form (a plain dict).

    ``min_interval`` floors the interval form: every scheduled run spends Max
    budget, so a "1m" schedule is almost certainly a mistake, not a wish.
    """
    if not isinstance(spec, dict):
        raise ValueError("spec must be an object with 'every' or 'at'")
    has_every = spec.get("every") not in (None, "")
    has_at = spec.get("at") not in (None, "")
    if has_every == has_at:
        raise ValueError("spec needs exactly one of 'every' or 'at'")
    floor = config.AGENT_SCHEDULE_MIN_INTERVAL if min_interval is None else min_interval
    if has_every:
        seconds = parse_every(spec["every"])
        if seconds < floor:
            raise ValueError(f"every must be at least {int(floor)}s (each run spends budget)")
        return {"every": _fmt_seconds(seconds)}
    hh, mm = parse_at(spec["at"])
    out: dict[str, Any] = {"at": f"{hh:02d}:{mm:02d}"}
    days = parse_days(spec.get("days"))
    if days:
        out["days"] = [_DAYS[i] for i in days]
    return out


def _fmt_seconds(seconds: float) -> str:
    s = int(seconds)
    for unit in "dhm":
        if s % _UNITS[unit] == 0:
            return f"{s // _UNITS[unit]}{unit}"
    return f"{s}s"


def describe_spec(spec: dict[str, Any]) -> str:
    if "every" in spec:
        return f"every {spec['every']}"
    days = spec.get("days")
    when = f"daily at {spec['at']}"
    return f"{when} ({', '.join(days)})" if days else when


def next_run(spec: dict[str, Any], now: float) -> float:
    """Epoch seconds of the first firing strictly after ``now``.

    The ``at`` form works in naive *local* wall-clock time and converts back
    with ``timestamp()`` (mktime), so "09:00" stays 09:00 across a DST change
    instead of drifting by the offset.
    """
    if "every" in spec:
        return now + parse_every(spec["every"])
    hh, mm = parse_at(spec["at"])
    days = parse_days(spec.get("days"))
    local = datetime.fromtimestamp(now)
    base = local.replace(hour=hh, minute=mm, second=0, microsecond=0)
    for i in range(0, 8):
        candidate = base + timedelta(days=i)
        if candidate <= local:
            continue
        if days and candidate.weekday() not in days:
            continue
        return candidate.timestamp()
    raise ValueError("no next run within a week")  # pragma: no cover - days non-empty guarantees one


# --- work items -------------------------------------------------------------
def normalize_work(raw: Any) -> dict[str, Any] | None:
    """A work schedule's source of issues, canonical form, or None.

    ``{}`` = any open issue assigned to the user; ``{"repo": "o/r"}`` narrows
    it to one repo, ``{"label": "relife"}`` to issues carrying that label —
    the user's way to say *which* of their issues an unattended agent may take.
    """
    if raw in (None, False, ""):
        return None
    if raw is True:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("work must be an object like {\"repo\": \"owner/name\"}")
    out: dict[str, Any] = {}
    repo = str(raw.get("repo") or "").strip()
    if repo:
        if not _REPO_RE.match(repo):
            raise ValueError("work.repo must look like owner/name")
        out["repo"] = repo
    label = str(raw.get("label") or "").strip()
    if label:
        if len(label) > 50:
            raise ValueError("work.label is too long")
        out["label"] = label
    return out


def check_grants_fit(
    grants: list[dict[str, Any]], work: dict[str, Any] | None, crew: dict[str, Any] | None = None
) -> None:
    """A ``pull_request`` grant only means something on a work schedule (the
    scheduler binds it to the issue's repo + branch); elsewhere it's refused
    rather than silently kept as a grant that can never apply. A crew schedule
    takes no grants at all: they pre-approve one agent's turn, and a crew's
    members each ask (unattended, a timeout denies)."""
    if crew is not None:
        if grants:
            raise ValueError("a crew schedule takes no pre-approvals; its members ask")
        if work is not None:
            raise ValueError("a schedule runs either a crew or work items, not both")
    if work is None and any(g.get("kind") == "pull_request" for g in grants):
        raise ValueError("a pull_request grant needs a work schedule")


def normalize_crew(raw: Any, *, agents: set[str] | frozenset[str] = frozenset()) -> dict[str, Any] | None:
    """A crew schedule's plan, validated like any untrusted crew spec (only the
    configured non-Claude models; ``agents`` = registered names it may reuse or
    inherit from), as a plain dict — or None."""
    if raw in (None, False, ""):
        return None
    from ..crew.spec import normalize_spec as normalize_crew_spec

    return normalize_crew_spec(raw, existing_agents=set(agents), allowed_llms=config.CREW_LLMS).to_dict()


def describe_crew(crew: dict[str, Any]) -> str:
    names = ", ".join(a.get("name", "?") for a in crew.get("agents", []))
    n = len(crew.get("tasks", []))
    return f"runs a crew ({names}) · {n} task{'' if n == 1 else 's'}"


def describe_work(work: dict[str, Any]) -> str:
    where = f" in {work['repo']}" if work.get("repo") else ""
    label = f" labelled {work['label']}" if work.get("label") else ""
    return f"works your assigned issues{where}{label} → branch + PR"


# --- record -----------------------------------------------------------------
@dataclass
class Schedule:
    id: str
    name: str
    task: str
    spec: dict[str, Any]
    workspace: str            # as requested (relative to the workspace root, or absolute inside it)
    enabled: bool = True
    created_at: float = 0.0
    next_run_at: float = 0.0
    last_run_at: float | None = None
    last_status: str | None = None
    session_id: str | None = None
    runs: list[dict[str, Any]] = field(default_factory=list)  # newest last, bounded
    # Pre-authorized outward actions for this schedule's runs (see
    # ``permissions.grant_allows``) — e.g. [{"kind": "email", "addresses": [me]}].
    grants: list[dict[str, Any]] = field(default_factory=list)
    # A *work schedule* (``work`` set): each firing takes the next assigned
    # GitHub issue not in ``worked`` and runs the `relife work` flow on it;
    # ``task`` becomes optional extra instructions.
    work: dict[str, Any] | None = None
    worked: list[str] = field(default_factory=list)  # issue refs attempted, newest last
    # A *crew schedule* (``crew`` set, a validated crew spec): each firing runs
    # that crew under the server (``server/crews.py``); ``task`` is optional.
    crew: dict[str, Any] | None = None

    @classmethod
    def new(
        cls,
        *,
        name: str,
        task: str,
        spec: dict[str, Any],
        workspace: str = "",
        enabled: bool = True,
        grants: list[dict[str, Any]] | None = None,
        work: Any = None,
        crew: Any = None,
        agents: set[str] | frozenset[str] = frozenset(),
        now: float | None = None,
    ) -> "Schedule":
        t = time.time() if now is None else now
        name = validate_name(name)
        work = normalize_work(work)
        crew = normalize_crew(crew, agents=agents)
        task = validate_task(task, required=work is None and crew is None)
        spec = normalize_spec(spec)
        grants = normalize_grants(grants)
        check_grants_fit(grants, work, crew)
        return cls(
            id=uuid.uuid4().hex[:12],
            name=name,
            task=task,
            spec=spec,
            workspace=str(workspace or ""),
            enabled=bool(enabled),
            created_at=t,
            next_run_at=next_run(spec, t),
            grants=grants,
            work=work,
            crew=crew,
        )

    def mark_worked(self, ref: str) -> None:
        if ref not in self.worked:
            self.worked.append(ref)
            del self.worked[:-MAX_WORKED]

    def is_due(self, now: float) -> bool:
        return self.enabled and self.next_run_at <= now

    def record_run(
        self, now: float, status: str, *, run_id: str | None = None, keep: int | None = None
    ) -> None:
        """Note an attempt (fired, skipped, failed) and advance to the next slot.

        ``run_id`` links the entry to its durable :class:`~.runs.RunRecord`; the
        recorder later upgrades the entry's ``status`` from ``submitted`` to the
        real outcome via :meth:`update_run`.
        """
        limit = config.AGENT_SCHEDULE_HISTORY if keep is None else keep
        self.last_run_at = now
        self.last_status = status
        entry: dict[str, Any] = {"at": now, "status": status}
        if run_id:
            entry["run_id"] = run_id
        self.runs.append(entry)
        del self.runs[:-limit]
        self.next_run_at = next_run(self.spec, now)

    def update_run(self, run_id: str, **fields: Any) -> bool:
        """Patch the history entry for ``run_id`` (and ``last_status`` if it's
        the latest). False if the entry has already aged out of the history."""
        for entry in reversed(self.runs):
            if entry.get("run_id") == run_id:
                entry.update(fields)
                if entry is self.runs[-1] and "status" in fields:
                    self.last_status = fields["status"]
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["spec_text"] = describe_spec(self.spec)
        d["grants_text"] = [describe_grant(g) for g in self.grants]
        d["work_text"] = describe_work(self.work) if self.work is not None else None
        d["crew_text"] = describe_crew(self.crew) if self.crew is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Schedule":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        try:
            known["grants"] = normalize_grants(known.get("grants"))
        except ValueError:
            known["grants"] = []  # a hand-edited, invalid grant is dropped, never widened
        try:
            known["work"] = normalize_work(known.get("work"))
        except ValueError:
            known["work"] = None
            known["enabled"] = False  # half a work schedule must not fire as a blank task
        crew = known.get("crew")
        if crew is not None and not (
            isinstance(crew, dict) and isinstance(crew.get("agents"), list) and isinstance(crew.get("tasks"), list)
        ):
            # Same rule: a broken crew must not fire as a blank task. (The full
            # check, against the live agent registry, runs when it fires.)
            known["crew"] = None
            known["enabled"] = False
        return cls(**known)


def validate_name(name: Any) -> str:
    s = str(name or "").strip()
    if not s:
        raise ValueError("name is required")
    if len(s) > MAX_NAME_CHARS:
        raise ValueError(f"name exceeds {MAX_NAME_CHARS} characters")
    return s


def validate_task(task: Any, *, required: bool = True) -> str:
    s = str(task or "").strip()
    if not s and required:
        raise ValueError("task is required")
    if len(s) > config.AGENT_MAX_MESSAGE_CHARS:
        raise ValueError(f"task exceeds {config.AGENT_MAX_MESSAGE_CHARS} characters")
    return s


# --- store ------------------------------------------------------------------
class ScheduleStore:
    """All schedules for one server, persisted as a single JSON file.

    Loaded once at construction (a missing file = no schedules; nothing is
    written until the first change, so building the app stays side-effect
    free). Every mutation rewrites the file atomically (tmp + ``os.replace``),
    so a crash mid-write can't leave a torn file behind.

    A file that can't be read (or holds records that can't be) is never
    silently overwritten: ``problem`` says what went wrong (``relife doctor``
    shows it), and the first save copies the original aside to
    ``schedules.json.corrupt-<time>`` before replacing it.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else config.AGENT_SCHEDULES_PATH
        self._items: dict[str, Schedule] = {}
        self.problem: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            self.problem = f"unreadable ({type(e).__name__}: {e})"
            return
        records = raw.get("schedules", []) if isinstance(raw, dict) else None
        if not isinstance(records, list):
            self.problem = "unexpected format (no schedules list)"
            return
        dropped = 0
        for d in records:
            try:
                s = Schedule.from_dict(d)
            except (TypeError, ValueError, AttributeError, KeyError):
                dropped += 1
                continue
            self._items[s.id] = s
        if dropped:
            self.problem = f"{dropped} unreadable schedule record(s) skipped"

    def _preserve_original(self) -> None:
        """Copy a file we couldn't fully read aside, once, before overwriting it."""
        if self.problem is None or not self.path.exists():
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
        try:
            shutil.copy2(self.path, backup)
        except OSError:
            return  # best effort; still better than refusing to save
        self.problem = f"{self.problem} — original kept as {backup.name}"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.problem and "original kept as" not in self.problem:
            self._preserve_original()
        payload = {"version": 1, "schedules": [asdict(s) for s in self._items.values()]}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def list(self) -> list[Schedule]:
        return sorted(self._items.values(), key=lambda s: s.created_at)

    def get(self, schedule_id: str) -> Schedule | None:
        return self._items.get(schedule_id)

    def count(self) -> int:
        return len(self._items)

    def add(self, schedule: Schedule) -> Schedule:
        self._items[schedule.id] = schedule
        self.save()
        return schedule

    def remove(self, schedule_id: str) -> bool:
        if self._items.pop(schedule_id, None) is None:
            return False
        self.save()
        return True

    def due(self, now: float) -> list[Schedule]:
        return [s for s in self.list() if s.is_due(now)]
