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
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .. import config

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_EVERY_RE = re.compile(r"^\s*(\d+)\s*([smhd])\s*$")
_AT_RE = re.compile(r"^\s*([01]?\d|2[0-3]):([0-5]\d)\s*$")
_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

MAX_NAME_CHARS = 80


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

    @classmethod
    def new(
        cls,
        *,
        name: str,
        task: str,
        spec: dict[str, Any],
        workspace: str = "",
        enabled: bool = True,
        now: float | None = None,
    ) -> "Schedule":
        t = time.time() if now is None else now
        name = validate_name(name)
        task = validate_task(task)
        spec = normalize_spec(spec)
        return cls(
            id=uuid.uuid4().hex[:12],
            name=name,
            task=task,
            spec=spec,
            workspace=str(workspace or ""),
            enabled=bool(enabled),
            created_at=t,
            next_run_at=next_run(spec, t),
        )

    def is_due(self, now: float) -> bool:
        return self.enabled and self.next_run_at <= now

    def record_run(self, now: float, status: str, *, keep: int | None = None) -> None:
        """Note an attempt (fired, skipped, failed) and advance to the next slot."""
        limit = config.AGENT_SCHEDULE_HISTORY if keep is None else keep
        self.last_run_at = now
        self.last_status = status
        self.runs.append({"at": now, "status": status})
        del self.runs[:-limit]
        self.next_run_at = next_run(self.spec, now)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["spec_text"] = describe_spec(self.spec)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Schedule":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


def validate_name(name: Any) -> str:
    s = str(name or "").strip()
    if not s:
        raise ValueError("name is required")
    if len(s) > MAX_NAME_CHARS:
        raise ValueError(f"name exceeds {MAX_NAME_CHARS} characters")
    return s


def validate_task(task: Any) -> str:
    s = str(task or "").strip()
    if not s:
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
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else config.AGENT_SCHEDULES_PATH
        self._items: dict[str, Schedule] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for d in raw.get("schedules", []):
            try:
                s = Schedule.from_dict(d)
            except TypeError:
                continue
            self._items[s.id] = s

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
