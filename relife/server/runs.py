"""Durable outcomes of scheduled runs.

A scheduled run happens while nobody is watching, and the session it ran in
is reaped an hour later — so the session's ring buffer must not be the only
copy of what happened. The scheduler records each run's event stream here as a
:class:`RunRecord`: the agent's closing summary, what it did (tool calls,
cost), and — the part that needs the user — every approval that was **denied
because no one was there to grant it**.

Pure summarization (:func:`summarize_events`) over the ``to_event`` taxonomy,
plus a small per-schedule file store under ``data/runs/<schedule_id>/``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .. import config

# Keep the closing summary readable in a panel; the full text is in the events.
SUMMARY_MAX_CHARS = 2000
# Cap the persisted event list so a runaway turn can't write an unbounded file.
EVENTS_MAX = 2000


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Distil one turn's events into what a user wants to know afterwards.

    ``summary`` is the agent's *closing* text — everything it said after its
    last tool call (the prompt asks it to end with a summary); if it never
    used a tool, all of its text. ``denied`` lists approvals that resolved to
    ``False`` (unattended ⇒ timeout ⇒ deny), each with the tool and the brief
    the user would have seen on the card, so they can do it by hand.
    """
    tool_calls = 0
    cost = None
    error = None
    last_tool_idx = -1
    pending: dict[str, dict[str, Any]] = {}
    denied: list[dict[str, Any]] = []
    for i, ev in enumerate(events):
        t = ev.get("type")
        if t == "tool_use":
            tool_calls += 1
            last_tool_idx = i
        elif t == "tool_result":
            last_tool_idx = i
        elif t == "result":
            cost = ev.get("cost_usd")
        elif t == "error":
            error = ev.get("message")
        elif t == "approval_request":
            pending[ev.get("approval_id", "")] = ev
        elif t == "approval_resolved" and not ev.get("approved"):
            req = pending.get(ev.get("approval_id", ""), {})
            denied.append(
                {
                    "tool": req.get("tool", "?"),
                    "brief": req.get("brief", ""),
                    "reason": req.get("reason", ""),
                }
            )
    closing = [
        ev["text"] for ev in events[last_tool_idx + 1 :] if ev.get("type") == "text" and ev.get("text")
    ]
    summary = "".join(closing).strip()
    if len(summary) > SUMMARY_MAX_CHARS:
        summary = summary[: SUMMARY_MAX_CHARS - 1].rstrip() + "…"
    return {
        "summary": summary,
        "tool_calls": tool_calls,
        "cost_usd": cost,
        "denied": denied,
        "error": error,
    }


@dataclass
class RunRecord:
    schedule_id: str
    run_id: str
    started_at: float
    finished_at: float | None = None
    status: str = "running"      # running | done | error | timeout | interrupted
    summary: str = ""
    tool_calls: int = 0
    cost_usd: float | None = None
    denied: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    session_id: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def new_id(now: float | None = None) -> str:
        t = time.time() if now is None else now
        return time.strftime("%Y%m%d-%H%M%S", time.localtime(t)) + f"-{int((t % 1) * 1000):03d}"

    def finish(self, status: str, events: list[dict[str, Any]], now: float | None = None) -> None:
        self.finished_at = time.time() if now is None else now
        self.status = status
        self.events = events[-EVENTS_MAX:]
        info = summarize_events(self.events)
        self.summary = info["summary"]
        self.tool_calls = info["tool_calls"]
        self.cost_usd = info["cost_usd"]
        self.denied = info["denied"]
        self.error = info["error"]

    def to_dict(self, *, with_events: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not with_events:
            d.pop("events")
            d["event_count"] = len(self.events)
        return d


class RunStore:
    """``<root>/<schedule_id>/<run_id>.json``, newest-last by run id (time-ordered).

    Bounded per schedule: saving a run prunes the oldest beyond
    ``AGENT_RUN_HISTORY``. Removing a schedule removes its runs.
    """

    def __init__(self, root: Path | None = None, *, keep: int | None = None) -> None:
        self.root = Path(root) if root is not None else config.AGENT_RUNS_DIR
        self.keep = config.AGENT_RUN_HISTORY if keep is None else keep

    def _dir(self, schedule_id: str) -> Path:
        return self.root / schedule_id

    def _path(self, schedule_id: str, run_id: str) -> Path:
        return self._dir(schedule_id) / f"{run_id}.json"

    def save(self, record: RunRecord) -> None:
        d = self._dir(record.schedule_id)
        d.mkdir(parents=True, exist_ok=True)
        path = self._path(record.schedule_id, record.run_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(record)), encoding="utf-8")
        os.replace(tmp, path)
        self._prune(record.schedule_id)

    def _prune(self, schedule_id: str) -> None:
        if not self.keep:
            return
        files = sorted(self._dir(schedule_id).glob("*.json"))
        for old in files[: max(0, len(files) - self.keep)]:
            try:
                old.unlink()
            except OSError:  # pragma: no cover - best effort
                pass

    def get(self, schedule_id: str, run_id: str) -> RunRecord | None:
        path = self._path(schedule_id, run_id)
        if not path.exists():
            return None
        try:
            return RunRecord(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None

    def list(self, schedule_id: str, limit: int | None = None) -> list[RunRecord]:
        """Newest first, without the cost of loading events for display."""
        d = self._dir(schedule_id)
        if not d.exists():
            return []
        out = []
        for path in sorted(d.glob("*.json"), reverse=True):
            if limit is not None and len(out) >= limit:
                break
            rec = self.get(schedule_id, path.stem)
            if rec is not None:
                out.append(rec)
        return out

    def remove_all(self, schedule_id: str) -> None:
        d = self._dir(schedule_id)
        if not d.exists():
            return
        for path in d.glob("*.json*"):
            try:
                path.unlink()
            except OSError:  # pragma: no cover
                pass
        try:
            d.rmdir()
        except OSError:  # pragma: no cover
            pass
