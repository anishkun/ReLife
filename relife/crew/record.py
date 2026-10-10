"""Durable outcome of a ``relife crew`` run.

A crew may run for a long time and spend budget across several agents, so what
it planned and what each member delivered is written down as it happens — the
terminal scrollback must not be the only copy. One :class:`CrewRunRecord` per
run at ``data/crews/<id>/record.json``: the task, the validated plan, every
task's outcome (agent, output, tool count, cost, approvals denied), the final
output, and status (``planned`` → ``running`` → ``done`` | ``error`` |
``interrupted``). Atomic writes (tmp + ``os.replace``), ids shaped like
``RunRecord``'s and checked before they name a file.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .. import config

_RUN_ID = re.compile(r"^\d{8}-\d{6}-\d{3}$")

# Under `relife serve` a crew's worker thread rewrites its record while HTTP
# routes read it. On Windows a file being read can't be replaced (and one
# being replaced can't be opened), so either side may see a PermissionError
# for an instant: retry briefly rather than fail a request or lose a save.
_RETRIES = 20
_RETRY_DELAY = 0.025


def _retrying(fn):
    for attempt in range(_RETRIES):
        try:
            return fn()
        except PermissionError:
            if attempt == _RETRIES - 1:
                raise
            time.sleep(_RETRY_DELAY)
    return None  # pragma: no cover


@dataclass
class TaskOutcome:
    name: str
    agent: str
    runtime: str
    output: str = ""
    tool_calls: int = 0
    cost_usd: float | None = None
    error: str | None = None
    denied: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CrewRunRecord:
    id: str
    task: str
    workspace: str
    spec: dict[str, Any]
    status: str = "planned"
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    plan_notes: list[str] = field(default_factory=list)
    plan_cost_usd: float = 0.0
    tasks: list[TaskOutcome] = field(default_factory=list)
    final_output: str = ""
    error: str | None = None

    @staticmethod
    def new_id(now: float | None = None) -> str:
        now = time.time() if now is None else now
        return time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}"

    @property
    def cost_usd(self) -> float:
        return self.plan_cost_usd + sum(t.cost_usd or 0.0 for t in self.tasks)

    def outcome(self, name: str) -> TaskOutcome | None:
        return next((t for t in self.tasks if t.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cost_usd"] = self.cost_usd
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CrewRunRecord:
        d = dict(d)
        d.pop("cost_usd", None)
        d["tasks"] = [TaskOutcome(**t) for t in d.get("tasks", [])]
        return cls(**d)


class CrewRunStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else config.CREWS_DIR

    def save(self, record: CrewRunRecord) -> Path:
        d = self.root / record.id
        d.mkdir(parents=True, exist_ok=True)
        path = d / "record.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        _retrying(lambda: os.replace(tmp, path))
        return path

    def get(self, run_id: str) -> CrewRunRecord | None:
        if not _RUN_ID.match(run_id or ""):
            return None  # never let a caller-supplied id name an arbitrary path
        path = self.root / run_id / "record.json"
        if not path.exists():
            return None
        return CrewRunRecord.from_dict(json.loads(_retrying(lambda: path.read_text(encoding="utf-8"))))

    def list(self, limit: int = 20) -> list[CrewRunRecord]:
        if not self.root.is_dir():
            return []
        out = []
        for d in sorted((p for p in self.root.iterdir() if _RUN_ID.match(p.name)), reverse=True)[:limit]:
            try:
                r = self.get(d.name)
            except (OSError, ValueError, TypeError):
                continue
            if r is not None:
                out.append(r)
        return out
