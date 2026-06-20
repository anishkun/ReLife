"""Durable plan + progress ledger for an orchestrated build.

One ledger per build, stored at ``data/builds/<build_id>/ledger.json`` with a
human-readable ``plan.md`` mirror re-rendered on every write. The JSON is the
source of truth; ``plan.md`` is a derived view the user can eyeball.

The ledger is what makes a huge build **resumable**: if the run dies, the
milestone statuses + the persisted ``session_id`` let the next ``relife build
--resume`` pick up where it left off. Pure and deterministic — no agent calls,
so it's fully unit-testable.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .. import config

STATUSES = ("pending", "in_progress", "done", "failed")


def _slug_id() -> str:
    """A sortable, filesystem-safe build id: ``YYYYmmdd-HHMMSS-xxxx``.

    The random suffix keeps ids unique even when builds are created within the
    same second (the timestamp prefix keeps them roughly chronological).
    """
    return time.strftime("%Y%m%d-%H%M%S", time.localtime()) + "-" + uuid.uuid4().hex[:4]


@dataclass
class Milestone:
    id: int
    title: str
    status: str = "pending"
    summary: str = ""
    notes: str = ""


@dataclass
class BuildLedger:
    build_id: str
    spec: str
    workspace: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    milestones: list[Milestone] = field(default_factory=list)

    # --- locating on disk --------------------------------------------------
    @staticmethod
    def dir_for(build_id: str) -> Path:
        return config.BUILDS_DIR / build_id

    @property
    def dir(self) -> Path:
        return self.dir_for(self.build_id)

    @property
    def json_path(self) -> Path:
        return self.dir / "ledger.json"

    @property
    def md_path(self) -> Path:
        return self.dir / "plan.md"

    # --- construction / IO -------------------------------------------------
    @classmethod
    def create(cls, spec: str, workspace: str | Path) -> "BuildLedger":
        config.BUILDS_DIR.mkdir(parents=True, exist_ok=True)
        led = cls(build_id=_slug_id(), spec=spec, workspace=str(workspace))
        led.dir.mkdir(parents=True, exist_ok=True)
        led.save()
        return led

    @classmethod
    def load(cls, build_id: str) -> "BuildLedger":
        data = json.loads(cls.dir_for(build_id).joinpath("ledger.json").read_text("utf-8"))
        ms = [Milestone(**m) for m in data.pop("milestones", [])]
        return cls(milestones=ms, **data)

    @classmethod
    def latest_for(cls, workspace: str | Path) -> Optional["BuildLedger"]:
        """Most recently updated ledger whose workspace matches (for --resume)."""
        ws = str(Path(workspace).resolve())
        if not config.BUILDS_DIR.exists():
            return None
        candidates: list[BuildLedger] = []
        for d in config.BUILDS_DIR.iterdir():
            jp = d / "ledger.json"
            if not jp.exists():
                continue
            try:
                led = cls.load(d.name)
            except Exception:  # noqa: BLE001 - skip corrupt/partial dirs
                continue
            if str(Path(led.workspace).resolve()) == ws:
                candidates.append(led)
        if not candidates:
            return None
        return max(candidates, key=lambda l: l.updated_at)

    def save(self) -> None:
        self.updated_at = time.time()
        self.dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = asdict(self)
        # atomic-ish write: temp then replace, so a crash mid-write can't corrupt
        tmp = self.json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.json_path)
        self.md_path.write_text(self.render_markdown(), encoding="utf-8")

    # --- mutations ---------------------------------------------------------
    def set_plan(self, titles: list[str]) -> None:
        """Replace the milestone list from an ordered list of titles."""
        self.milestones = [Milestone(id=i + 1, title=t) for i, t in enumerate(titles)]
        self.save()

    def update_milestone(self, mid: int, status: str, summary: str = "") -> None:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
        for m in self.milestones:
            if m.id == mid:
                m.status = status
                if summary:
                    m.summary = summary
                self.save()
                return
        raise KeyError(f"no milestone with id {mid}")

    def set_session_id(self, session_id: str | None) -> None:
        self.session_id = session_id
        self.save()

    # --- queries -----------------------------------------------------------
    def pending(self) -> list[Milestone]:
        return [m for m in self.milestones if m.status in ("pending", "in_progress")]

    def done(self) -> list[Milestone]:
        return [m for m in self.milestones if m.status == "done"]

    def is_complete(self) -> bool:
        return bool(self.milestones) and all(m.status == "done" for m in self.milestones)

    # --- rendering ---------------------------------------------------------
    _ICON = {"pending": "○", "in_progress": "◐", "done": "●", "failed": "✗"}

    def render_markdown(self) -> str:
        lines = [
            f"# Build {self.build_id}",
            "",
            f"**Spec:** {self.spec}",
            f"**Workspace:** {self.workspace}",
            f"**Session:** {self.session_id or '(not yet started)'}",
            "",
            "## Milestones",
            "",
        ]
        if not self.milestones:
            lines.append("_(not yet decomposed)_")
        for m in self.milestones:
            icon = self._ICON.get(m.status, "?")
            lines.append(f"{icon} **{m.id}. {m.title}** — _{m.status}_")
            if m.summary:
                lines.append(f"    - {m.summary}")
        return "\n".join(lines) + "\n"

    def status_brief(self) -> str:
        """Compact text block fed into the orchestrator on resume."""
        if not self.milestones:
            return "No milestones recorded yet."
        return "\n".join(
            f"{m.id}. [{m.status}] {m.title}" + (f" — {m.summary}" if m.summary else "")
            for m in self.milestones
        )
