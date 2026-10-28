"""What each model has been asked for lately, and the limits the user declared.

The policy (``llm_limits.py``) is pure; this is the state it reads:

- **The usage ledger** — ``data/llm_usage.db`` (SQLite, like the spend ledger in
  ``budget.py``) — one row per model call and what the provider told us
  (``learned``: the AIMD factor, a cooldown, an exhausted-until). One file, so
  the CLI, the server and a crew running on a worker thread share one view of a
  free key's quota: two crews on the same model are paced together, not each
  as if it were alone. Rows older than two days are pruned as new ones land.
- **The limits file** — ``data/llm_limits.json`` (``relife llm set``) — with
  the ``AgentStore`` discipline: nothing written until the first change,
  atomic writes, unreadable records dropped (never widened to "unlimited") and
  the original kept aside before the first overwrite.

Never fails a run: a ledger error is logged and swallowed, and pacing then
proceeds as if nothing had been called (unknown usage beats work lost).
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import _atomic, config
from ._filelock import file_lock
from .llm_limits import (
    DAY,
    MINUTE,
    Learned,
    ModelLimits,
    Usage,
    adapt,
    decide,
    effective_factor,
    limits_from_dict,
    match_limits,
    normalize_limits,
    tightness,
)

log = logging.getLogger(__name__)

KEEP = 2 * DAY  # usage rows kept (the policy looks back one rolling day)


def _conn():
    from .memory._sqlite import connect

    conn = connect(config.LLM_USAGE_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS calls ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, model TEXT NOT NULL,"
        " tokens INTEGER NOT NULL DEFAULT 0, outcome TEXT NOT NULL DEFAULT 'ok')"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS calls_model_at ON calls(model, at)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS learned ("
        " model TEXT PRIMARY KEY, factor REAL NOT NULL, cooldown_until REAL NOT NULL,"
        " exhausted_until REAL NOT NULL, updated_at REAL NOT NULL)"
    )
    return conn


def record_call(model: str, tokens: int = 0, outcome: str = "ok", *, now: float | None = None) -> None:
    """One request sent to ``model`` (a refused one counts too: the provider
    counted it against the same window)."""
    t = time.time() if now is None else now
    try:
        conn = _conn()
        with conn:
            conn.execute(
                "INSERT INTO calls (at, model, tokens, outcome) VALUES (?, ?, ?, ?)",
                (t, model, max(0, int(tokens)), outcome[:20]),
            )
            conn.execute("DELETE FROM calls WHERE at < ?", (t - KEEP,))
    except Exception:  # noqa: BLE001 — bookkeeping never fails a call
        log.warning("could not record a model call", exc_info=True)


def usage(model: str, *, now: float | None = None, day: bool = False) -> Usage:
    """The calls in the last minute (and the last rolling day, when ``day``)."""
    t = time.time() if now is None else now
    try:
        conn = _conn()
        minute = tuple(
            (float(a), int(k))
            for a, k in conn.execute(
                "SELECT at, tokens FROM calls WHERE model = ? AND at > ? ORDER BY at",
                (model, t - MINUTE),
            )
        )
        days: tuple[tuple[float, int], ...] = ()
        if day:
            days = tuple(
                (float(a), int(k))
                for a, k in conn.execute(
                    "SELECT at, tokens FROM calls WHERE model = ? AND at > ? ORDER BY at",
                    (model, t - DAY),
                )
            )
        return Usage(minute=minute, day=days)
    except Exception:  # noqa: BLE001
        log.warning("could not read model usage", exc_info=True)
        return Usage()


def learned(model: str) -> Learned:
    try:
        row = _conn().execute(
            "SELECT factor, cooldown_until, exhausted_until, updated_at FROM learned WHERE model = ?",
            (model,),
        ).fetchone()
    except Exception:  # noqa: BLE001
        log.warning("could not read learned limits", exc_info=True)
        return Learned()
    if row is None:
        return Learned()
    return Learned(factor=float(row[0]), cooldown_until=float(row[1]),
                   exhausted_until=float(row[2]), updated_at=float(row[3]))


def _put_learned(model: str, value: Learned) -> None:
    conn = _conn()
    with conn:
        conn.execute(
            "INSERT INTO learned (model, factor, cooldown_until, exhausted_until, updated_at)"
            " VALUES (?, ?, ?, ?, ?) ON CONFLICT(model) DO UPDATE SET factor=excluded.factor,"
            " cooldown_until=excluded.cooldown_until, exhausted_until=excluded.exhausted_until,"
            " updated_at=excluded.updated_at",
            (model, value.factor, value.cooldown_until, value.exhausted_until, value.updated_at),
        )


def record_error(
    model: str, kind: str, retry_after: float | None = None, *, now: float | None = None
) -> Learned:
    """Fold a provider refusal (``throttle``/``quota``) into what is learned
    about ``model``; the refused request is counted as a call too."""
    t = time.time() if now is None else now
    value = adapt(learned(model), kind, t, retry_after)
    record_call(model, 0, kind, now=t)
    try:
        _put_learned(model, value)
    except Exception:  # noqa: BLE001
        log.warning("could not record learned limits", exc_info=True)
    return value


def mark_exhausted(model: str, until: float, *, now: float | None = None) -> None:
    """The policy found the declared daily limit used up (no provider error
    needed): remember it so every process stops asking until ``until``."""
    t = time.time() if now is None else now
    try:
        cur = learned(model)
        _put_learned(model, replace(cur, exhausted_until=max(cur.exhausted_until, until), updated_at=t))
    except Exception:  # noqa: BLE001
        log.warning("could not record an exhausted model", exc_info=True)


def reset(model: str) -> None:
    """Forget what was learned about ``model`` (``relife llm reset``)."""
    try:
        conn = _conn()
        with conn:
            conn.execute("DELETE FROM learned WHERE model = ?", (model,))
    except Exception:  # noqa: BLE001
        log.warning("could not reset learned limits", exc_info=True)


def chain_blocked(
    model: str, store: LimitsStore | None = None, *, now: float | None = None
) -> tuple[float, str] | None:
    """``(reset_at, reason)`` when ``model`` *and* every model in its fallback
    chain are out of quota right now (the soonest reset), else ``None``. What a
    resume checks before it starts anything."""
    t = time.time() if now is None else now
    store = store or LimitsStore()
    first = store.match(model)
    soonest: tuple[float, str] | None = None
    for m in [model, *first.fallback]:
        lim = store.match(m)
        d = decide(lim, usage(m, now=t, day=bool(lim.rpd or lim.tpd)), learned(m), t)
        if d.action != "exhausted":
            return None
        if soonest is None or d.reset_at < soonest[0]:
            soonest = (d.reset_at, d.reason)
    return soonest


def model_status(limits: ModelLimits, *, now: float | None = None) -> dict[str, Any]:
    """One model's limits, what was learned, and recent usage — for
    ``relife llm list``, ``relife doctor`` and ``GET /llms``."""
    t = time.time() if now is None else now
    u = usage(limits.model, now=t, day=True)
    lr = learned(limits.model)
    if lr.exhausted_until > t:
        state, until = "exhausted", lr.exhausted_until
    elif lr.cooldown_until > t:
        state, until = "cooling", lr.cooldown_until
    else:
        state, until = "ok", None
    return {
        "model": limits.model,
        "declared": limits.declared,
        "limits": limits.to_dict(),
        "tightness": tightness(limits),
        "state": state,
        "until": until,
        "factor": round(effective_factor(lr, t), 3),
        "calls_last_minute": len(u.minute),
        "calls_last_day": len(u.day),
        "tokens_last_day": sum(k for _, k in u.day),
    }


# --- the declared limits ---------------------------------------------------
_WRITE_LOCK = threading.RLock()


class LimitsStore:
    """The user's declared limits, one JSON file (see module doc)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else config.LLM_LIMITS_PATH
        self._items: dict[str, ModelLimits] = {}
        self.problem: str | None = None
        with _WRITE_LOCK:
            self._items, self.problem = self._read()

    def _read(self) -> tuple[dict[str, ModelLimits], str | None]:
        if not self.path.exists():
            return {}, None
        try:
            items, dropped = normalize_limits(json.loads(_atomic.read_text(self.path)))
        except (OSError, ValueError) as e:
            return {}, f"unreadable ({type(e).__name__}: {e})"
        return items, (f"{dropped} unreadable model limit record(s) skipped" if dropped else None)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.problem and "original kept as" not in self.problem and self.path.exists():
            backup = self.path.with_name(f"{self.path.name}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}")
            try:
                shutil.copy2(self.path, backup)
                self.problem = f"{self.problem} — original kept as {backup.name}"
            except OSError:
                pass
        payload = {"version": 1, "models": {m: lim.to_dict() for m, lim in sorted(self._items.items())}}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _atomic.replace(tmp, self.path)

    def declared(self) -> dict[str, ModelLimits]:
        return dict(self._items)

    def match(self, model: str) -> ModelLimits:
        return match_limits(model, self._items)

    def set(self, model: str, values: dict[str, Any]) -> ModelLimits:
        """Declare (replace) ``model``'s limits from untrusted ``values``."""
        lim = limits_from_dict(model, values)
        with _WRITE_LOCK, file_lock(self.path):
            items, problem = self._read()
            if problem is None:
                self._items = items
            elif self.problem is None:
                self.problem = problem
            self._items[lim.model] = lim
            self._save()
        return lim

    def remove(self, model: str) -> bool:
        with _WRITE_LOCK, file_lock(self.path):
            items, problem = self._read()
            if problem is None:
                self._items = items
            if self._items.pop(model, None) is None:
                return False
            self._save()
        return True
