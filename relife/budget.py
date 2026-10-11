"""What ReLife has spent of the Max budget, and the optional daily cap.

Every model call ReLife starts — a CLI turn, a server session's turn, a crew
member's task, a planner or critic one-shot, a scheduled run — reports a cost
when it ends (the CLI's ``total_cost_usd``; on Max it is *usage-equivalent*,
not billed, but it tracks how fast the subscription's limits are being used).
They all land in one ledger, ``data/spend.db`` (SQLite, so the CLI, the server
and the memory daemon can all write it), so there is one answer to "how much
did ReLife use today?" — shown by ``relife doctor`` and ``GET /budget``.

**The cap is off until set.** With ``RELIFE_DAILY_BUDGET_USD`` (> 0), new work
is refused once today's spend reaches it: a CLI command exits with a message,
the server answers 429 to a typed turn or a crew, a schedule's firing is
recorded as skipped, and a running crew stops at its next task boundary. Work
already in flight is never cut mid-call.

**Costs are cumulative per client.** The CLI fills ``total_cost_usd`` from its
process-wide session total, so a long-lived client (``relife chat``, a server
session) reports the running total after each turn, not that turn's cost. A
:class:`CostMeter` per client turns totals into per-turn spend (a lower total
than last time means a new process — counted from zero).

Never fails a run: a ledger error is swallowed (spend unknown beats work lost).
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from . import config

log = logging.getLogger(__name__)


class BudgetExceeded(Exception):
    """Today's spend has reached ``RELIFE_DAILY_BUDGET_USD``."""


def _today(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(time.time() if now is None else now))


def _conn():
    from .memory._sqlite import connect

    conn = connect(config.SPEND_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS spend ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, day TEXT NOT NULL,"
        " usd REAL NOT NULL, kind TEXT NOT NULL, source TEXT NOT NULL DEFAULT '')"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS spend_day ON spend(day)")
    return conn


def record(usd: Any, kind: str, source: str = "", *, now: float | None = None) -> float:
    """Add ``usd`` (one call's or one turn's cost) to today's spend. Returns
    what was recorded (0 for a missing, negative or non-finite cost)."""
    try:
        v = float(usd)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v) or v <= 0:
        return 0.0
    t = time.time() if now is None else now
    try:
        conn = _conn()
        with conn:
            conn.execute(
                "INSERT INTO spend (at, day, usd, kind, source) VALUES (?, ?, ?, ?, ?)",
                (t, _today(t), v, kind, source[:200]),
            )
    except Exception:  # noqa: BLE001 — never fail the work over bookkeeping
        log.warning("could not record spend", exc_info=True)
        return 0.0
    return v


def spent(day: str | None = None) -> float:
    try:
        row = _conn().execute(
            "SELECT COALESCE(SUM(usd), 0) FROM spend WHERE day = ?", (day or _today(),)
        ).fetchone()
        return float(row[0])
    except Exception:  # noqa: BLE001
        return 0.0


def by_kind(day: str | None = None) -> dict[str, float]:
    try:
        return {
            r[0]: float(r[1])
            for r in _conn().execute(
                "SELECT kind, SUM(usd) FROM spend WHERE day = ? GROUP BY kind ORDER BY 2 DESC",
                (day or _today(),),
            )
        }
    except Exception:  # noqa: BLE001
        return {}


def cap() -> float:
    return config.DAILY_BUDGET_USD if config.DAILY_BUDGET_USD > 0 else 0.0


def exceeded() -> bool:
    c = cap()
    return bool(c) and spent() >= c


def check(what: str = "this") -> None:
    """Raise :class:`BudgetExceeded` if the daily cap is set and reached."""
    c = cap()
    if not c:
        return
    s = spent()
    if s >= c:
        raise BudgetExceeded(
            f"daily budget reached: ${s:.2f} of ${c:.2f} used today, so {what} won't start "
            "(raise RELIFE_DAILY_BUDGET_USD, or wait until tomorrow)"
        )


def status() -> dict[str, Any]:
    c = cap()
    s = spent()
    return {
        "day": _today(),
        "spent_usd": round(s, 4),
        "cap_usd": c or None,
        "remaining_usd": round(max(c - s, 0.0), 4) if c else None,
        "exceeded": bool(c) and s >= c,
        "by_kind": {k: round(v, 4) for k, v in by_kind().items()},
    }


class CostMeter:
    """Per-client: turns the CLI's cumulative ``total_cost_usd`` into what each
    turn added, and records it (see module doc)."""

    def __init__(self, kind: str, source: str = "") -> None:
        self.kind = kind
        self.source = source
        self._last = 0.0

    def observe(self, total: Any) -> float | None:
        """Feed one result's ``total_cost_usd``; return the turn's own cost
        (``None`` when the result carried none)."""
        try:
            t = float(total)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(t) or t < 0:
            return None
        delta = t - self._last if t >= self._last else t  # lower ⇒ a fresh process
        self._last = t
        record(delta, self.kind, self.source)
        return delta
