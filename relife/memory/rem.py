"""REM ("dream") pass — the OPT-IN, LLM-driven deep review of memory.

Where ``consolidate.py`` is the cheap, deterministic, LLM-free "sleep" pass that
auto-runs after tasks, REM is the **opt-in** pass the user triggers (``relife
dream``) when Max budget is comfortable. It asks the model to act as an
*adversarial critic* over recent memories — the qualitative judgement the
deterministic pass cannot do:

- **contradictions** — a new memory that conflicts with established knowledge;
- **safety / alignment** — memories that would push the agent toward unsafe action;
- **hallucinated / garbage** — incoherent or clearly-wrong memories;
- **mis-weighted importance** — over- or under-rated salience.

Crucially, the model is an **advisor only**. Its verdicts are applied here,
deterministically and *reversibly*:

- the only destructive action is **archive** (recoverable), never ``delete``;
- the critic may **not** edit stored text — only ``prune`` (archive) or ``reweight``
  (importance). (Decided scope.)
- every verdict is **confidence-gated** (``REM_MIN_CONFIDENCE``);
- a **safety cap** (``REM_MAX_PRUNE_FRACTION``) bounds how much one pass may archive;
- every applied action is **journaled** to ``data/rem_journal.jsonl`` so a bad
  prune is auditable and recoverable.

So even a misbehaving critic cannot corrupt memory. This module is kept out of
``consolidate.run_consolidation()`` on purpose: that pass must stay LLM-free.

The model call is **injected** (``ask_model``) so this module stays SDK-free and
the application logic is fully unit-testable with a stub that returns canned JSON.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .. import config
from . import store

# (system_prompt, user_prompt) -> (text, cost_usd) | text
AskModel = Callable[[str, str], Awaitable]

_STATE_PATH = config.DATA_DIR / "rem_state.json"
_JOURNAL_PATH = config.DATA_DIR / "rem_journal.jsonl"


@dataclass
class RemReport:
    reviewed: int = 0
    pruned: int = 0
    reweighted: int = 0
    contradictions: int = 0
    skipped_low_conf: int = 0
    skipped_cap: int = 0
    cost_usd: float | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        cost = f", cost ${self.cost_usd:.4f}" if self.cost_usd else ""
        return (
            f"reviewed {self.reviewed}, pruned {self.pruned}, "
            f"reweighted {self.reweighted}, contradictions {self.contradictions}, "
            f"skipped {self.skipped_low_conf} (low-confidence) / "
            f"{self.skipped_cap} (over cap){cost}"
        )


# --- watermark state (so each pass reviews only what's new since last time) ---
def _read_state() -> dict:
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(state: dict) -> None:
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
    except Exception:
        pass


def _journal(entry: dict) -> None:
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with _JOURNAL_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# --- candidate selection -----------------------------------------------------
def _select_buffer(batch_max: int) -> list:
    """The "experience replay buffer": active memories new since the last pass,
    most salient first. Importance is the dopamine/surprise proxy — surprising
    saves carry high importance. Falls back to most-recent if nothing is new."""
    watermark = int(_read_state().get("last_reviewed_id", 0))
    active = store.all_memories(include_archived=False)
    fresh = [m for m in active if m.id > watermark]
    pool = fresh or active  # nothing new → re-review the most recent instead
    pool.sort(key=lambda m: (m.importance, m.id), reverse=True)
    if not fresh:
        # Fallback: most recent by id (creation order) rather than by importance.
        pool = sorted(active, key=lambda m: m.id, reverse=True)
    return pool[:batch_max]


def _reference_set(exclude_ids: set[int]) -> list:
    """Established knowledge the critic checks contradictions against:
    preferences, learned patterns, and pinned (high-importance) memories."""
    active = store.all_memories(include_archived=False)
    ref = [
        m
        for m in active
        if m.id not in exclude_ids
        and (m.kind in ("preference", "pattern") or m.importance >= config.PIN_THRESHOLD)
    ]
    ref.sort(key=lambda m: (m.importance, m.id), reverse=True)
    return ref[: config.REM_REFERENCE_MAX]


# --- critic prompt + parsing --------------------------------------------------
def _fmt(m) -> str:
    tags = f" [tags: {m.tags}]" if m.tags else ""
    return f"  #{m.id} ({m.kind}, importance {m.importance:.2f}){tags}: {m.text}"


def _build_prompt(buffer: list, reference: list) -> str:
    buf = "\n".join(_fmt(m) for m in buffer) or "  (none)"
    ref = "\n".join(_fmt(m) for m in reference) or "  (none)"
    return (
        "Review these recently-formed MEMORIES. For each, decide whether to keep "
        "it, prune it (archive — for contradictory, unsafe, hallucinated, or "
        "useless memories), or reweight its importance. You may NOT edit memory "
        "text. When unsure, keep.\n\n"
        "MEMORIES UNDER REVIEW:\n" + buf + "\n\n"
        "ESTABLISHED KNOWLEDGE (for contradiction-checking only — do not act on "
        "these directly unless naming one as the 'archive' side of a "
        "contradiction):\n" + ref + "\n\n"
        "Respond with STRICT JSON only, no prose, in exactly this shape:\n"
        '{\n'
        '  "verdicts": [\n'
        '    {"id": <int>, "action": "keep|prune|reweight", '
        '"importance": <0..1 or null>, "reason": "<short>", '
        '"confidence": <0..1>}\n'
        '  ],\n'
        '  "contradictions": [\n'
        '    {"keep": <id>, "archive": <id>, "reason": "<short>", '
        '"confidence": <0..1>}\n'
        '  ]\n'
        '}\n'
        "Include a verdict for every memory under review. Use the highest "
        "confidence only when you are certain."
    )


def _parse(text: str) -> dict:
    """Defensively extract the JSON object from the critic's reply.

    Tolerates leading/trailing prose or code fences. On any failure returns an
    empty result, which the apply step treats as 'keep everything' (a no-op)."""
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


# --- deterministic, reversible application ------------------------------------
def _apply(buffer: list, reference: list, parsed: dict, report: RemReport) -> None:
    known = {m.id: m for m in (*buffer, *reference)}
    cap = int(len(buffer) * config.REM_MAX_PRUNE_FRACTION)
    archived_count = 0

    # Gather archive intents (from prune verdicts + contradiction losers), then
    # apply the most-confident ones first, bounded by the safety cap.
    archive_intents: list[tuple[float, int, str]] = []  # (confidence, id, reason)
    reweights: list[tuple[int, float, str, float]] = []  # (id, new_imp, reason, conf)

    for v in parsed.get("verdicts", []) or []:
        if not isinstance(v, dict):
            continue
        mid = v.get("id")
        if mid not in known:
            continue  # ignore hallucinated / out-of-set ids
        action = str(v.get("action", "keep")).lower()
        conf = float(v.get("confidence", 0.0) or 0.0)
        reason = str(v.get("reason", ""))[:200]
        if conf < config.REM_MIN_CONFIDENCE:
            if action in ("prune", "reweight"):
                report.skipped_low_conf += 1
            continue
        if action == "prune":
            archive_intents.append((conf, int(mid), reason))
        elif action == "reweight":
            imp = v.get("importance")
            if imp is not None:
                reweights.append((int(mid), float(imp), reason, conf))

    for c in parsed.get("contradictions", []) or []:
        if not isinstance(c, dict):
            continue
        loser = c.get("archive")
        if loser not in known:
            continue
        conf = float(c.get("confidence", 0.0) or 0.0)
        if conf < config.REM_MIN_CONFIDENCE:
            report.skipped_low_conf += 1
            continue
        reason = "contradiction: " + str(c.get("reason", ""))[:180]
        archive_intents.append((conf, int(loser), reason))
        report.contradictions += 1

    # De-duplicate archive ids (keep the most-confident reason), highest first.
    best: dict[int, tuple[float, str]] = {}
    for conf, mid, reason in archive_intents:
        if mid not in best or conf > best[mid][0]:
            best[mid] = (conf, reason)
    ordered = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)

    for mid, (conf, reason) in ordered:
        if archived_count >= cap:
            report.skipped_cap += 1
            continue
        store.archive(mid)
        archived_count += 1
        report.pruned += 1
        _journal(
            {
                "ts": time.time(),
                "id": mid,
                "action": "archive",
                "old_importance": known[mid].importance,
                "new_importance": known[mid].importance,
                "reason": reason,
                "confidence": conf,
            }
        )

    archived_ids = {mid for mid, _ in ordered[:archived_count]}
    for mid, new_imp, reason, conf in reweights:
        if mid in archived_ids:
            continue  # don't bother reweighting something we just archived
        old = known[mid].importance
        clamped = max(0.0, min(1.0, new_imp))
        store.set_importance(mid, clamped)
        report.reweighted += 1
        _journal(
            {
                "ts": time.time(),
                "id": mid,
                "action": "reweight",
                "old_importance": old,
                "new_importance": clamped,
                "reason": reason,
                "confidence": conf,
            }
        )


# --- public entry point -------------------------------------------------------
async def _default_ask_model(system_prompt: str, user_prompt: str):
    # Lazy import keeps the memory layer free of the Agent SDK.
    from ..agent import ask_model_oneshot

    return await ask_model_oneshot(system_prompt, user_prompt)


def _critic_system_prompt() -> str:
    path = config.PROMPTS_DIR / "rem.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


async def run_rem(
    ask_model: AskModel | None = None,
    *,
    now: float | None = None,
    batch_max: int | None = None,
) -> RemReport:
    """Run one REM ('dream') pass and return a report.

    ``ask_model`` is injected for testing; it defaults to the real one-shot model
    call. The model is an advisor — all mutations here are deterministic,
    reversible, confidence-gated, and capped.
    """
    now = time.time() if now is None else now
    ask = ask_model or _default_ask_model
    batch_max = batch_max or config.REM_BATCH_MAX
    store.init_db()
    report = RemReport()

    buffer = _select_buffer(batch_max)
    report.reviewed = len(buffer)
    if not buffer:
        _write_state({**_read_state(), "last_run": now})
        return report

    reference = _reference_set({m.id for m in buffer})
    system = _critic_system_prompt()
    user = _build_prompt(buffer, reference)

    result = await ask(system, user)
    text, cost = result if isinstance(result, tuple) else (result, None)
    report.cost_usd = cost

    parsed = _parse(text or "")
    _apply(buffer, reference, parsed, report)

    _write_state(
        {
            "last_reviewed_id": max(m.id for m in buffer),
            "last_run": now,
        }
    )
    return report
