"""Model limits — how fast each LLM may be called, as pure policy.

A crew member on a free API tier (10 requests a minute, 250 a day) that is
called as fast as CrewAI's loop can go hits 429 after 429, and once the day's
quota is gone the run dies. This module decides, before every call, whether the
call may go now, should wait, or cannot go until the quota resets — so ReLife
paces a model *under* its limits instead of bouncing off them.

- **Declared limits** (``ModelLimits``) come from ``data/llm_limits.json``
  (``relife llm set``), matched by exact model name or a glob (``ollama/*``).
  Built-ins: ``claude-max`` is the subscription (no rate limits here — the
  daily budget governs it), local Ollama models are unlimited, and any other
  undeclared model is paced at a conservative ``RELIFE_LLM_DEFAULT_RPM``.
- **Learned limits** (``Learned``) tighten them from what the provider says:
  a 429 halves the effective rate and starts a cooldown (from ``retry-after``
  when given), the rate recovers 10% per quiet minute, and a *daily* quota
  error marks the model exhausted until it resets (AIMD — additive increase,
  multiplicative decrease, the TCP congestion rule).
- :func:`decide` checks a sliding 60 s window and a **rolling 24 h** window, so
  it is right whatever time of day the provider resets. Per-minute limits are
  paced at ``headroom`` (80%) of what was declared; daily limits are used in
  full (a quota error catches a key shared with another app).

Pure — no I/O, no clock (``now`` is passed in), no CrewAI — the same discipline
as ``memory/cognitive.py``. The ledger that feeds it is ``llm_usage.py``; the
wrapper that applies it to a CrewAI LLM is ``crew/governed.py``.
"""

from __future__ import annotations

import fnmatch
import math
import re
from dataclasses import asdict, dataclass, replace
from email.utils import parsedate_to_datetime
from typing import Any, Iterable, Sequence

from . import config

TIERS = ("subscription", "paid", "free", "local", "unknown")
MINUTE = 60.0
DAY = 86_400.0
MAX_MODEL = 200
MAX_FALLBACKS = 5

# AIMD: a 429 halves the rate (never below MIN_FACTOR of what was declared);
# each quiet minute gives back RECOVER_PER_MINUTE of it.
MIN_FACTOR = 0.1
RECOVER_PER_MINUTE = 0.1
# Cooldown after a 429 that named no retry-after.
DEFAULT_COOLDOWN = 10.0


@dataclass(frozen=True)
class ModelLimits:
    """What one model may be asked for. ``None`` = no such limit known."""

    model: str
    rpm: int | None = None   # requests per minute
    rpd: int | None = None   # requests per (rolling) day
    tpm: int | None = None   # tokens per minute (estimated: chars / 4)
    tpd: int | None = None   # tokens per day
    tier: str = "unknown"
    headroom: float = 0.8    # pace per-minute limits at this fraction
    fallback: tuple[str, ...] = ()  # models to switch to when this one is exhausted
    max_wait_s: float | None = None  # wait this long for a reset before pausing (None = config)
    declared: bool = True    # False: a built-in default, not something the user said

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fallback"] = list(self.fallback)
        d.pop("declared")
        d.pop("model")
        return {k: v for k, v in d.items() if v not in (None, [], ())}

    @property
    def wait_budget(self) -> float:
        return config.LLM_MAX_WAIT if self.max_wait_s is None else self.max_wait_s


@dataclass(frozen=True)
class Learned:
    """What the provider has told us about this model lately."""

    factor: float = 1.0          # multiplier on per-minute limits (AIMD)
    cooldown_until: float = 0.0  # a 429 said: nothing before this
    exhausted_until: float = 0.0  # a daily quota error said: nothing before this
    updated_at: float = 0.0


@dataclass(frozen=True)
class Usage:
    """Calls already made, as ``(at, tokens)`` pairs, oldest first. ``day`` is
    only needed when a daily limit is declared (the ledger skips it otherwise)."""

    minute: tuple[tuple[float, int], ...] = ()
    day: tuple[tuple[float, int], ...] = ()


@dataclass(frozen=True)
class Decision:
    action: str            # go | wait | exhausted
    wait_s: float = 0.0    # for wait
    reset_at: float = 0.0  # for exhausted
    reason: str = ""


GO = Decision("go")


# --- the limits table ------------------------------------------------------
BUILTIN: tuple[ModelLimits, ...] = (
    ModelLimits(model="claude-max", tier="subscription", declared=False),
    ModelLimits(model="ollama/*", tier="local", declared=False),
    ModelLimits(model="ollama_chat/*", tier="local", declared=False),
)


def _pos_int(v: Any, key: str, where: str) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 1:
        raise ValueError(f"{where}: {key} must be a positive number")
    return int(v)


def validate_model_name(name: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("a model name is a non-empty string")
    name = name.strip()
    if len(name) > MAX_MODEL or any(ord(c) < 32 for c in name):
        raise ValueError(f"invalid model name {name[:40]!r}")
    return name


def limits_from_dict(model: str, d: Any) -> ModelLimits:
    """One declared record from untrusted data (the file, a request body).
    Raises ``ValueError`` saying what's wrong."""
    model = validate_model_name(model)
    where = f"limits for {model}"
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be an object")
    tier = d.get("tier", "unknown")
    if tier not in TIERS:
        raise ValueError(f"{where}: tier must be one of {', '.join(TIERS)}")
    headroom = d.get("headroom", 0.8)
    if isinstance(headroom, bool) or not isinstance(headroom, (int, float)) or not 0.1 <= headroom <= 1:
        raise ValueError(f"{where}: headroom must be between 0.1 and 1")
    fallback = d.get("fallback") or []
    if isinstance(fallback, str):
        fallback = [fallback]
    if not isinstance(fallback, list) or len(fallback) > MAX_FALLBACKS:
        raise ValueError(f"{where}: fallback must be a list of at most {MAX_FALLBACKS} models")
    chain: list[str] = []
    for f in fallback:
        f = validate_model_name(f)
        if f != model and f not in chain:
            chain.append(f)
    max_wait = d.get("max_wait_s")
    if max_wait is not None and (
        isinstance(max_wait, bool) or not isinstance(max_wait, (int, float)) or not 0 <= max_wait <= DAY
    ):
        raise ValueError(f"{where}: max_wait_s must be between 0 and 86400 seconds")
    return ModelLimits(
        model=model,
        rpm=_pos_int(d.get("rpm"), "rpm", where),
        rpd=_pos_int(d.get("rpd"), "rpd", where),
        tpm=_pos_int(d.get("tpm"), "tpm", where),
        tpd=_pos_int(d.get("tpd"), "tpd", where),
        tier=tier,
        headroom=float(headroom),
        fallback=tuple(chain),
        max_wait_s=None if max_wait is None else float(max_wait),
    )


def normalize_limits(raw: Any) -> tuple[dict[str, ModelLimits], int]:
    """The file's ``{"models": {name: {...}}}`` → declared limits, plus how many
    records were unreadable (dropped — a bad record never widens a limit into
    "unlimited"; it falls back to the conservative default)."""
    models = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(models, dict):
        raise ValueError("unexpected format (no models object)")
    out: dict[str, ModelLimits] = {}
    dropped = 0
    for name, d in models.items():
        try:
            lim = limits_from_dict(name, d)
        except ValueError:
            dropped += 1
            continue
        out[lim.model] = lim
    return out, dropped


def match_limits(model: str, declared: dict[str, ModelLimits] | None = None) -> ModelLimits:
    """The limits that apply to ``model``: a declared exact name, then the
    longest matching declared glob, then the built-ins, then the default."""
    declared = declared or {}
    if model in declared:
        return declared[model]
    globs = sorted((p for p in declared if any(c in p for c in "*?[")), key=len, reverse=True)
    for pattern in globs:
        if fnmatch.fnmatchcase(model, pattern):
            return replace(declared[pattern], model=model)
    for b in BUILTIN:
        if fnmatch.fnmatchcase(model, b.model):
            return replace(b, model=model)
    rpm = config.LLM_DEFAULT_RPM if config.LLM_DEFAULT_RPM > 0 else None
    return ModelLimits(model=model, rpm=rpm, declared=False)


def tightness(limits: ModelLimits) -> str:
    """``tight`` when the user declared limits low enough that memory should be
    frugal (collective, no tool round-trips); ``ample`` otherwise. Only declared
    numbers count: the conservative pacing default for an unknown model is a
    guess, and a guess shouldn't decide where an agent's memory lives."""
    if not limits.declared or limits.tier in ("subscription", "local"):
        return "ample"
    if limits.rpm is not None and limits.rpm <= config.LLM_TIGHT_RPM:
        return "tight"
    if limits.rpd is not None and limits.rpd <= config.LLM_TIGHT_RPD:
        return "tight"
    return "ample"


def describe(limits: ModelLimits) -> str:
    """A short human line: ``10 rpm · 250/day · free``."""
    parts = []
    if limits.rpm:
        parts.append(f"{limits.rpm} rpm")
    if limits.rpd:
        parts.append(f"{limits.rpd}/day")
    if limits.tpm:
        parts.append(f"{limits.tpm} tok/min")
    if limits.tpd:
        parts.append(f"{limits.tpd} tok/day")
    if not parts:
        parts.append("no rate limits")
    if limits.tier != "unknown":
        parts.append(limits.tier)
    if not limits.declared and limits.tier == "unknown":
        parts.append("default")  # the conservative guess for an undeclared model
    return " · ".join(parts)


# --- the decision ---------------------------------------------------------
def effective_factor(learned: Learned, now: float) -> float:
    """The AIMD factor right now: it recovers while nothing goes wrong, so a
    model that was throttled an hour ago runs at full declared speed again."""
    if learned.factor >= 1.0:
        return 1.0
    quiet_minutes = max(0.0, now - learned.updated_at) / MINUTE
    return min(1.0, learned.factor + RECOVER_PER_MINUTE * quiet_minutes)


def _requests_free_at(window: Sequence[tuple[float, int]], cap: int, span: float) -> float | None:
    """When the window (oldest first) next has room for one more request under
    ``cap``, or ``None`` if it already does."""
    if len(window) < cap:
        return None
    return window[len(window) - cap][0] + span


def _tokens_free_at(
    window: Sequence[tuple[float, int]], cap: int, est: int, span: float
) -> float | None:
    """When enough tokens have aged out for ``est`` more, or ``None`` if they
    already fit. A single call larger than the cap goes once the window is
    empty — waiting longer can't make it fit."""
    used = sum(t for _, t in window)
    excess = used + est - cap
    if excess <= 0 or not window:
        return None
    freed = 0
    for at, tokens in window:
        freed += tokens
        if freed >= excess:
            return at + span
    return window[-1][0] + span


def decide(
    limits: ModelLimits,
    usage: Usage,
    learned: Learned,
    now: float,
    est_tokens: int = 0,
) -> Decision:
    """May one call of about ``est_tokens`` go to this model at ``now``?"""
    if learned.exhausted_until > now:
        return Decision(
            "exhausted", reset_at=learned.exhausted_until,
            reason=f"{limits.model}: the provider said its quota is used up",
        )
    day = [(at, t) for at, t in usage.day if at > now - DAY]
    if limits.rpd:
        free = _requests_free_at(day, limits.rpd, DAY)
        if free is not None:
            return Decision("exhausted", reset_at=free, reason=f"{limits.model}: {limits.rpd} requests/day used")
    if limits.tpd:
        free = _tokens_free_at(day, limits.tpd, est_tokens, DAY)
        if free is not None:
            return Decision("exhausted", reset_at=free, reason=f"{limits.model}: {limits.tpd} tokens/day used")

    waits: list[tuple[float, str]] = []
    if learned.cooldown_until > now:
        waits.append((learned.cooldown_until, "cooling down after the provider refused a request"))
    minute = [(at, t) for at, t in usage.minute if at > now - MINUTE]
    scale = limits.headroom * effective_factor(learned, now)
    if limits.rpm:
        cap = max(1, math.floor(limits.rpm * scale))
        free = _requests_free_at(minute, cap, MINUTE)
        if free is not None:
            waits.append((free, f"pacing at {cap} of {limits.rpm} requests/min"))
    if limits.tpm:
        cap = max(1, math.floor(limits.tpm * scale))
        free = _tokens_free_at(minute, cap, est_tokens, MINUTE)
        if free is not None:
            waits.append((free, f"pacing at {cap} of {limits.tpm} tokens/min"))
    if waits:
        until, reason = max(waits)
        return Decision("wait", wait_s=max(0.0, until - now), reason=f"{limits.model}: {reason}")
    return GO


def adapt(
    learned: Learned, kind: str, now: float, retry_after: float | None = None
) -> Learned:
    """Fold one provider error into what we know. ``kind`` is
    :func:`classify_error`'s: ``throttle`` halves the rate and cools down,
    ``quota`` marks the model exhausted (for ``retry_after`` if the provider
    said, else ``RELIFE_LLM_QUOTA_BACKOFF``)."""
    if kind == "throttle":
        factor = max(MIN_FACTOR, effective_factor(learned, now) * 0.5)
        cooldown = now + (retry_after if retry_after and retry_after > 0 else DEFAULT_COOLDOWN)
        return replace(
            learned, factor=factor, cooldown_until=max(learned.cooldown_until, cooldown), updated_at=now
        )
    if kind == "quota":
        backoff = retry_after if retry_after and retry_after > MINUTE else config.LLM_QUOTA_BACKOFF
        return replace(learned, exhausted_until=max(learned.exhausted_until, now + backoff), updated_at=now)
    return learned


# --- reading a provider's error -------------------------------------------
_THROTTLE_CODES = frozenset({
    "429", "ratelimiterror", "ratelimitexceeded", "ratelimitexceedederror", "resourceexhausted",
    "toomanyrequests", "throttlingexception",
})
_THROTTLE_MARKERS = (
    "rate limit", "rate-limit", "ratelimit", "too many requests", "resource exhausted",
    "resource_exhausted", "exceeded your current quota", "quota exceeded",
)
# A daily (or billing) quota: waiting a minute won't help. Gemini words its
# per-minute and per-day 429s identically except for the metric name.
_QUOTA_MARKERS = (
    "insufficient_quota", "perday", "per day", "per_day", "requests per day", "tokens per day",
    "daily limit", "daily quota",
)
_RETRY_DELAY = re.compile(r"retry[_ -]?delay[\"']?\s*[:=]\s*[\"']?(\d+(?:\.\d+)?)s", re.I)
_TRY_AGAIN = re.compile(
    r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:(\d+(?:\.\d+)?)s)?\s*(?:(\d+(?:\.\d+)?)ms)?", re.I
)


def _chain(exc: BaseException) -> Iterable[BaseException]:
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def _status(e: BaseException) -> int | None:
    s = getattr(e, "status_code", None)
    if s is None:
        s = getattr(getattr(e, "response", None), "status_code", None)
    return s if isinstance(s, int) else None


def _code(e: BaseException) -> str:
    code = getattr(e, "code", None)
    return str(code or type(e).__name__).replace("_", "").lower()


def _header_retry_after(e: BaseException) -> float | None:
    headers = getattr(getattr(e, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        ms = headers.get("retry-after-ms")
        if ms is not None:
            return float(ms) / 1000.0
        v = headers.get("retry-after")
    except Exception:  # noqa: BLE001 — a header object we don't understand
        return None
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    try:
        when = parsedate_to_datetime(str(v))
    except (TypeError, ValueError):
        return None
    import time

    return max(0.0, when.timestamp() - time.time())


def _text_retry_after(text: str) -> float | None:
    m = _RETRY_DELAY.search(text)
    if m:
        return float(m.group(1))
    m = _TRY_AGAIN.search(text)
    if m and any(m.groups()):
        h, mi, s, ms = m.groups()
        return (
            int(h or 0) * 3600 + int(mi or 0) * 60 + float(s or 0) + float(ms or 0) / 1000.0
        )
    return None


def classify_error(exc: BaseException) -> tuple[str, float | None]:
    """``(kind, retry_after_s)`` for an exception from a model call: ``throttle``
    (slow down and retry), ``quota`` (exhausted until a reset), or ``other``
    (not ours to handle — re-raise). Walks the cause chain: SDKs wrap."""
    throttled = False
    text_parts: list[str] = []
    retry_after: float | None = None
    for e in _chain(exc):
        text_parts.append(str(e).lower())
        if _status(e) == 429 or _code(e) in _THROTTLE_CODES or "ratelimit" in type(e).__name__.lower():
            throttled = True
        if retry_after is None:
            retry_after = _header_retry_after(e)
    text = " ".join(text_parts)
    if "insufficient_quota" in text:
        return "quota", retry_after or _text_retry_after(text)
    if not throttled and not any(m in text for m in _THROTTLE_MARKERS):
        return "other", None
    if retry_after is None:
        retry_after = _text_retry_after(text)
    if any(m in text for m in _QUOTA_MARKERS):
        return "quota", retry_after
    return "throttle", retry_after


@dataclass
class Switch:
    """One fallback a governed LLM made (recorded on the task outcome)."""

    at: float
    from_model: str
    to_model: str
    reason: str = ""

    def line(self) -> str:
        return f"{self.from_model} → {self.to_model}" + (f" ({self.reason})" if self.reason else "")
