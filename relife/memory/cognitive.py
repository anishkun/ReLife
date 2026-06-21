"""The cognitive model: how a memory's relevance rises and fades over time.

This is the deterministic heart of ReLife's brain-like memory. It has **no I/O
and no LLM** — just pure functions over a memory's stats, so it is trivially
unit-testable and the same math is reused by recall (ranking) and consolidation
(forgetting).

Inspired by ACT-R's base-level activation: a memory grows stronger the more
often and more recently it is used, and decays as it goes unused. Recall fuses
this activation with semantic similarity, keyword overlap, and explicit
importance into a single score.
"""

from __future__ import annotations

import math
import time

from .. import config


def sigmoid(x: float) -> float:
    """Squash an unbounded activation into (0, 1) for fusing with other signals."""
    if x < -60:  # avoid overflow on extreme inputs
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def age_days(timestamp: float, now: float | None = None) -> float:
    """Whole-and-fractional days since ``timestamp`` (never negative)."""
    now = time.time() if now is None else now
    return max(0.0, (now - timestamp) / 86400.0)


def activation(
    *,
    use_count: int,
    last_used_at: float,
    importance: float = 0.5,
    now: float | None = None,
) -> float:
    """ACT-R-inspired base-level activation of a memory.

    Rises with how often it has been used (``use_count``) and how recently
    (``last_used_at``); decays with idle age at rate ``config.DECAY``. Explicit
    ``importance`` (0..1) adds a steady lift so salient memories resist decay.

        activation = ln(1 + use_count)
                     - DECAY * ln(1 + age_days(last_used))
                     + IMPORTANCE_BOOST * importance
    """
    recency = config.DECAY * math.log1p(age_days(last_used_at, now))
    frequency = math.log1p(max(0, use_count))
    return frequency - recency + config.IMPORTANCE_BOOST * float(importance)


def fused_score(
    *,
    semantic: float,
    keyword: float,
    act: float,
    importance: float,
    kind: str | None = None,
) -> float:
    """Combine the four recall signals into one ranking score.

    ``semantic`` and ``keyword`` are expected in [0, 1]; ``act`` is raw
    activation (squashed here); ``importance`` is [0, 1]. Weights live in config
    so the balance can be tuned in one place. ``kind`` adds a small per-kind
    prior (``config.KIND_RECALL_BOOST``) so durable kinds rank slightly higher
    at equal evidence; it is optional and defaults to neutral.
    """
    boost = config.KIND_RECALL_BOOST.get(kind, 0.0) if kind else 0.0
    return (
        config.W_SEM * float(semantic)
        + config.W_KW * float(keyword)
        + config.W_ACT * sigmoid(act)
        + config.W_IMP * float(importance)
        + boost
    )


def should_archive(
    *,
    use_count: int,
    last_used_at: float,
    importance: float,
    kind: str,
    now: float | None = None,
) -> bool:
    """Whether a memory has faded enough to be archived (soft-forgotten).

    A memory is archived only when ALL hold: it is idle past
    ``MIN_FORGET_AGE_DAYS``, its activation has dropped below
    ``FORGET_THRESHOLD``, and it is not pinned. ``preference`` memories and items
    with ``importance >= PIN_THRESHOLD`` are never archived — like core facts a
    person keeps regardless of use.
    """
    if kind == "preference" or importance >= config.PIN_THRESHOLD:
        return False
    if age_days(last_used_at, now) < config.MIN_FORGET_AGE_DAYS:
        return False
    act = activation(
        use_count=use_count,
        last_used_at=last_used_at,
        importance=importance,
        now=now,
    )
    return act < config.FORGET_THRESHOLD


def should_hard_delete(
    *,
    last_used_at: float,
    importance: float,
    kind: str,
    now: float | None = None,
) -> bool:
    """Whether an already-archived memory has been idle long enough to delete.

    The slow second tier of forgetting: a memory that was soft-archived and then
    left untouched past ``HARD_DELETE_AGE_DAYS`` is removed for good, so the store
    doesn't grow without bound. ``preference`` memories and pinned items
    (``importance >= PIN_THRESHOLD``) are exempt — though in practice they are
    never archived, so this is just defence in depth. Callers apply this only to
    rows already in the ``archived`` state.
    """
    if kind == "preference" or importance >= config.PIN_THRESHOLD:
        return False
    return age_days(last_used_at, now) >= config.HARD_DELETE_AGE_DAYS
