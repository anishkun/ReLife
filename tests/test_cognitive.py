"""Unit tests for the cognitive model (pure functions — the brain math)."""

import time

from relife import config
from relife.memory import cognitive


def test_activation_rises_with_use_count():
    now = time.time()
    low = cognitive.activation(use_count=0, last_used_at=now, importance=0.5, now=now)
    high = cognitive.activation(use_count=10, last_used_at=now, importance=0.5, now=now)
    assert high > low


def test_activation_falls_with_age():
    now = time.time()
    fresh = cognitive.activation(use_count=3, last_used_at=now, importance=0.5, now=now)
    stale = cognitive.activation(
        use_count=3, last_used_at=now - 90 * 86400, importance=0.5, now=now
    )
    assert fresh > stale


def test_importance_lifts_activation():
    now = time.time()
    plain = cognitive.activation(use_count=1, last_used_at=now, importance=0.1, now=now)
    salient = cognitive.activation(use_count=1, last_used_at=now, importance=0.9, now=now)
    assert salient > plain


def test_sigmoid_bounds():
    assert cognitive.sigmoid(0) == 0.5
    assert 0.0 <= cognitive.sigmoid(-1000) < 0.01
    assert 0.99 < cognitive.sigmoid(1000) <= 1.0


def test_should_archive_respects_age_and_pins():
    now = time.time()
    old = now - (config.MIN_FORGET_AGE_DAYS + 30) * 86400

    # Stale, low-importance, unused → forgotten.
    assert cognitive.should_archive(
        use_count=0, last_used_at=old, importance=0.1, kind="fact", now=now
    )
    # A preference is never forgotten, however stale.
    assert not cognitive.should_archive(
        use_count=0, last_used_at=old, importance=0.1, kind="preference", now=now
    )
    # A pinned (high-importance) memory is never forgotten.
    assert not cognitive.should_archive(
        use_count=0, last_used_at=old, importance=0.95, kind="fact", now=now
    )
    # Recently used → not yet eligible regardless of activation.
    assert not cognitive.should_archive(
        use_count=0, last_used_at=now, importance=0.1, kind="fact", now=now
    )


def test_fused_score_monotonic():
    base = dict(semantic=0.5, keyword=0.5, act=0.0, importance=0.5)
    s0 = cognitive.fused_score(**base)
    assert cognitive.fused_score(**{**base, "semantic": 0.9}) > s0
    assert cognitive.fused_score(**{**base, "keyword": 0.9}) > s0
    assert cognitive.fused_score(**{**base, "act": 5.0}) > s0
    assert cognitive.fused_score(**{**base, "importance": 0.9}) > s0
