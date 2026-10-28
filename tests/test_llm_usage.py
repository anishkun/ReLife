"""The model-usage ledger and the declared-limits file."""

from __future__ import annotations

import json

import pytest

from relife import config, llm_usage
from relife.llm_limits import DAY

NOW = 2_000_000.0


def test_calls_land_in_the_minute_and_day_windows():
    for age in (3000, 50, 10):
        llm_usage.record_call("m", 100, now=NOW - age)
    llm_usage.record_call("other", 5, now=NOW)
    u = llm_usage.usage("m", now=NOW)
    assert [round(NOW - a) for a, _ in u.minute] == [50, 10] and u.day == ()
    u = llm_usage.usage("m", now=NOW, day=True)
    assert len(u.day) == 3 and sum(t for _, t in u.day) == 300


def test_old_rows_are_pruned():
    llm_usage.record_call("m", now=NOW - 3 * DAY)
    llm_usage.record_call("m", now=NOW)
    rows = llm_usage._conn().execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    assert rows == 1


def test_errors_are_learned_counted_and_reset():
    lr = llm_usage.record_error("m", "throttle", 5, now=NOW)
    assert lr.factor == 0.5 and lr.cooldown_until == NOW + 5
    assert llm_usage.learned("m") == lr
    assert len(llm_usage.usage("m", now=NOW).minute) == 1  # the refused request counted
    llm_usage.mark_exhausted("m", NOW + 900, now=NOW)
    assert llm_usage.learned("m").exhausted_until == NOW + 900
    st = llm_usage.model_status(llm_usage.LimitsStore().match("m"), now=NOW)
    assert st["state"] == "exhausted" and st["until"] == NOW + 900
    llm_usage.reset("m")
    assert llm_usage.learned("m").factor == 1.0


def test_a_broken_ledger_never_fails_the_caller(monkeypatch):
    def boom():
        raise OSError("disk gone")

    monkeypatch.setattr(llm_usage, "_conn", boom)
    llm_usage.record_call("m")
    assert llm_usage.usage("m").minute == ()
    assert llm_usage.learned("m").factor == 1.0
    assert llm_usage.record_error("m", "throttle").factor == 0.5  # still decided, just not kept


def test_limits_file_round_trip_and_bad_records(tmp_path):
    store = llm_usage.LimitsStore()
    assert not config.LLM_LIMITS_PATH.exists()  # nothing written until a change
    store.set("gemini/flash", {"rpm": 10, "rpd": 250, "tier": "free", "fallback": ["claude-max"]})
    again = llm_usage.LimitsStore()
    assert again.match("gemini/flash").rpd == 250
    assert again.match("gemini/flash").fallback == ("claude-max",)
    with pytest.raises(ValueError):
        store.set("x", {"rpm": -1})
    assert store.remove("gemini/flash") and not store.remove("gemini/flash")

    config.LLM_LIMITS_PATH.write_text(
        json.dumps({"version": 1, "models": {"ok": {"rpm": 3}, "bad": {"rpm": "x"}}}), encoding="utf-8"
    )
    s = llm_usage.LimitsStore()
    assert s.match("ok").rpm == 3 and "1 unreadable" in s.problem


def test_an_unreadable_file_is_kept_aside_before_the_first_write():
    config.LLM_LIMITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.LLM_LIMITS_PATH.write_text("{nope", encoding="utf-8")
    s = llm_usage.LimitsStore()
    assert s.problem.startswith("unreadable")
    s.set("m", {"rpm": 4})
    assert list(config.LLM_LIMITS_PATH.parent.glob("llm_limits.json.corrupt-*"))
    assert llm_usage.LimitsStore().match("m").rpm == 4
