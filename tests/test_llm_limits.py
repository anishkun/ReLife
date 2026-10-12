"""Model limits: the pure pacing policy and the provider-error reader."""

from __future__ import annotations

import pytest

from relife import config
from relife import llm_limits as L

NOW = 1_000_000.0


def lim(**kw):
    return L.ModelLimits(model=kw.pop("model", "gemini/flash"), **kw)


def calls(*ages, tokens=0):
    """``(at, tokens)`` pairs for calls made ``ages`` seconds ago, oldest first."""
    return tuple((NOW - a, tokens) for a in sorted(ages, reverse=True))


# --- decide ---------------------------------------------------------------------
def test_an_unlimited_model_always_goes():
    assert L.decide(lim(), L.Usage(minute=calls(*range(50))), L.Learned(), NOW).action == "go"


def test_per_minute_pacing_uses_headroom():
    # 10 rpm at 80% → 8 per rolling minute.
    limits = lim(rpm=10)
    assert L.decide(limits, L.Usage(minute=calls(50, 40, 30, 20, 10, 5, 2)), L.Learned(), NOW).action == "go"
    d = L.decide(limits, L.Usage(minute=calls(50, 40, 30, 20, 10, 5, 2, 1)), L.Learned(), NOW)
    assert d.action == "wait"
    # The oldest of the 8 ages out of the window in 10 s.
    assert d.wait_s == pytest.approx(10.0)
    assert "8 of 10" in d.reason


def test_calls_older_than_the_window_dont_count():
    d = L.decide(lim(rpm=1, headroom=1.0), L.Usage(minute=calls(61)), L.Learned(), NOW)
    assert d.action == "go"


def test_tokens_per_minute():
    limits = lim(tpm=1000, headroom=1.0)
    usage = L.Usage(minute=calls(30, 20, tokens=400))
    assert L.decide(limits, usage, L.Learned(), NOW, est_tokens=150).action == "go"
    d = L.decide(limits, usage, L.Learned(), NOW, est_tokens=300)
    assert d.action == "wait" and d.wait_s == pytest.approx(30.0)  # the first 400 must age out
    # One call bigger than the whole cap goes once the window is empty.
    assert L.decide(limits, L.Usage(), L.Learned(), NOW, est_tokens=5000).action == "go"


def test_daily_limit_is_used_in_full_and_reports_its_reset():
    limits = lim(rpd=3)
    assert L.decide(limits, L.Usage(day=calls(5000, 4000)), L.Learned(), NOW).action == "go"
    d = L.decide(limits, L.Usage(day=calls(5000, 4000, 3000)), L.Learned(), NOW)
    assert d.action == "exhausted"
    assert d.reset_at == pytest.approx(NOW - 5000 + L.DAY)


def test_daily_tokens():
    d = L.decide(lim(tpd=1000), L.Usage(day=calls(100, tokens=900)), L.Learned(), NOW, est_tokens=200)
    assert d.action == "exhausted" and d.reset_at == pytest.approx(NOW - 100 + L.DAY)


def test_learned_exhaustion_and_cooldown():
    assert L.decide(lim(), L.Usage(), L.Learned(exhausted_until=NOW + 50), NOW).action == "exhausted"
    d = L.decide(lim(), L.Usage(), L.Learned(cooldown_until=NOW + 7), NOW)
    assert d.action == "wait" and d.wait_s == pytest.approx(7)


def test_the_learned_factor_tightens_pacing_and_recovers():
    limits = lim(rpm=10, headroom=1.0)
    halved = L.Learned(factor=0.5, updated_at=NOW)
    usage = L.Usage(minute=calls(30, 20, 10, 5, 1))
    assert L.decide(limits, usage, halved, NOW).action == "wait"  # cap 5
    # Five quiet minutes later it is back to the declared rate.
    later = NOW + 300
    assert L.effective_factor(halved, later) == 1.0
    assert L.decide(limits, L.Usage(minute=tuple((later - a, 0) for a in (30, 20, 10, 5, 1))), halved, later).action == "go"


# --- adapt (AIMD) ------------------------------------------------------------------
def test_a_throttle_halves_and_cools_down():
    out = L.adapt(L.Learned(), "throttle", NOW, retry_after=12)
    assert out.factor == 0.5 and out.cooldown_until == NOW + 12
    out = L.adapt(out, "throttle", NOW)
    assert out.factor == 0.25 and out.cooldown_until == NOW + 12  # never shortened
    for _ in range(10):
        out = L.adapt(out, "throttle", NOW)
    assert out.factor == L.MIN_FACTOR


def test_a_quota_error_marks_the_model_exhausted(monkeypatch):
    monkeypatch.setattr(config, "LLM_QUOTA_BACKOFF", 3600.0)
    assert L.adapt(L.Learned(), "quota", NOW).exhausted_until == NOW + 3600
    # A provider-given reset longer than a minute is believed.
    assert L.adapt(L.Learned(), "quota", NOW, retry_after=7200).exhausted_until == NOW + 7200
    assert L.adapt(L.Learned(), "other", NOW) == L.Learned()


# --- the table -------------------------------------------------------------------
def test_matching_exact_then_glob_then_builtin_then_default(monkeypatch):
    monkeypatch.setattr(config, "LLM_DEFAULT_RPM", 10)
    declared = {
        "openrouter/*": L.limits_from_dict("openrouter/*", {"rpm": 20}),
        "openrouter/*:free": L.limits_from_dict("openrouter/*:free", {"rpm": 5, "rpd": 50, "tier": "free"}),
        "gpt-4.1": L.limits_from_dict("gpt-4.1", {"rpm": 500}),
    }
    assert L.match_limits("gpt-4.1", declared).rpm == 500
    m = L.match_limits("openrouter/llama:free", declared)
    assert m.rpm == 5 and m.model == "openrouter/llama:free"  # the longest glob wins
    assert L.match_limits("openrouter/llama", declared).rpm == 20
    assert L.match_limits("claude-max", declared).tier == "subscription"
    assert L.match_limits("ollama/llama3.1", declared).rpm is None
    unknown = L.match_limits("gemini/flash", declared)
    assert unknown.rpm == 10 and not unknown.declared
    monkeypatch.setattr(config, "LLM_DEFAULT_RPM", 0)
    assert L.match_limits("gemini/flash", {}).rpm is None


def test_tightness_counts_only_declared_numbers(monkeypatch):
    monkeypatch.setattr(config, "LLM_DEFAULT_RPM", 2)
    assert L.tightness(L.match_limits("gemini/flash", {})) == "ample"  # a default is a guess
    assert L.tightness(L.limits_from_dict("g", {"rpm": 10})) == "tight"
    assert L.tightness(L.limits_from_dict("g", {"rpd": 250})) == "tight"
    assert L.tightness(L.limits_from_dict("g", {"rpm": 500, "rpd": 10000})) == "ample"
    assert L.tightness(L.limits_from_dict("g", {"rpm": 1, "tier": "local"})) == "ample"


@pytest.mark.parametrize("bad", [
    {"rpm": 0}, {"rpm": -1}, {"rpm": "10"}, {"rpm": True}, {"rpm": float("nan")},
    {"tier": "gold"}, {"headroom": 2}, {"fallback": "x" * 300}, {"fallback": ["a"] * 9},
    {"max_wait_s": -5}, "not an object",
])
def test_bad_records_are_refused(bad):
    with pytest.raises(ValueError):
        L.limits_from_dict("m", bad)


def test_normalize_drops_bad_records_and_dedupes_fallbacks():
    out, dropped = L.normalize_limits({"models": {
        "good": {"rpm": 5, "fallback": ["claude-max", "claude-max", "good"]},
        "bad": {"rpm": "fast"},
    }})
    assert dropped == 1 and out["good"].fallback == ("claude-max",)
    with pytest.raises(ValueError):
        L.normalize_limits(["not", "a", "file"])


def test_describe():
    assert L.describe(L.limits_from_dict("m", {"rpm": 10, "rpd": 250, "tier": "free"})) == "10 rpm · 250/day · free"
    assert "no rate limits" in L.describe(L.match_limits("ollama/x", {}))


# --- reading provider errors --------------------------------------------------------------
class _Resp:
    def __init__(self, status=429, headers=None):
        self.status_code = status
        self.headers = headers or {}


class ProviderError(Exception):
    def __init__(self, msg, status=None, headers=None, code=None):
        super().__init__(msg)
        if status is not None:
            self.response = _Resp(status, headers)
        if code is not None:
            self.code = code


def test_a_429_with_retry_after_is_a_throttle():
    assert L.classify_error(ProviderError("slow down", 429, {"retry-after": "7"})) == ("throttle", 7.0)
    assert L.classify_error(ProviderError("slow down", 429, {"retry-after-ms": "1500"})) == ("throttle", 1.5)


def test_message_only_rate_limits_and_retry_hints():
    kind, ra = L.classify_error(RuntimeError("Rate limit reached for model x. Please try again in 7.66s."))
    assert kind == "throttle" and ra == pytest.approx(7.66)
    kind, ra = L.classify_error(RuntimeError('429 RESOURCE_EXHAUSTED ... "retryDelay": "37s"'))
    assert kind == "throttle" and ra == 37
    kind, ra = L.classify_error(RuntimeError("Too Many Requests: try again in 1m30s"))
    assert kind == "throttle" and ra == 90


def test_daily_and_billing_quotas_are_not_throttles():
    gemini_day = ProviderError(
        "You exceeded your current quota. quota_id: GenerateRequestsPerDayPerProjectPerModel-FreeTier", 429
    )
    assert L.classify_error(gemini_day)[0] == "quota"
    groq_day = RuntimeError("Rate limit reached for model on requests per day (RPD): Limit 14400")
    assert L.classify_error(groq_day)[0] == "quota"
    assert L.classify_error(ProviderError("insufficient_quota", 429))[0] == "quota"
    # Gemini words its per-minute 429 the same way, minus "per day".
    gemini_minute = ProviderError(
        "You exceeded your current quota. quota_id: GenerateRequestsPerMinutePerProjectPerModel-FreeTier", 429
    )
    assert L.classify_error(gemini_minute)[0] == "throttle"


def test_the_cause_chain_is_walked_and_other_errors_are_other():
    try:
        try:
            raise ProviderError("nope", 429)
        except ProviderError as inner:
            raise RuntimeError("LLM call failed") from inner
    except RuntimeError as wrapped:
        assert L.classify_error(wrapped)[0] == "throttle"
    assert L.classify_error(ValueError("bad JSON in the answer")) == ("other", None)
    assert L.classify_error(ProviderError("unauthorized", 401)) == ("other", None)
