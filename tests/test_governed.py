"""GovernedLLM — crew model calls paced under each model's limits.

Real CrewAI (the [crewai] extra), fake LLMs and a fake clock: no model call and
no real sleep anywhere.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("crewai")

from crewai import BaseLLM  # noqa: E402
from crewai.hooks.dispatch import HookAborted  # noqa: E402
from crewai.llms.base_llm import call_stop_override  # noqa: E402
from crewai.llms.retry import _ThrottlingErrorClassifier  # noqa: E402
from pydantic import PrivateAttr  # noqa: E402

from relife import llm_usage  # noqa: E402
from relife.crew.governed import GovernedLLM, GovernorStopped, QuotaPaused, govern  # noqa: E402
from relife.llm_limits import limits_from_dict, match_limits  # noqa: E402


class Clock:
    def __init__(self, t: float = 5_000_000.0) -> None:
        self.t = t
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


class _Resp:
    def __init__(self, status, headers):
        self.status_code = status
        self.headers = headers


class ProviderError(Exception):
    def __init__(self, msg: str, status: int = 429, headers: dict | None = None) -> None:
        super().__init__(msg)
        self.response = _Resp(status, headers or {})


class FakeLLM(BaseLLM):
    """Answers from a script; an Exception in the script is raised instead."""

    _script: list[Any] = PrivateAttr(default_factory=list)
    _fc: bool = PrivateAttr(default=False)
    _calls: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    def __init__(self, *script: Any, fc: bool = False, model: str = "fake") -> None:
        super().__init__(model=model)
        self._script = list(script)
        self._fc = fc

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None, **kwargs):
        self._calls.append({"tools": tools, "stop": list(self.stop_sequences)})
        item = self._script.pop(0) if self._script else "Thought: done\nFinal Answer: ok"
        if isinstance(item, Exception):
            raise item
        return item

    def supports_function_calling(self) -> bool:
        return self._fc


def table(**declared):
    t = {m.replace("__", "/"): limits_from_dict(m.replace("__", "/"), d) for m, d in declared.items()}
    return lambda model: match_limits(model, t)


def governed(inner, clock, *, model="free/m", limits_for=None, factory=None, **kw) -> GovernedLLM:
    return GovernedLLM(
        inner, model=model, limits_for=limits_for or table(), factory=factory,
        sleep=clock.sleep, now=clock.now, **kw,
    )


def test_calls_are_paced_under_the_declared_rate():
    clock = Clock()
    inner = FakeLLM("a", "b", "c")
    llm = governed(inner, clock, limits_for=table(free__m={"rpm": 2, "headroom": 1.0}))
    assert [llm.call("hi"), llm.call("hi")] == ["a", "b"]
    assert clock.slept == []
    assert llm.call("hi") == "c"
    assert sum(clock.slept) == pytest.approx(60.0)  # the first call aged out of the minute
    assert max(clock.slept) <= 5.0  # in slices, so a stop request is seen
    assert len(llm_usage.usage("free/m", now=clock.t).minute) == 1


def test_a_429_backs_off_once_and_is_learned_without_crewai_retrying_too():
    clock = Clock()
    inner = FakeLLM(ProviderError("slow down", headers={"retry-after": "3"}), "fine")
    llm = governed(inner, clock)
    assert llm.call("hi") == "fine"
    assert len(inner._calls) == 2  # CrewAI's nested retry stayed out of it
    assert sum(clock.slept) == pytest.approx(3.0)
    assert llm_usage.learned("free/m").factor == 0.5


def test_an_exhausted_model_falls_back_and_returns_when_it_resets():
    clock = Clock()
    inner = FakeLLM(ProviderError("insufficient_quota"))
    backup = FakeLLM("from backup", "backup again")
    built: list[str] = []

    def factory(model):
        built.append(model)
        return backup

    notes: list[str] = []
    llm = governed(inner, clock, factory=factory, on_note=notes.append,
                   limits_for=table(free__m={"rpm": 100, "fallback": ["claude-max"]}))
    assert llm.call("hi") == "from backup"
    assert built == ["claude-max"] and llm.current_model == "claude-max"
    assert [s.line().split(" (")[0] for s in llm.switches] == ["free/m → claude-max"]
    assert any("switching" in n for n in notes)
    assert llm.call("hi") == "backup again"  # still exhausted: no new attempt on free/m
    assert len(inner._calls) == 1
    # After the reset the preferred model is used again.
    clock.t = llm_usage.learned("free/m").exhausted_until + 1
    inner._script = ["preferred again"]
    assert llm.call("hi") == "preferred again" and llm.current_model == "free/m"


def test_with_no_fallback_a_far_reset_pauses_the_crew():
    clock = Clock()
    llm = governed(FakeLLM(ProviderError("insufficient_quota")), clock,
                   limits_for=table(free__m={"max_wait_s": 60}))
    with pytest.raises(QuotaPaused) as e:
        llm.call("hi")
    assert isinstance(e.value, HookAborted)  # CrewAI's agent won't re-run the task
    assert e.value.reset_at > clock.t + 60
    # ... and CrewAI's own retry must not mistake it for a throttle.
    assert not _ThrottlingErrorClassifier.is_throttling_error(e.value)
    assert e.value.__context__ is None


def test_a_near_reset_is_waited_for():
    clock = Clock()
    for age in (300, 200):
        llm_usage.record_call("free/m", now=clock.t - age)
    llm = governed(FakeLLM("after the wait"), clock,
                   limits_for=table(free__m={"rpd": 2, "max_wait_s": 86_400}))
    # Rolling day: a slot frees when the 300 s-old call turns a day old.
    assert llm.call("hi") == "after the wait"
    assert sum(clock.slept) == pytest.approx(86_400 - 300, abs=1)


def test_a_stop_request_ends_the_wait():
    clock = Clock()
    stop = {"now": False}

    def sleep(s):
        clock.sleep(s)
        stop["now"] = True

    llm = GovernedLLM(FakeLLM(ProviderError("slow", headers={"retry-after": "30"})), model="free/m",
                      limits_for=table(), sleep=sleep, now=clock.now, should_stop=lambda: stop["now"])
    with pytest.raises(GovernorStopped):
        llm.call("hi")
    assert clock.slept == [5.0]


def test_other_errors_pass_through_untouched():
    llm = governed(FakeLLM(ProviderError("unauthorized", status=401)), Clock())
    with pytest.raises(ProviderError):
        llm.call("hi")


def test_stop_words_and_capabilities_reach_the_inner_llm():
    inner = FakeLLM("x", fc=True)
    llm = governed(inner, Clock())
    with call_stop_override(llm, ["\nObservation:"]):
        llm.call("hi", tools=[{"name": "t"}])
    assert inner._calls[0]["stop"] == ["\nObservation:"]
    assert inner._calls[0]["tools"] == [{"name": "t"}]
    assert llm.supports_function_calling() is True
    assert govern(llm, "free/m") is llm


def test_a_native_function_calling_model_runs_a_real_crew_through_the_wrapper():
    """The spike: CrewAI's executor drives a delegating BaseLLM like any other."""
    from crewai import Agent, Crew, Task

    from relife.crew.memory_tools import memory_tools
    from relife.memory.client import LocalMemoryClient

    inner = FakeLLM("the final answer", fc=True)
    llm = governed(inner, Clock())
    agent = Agent(role="r", goal="g", backstory="b", llm=llm,
                  tools=memory_tools(LocalMemoryClient()), allow_delegation=False)
    crew = Crew(agents=[agent], tasks=[Task(description="d", expected_output="e", agent=agent)],
                memory=False, planning=False)
    assert crew.kickoff().raw == "the final answer"
    assert inner._calls and inner._calls[0]["tools"]  # native tools were handed through
