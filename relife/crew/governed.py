"""``GovernedLLM`` — every crew model call paced under that model's limits.

CrewAI calls a member's LLM as fast as its agent loop can go, and its own
rate-limit retry gives up after three tries and eight seconds. On a free tier
that means a burst of 429s, then a dead run once the day's quota is gone. This
wrapper sits between CrewAI and the real LLM and, before every call, asks the
pure policy (``relife/llm_limits.decide``) with the shared ledger's view of the
model (``relife/llm_usage``):

- **go** — call the model, and record the call (tokens from the LLM's own
  usage counter when it has one, else estimated at chars / 4);
- **wait** — sleep (in short slices, so stopping a crew stays responsive)
  until the window has room: per-minute pacing, or a cooldown after a 429;
- **exhausted** — the model's daily quota is used up: switch to the next model
  in its declared ``fallback`` chain that isn't, and switch back to the
  preferred one as soon as it resets; with none left, wait if the reset is
  within the model's wait budget, else raise :class:`QuotaPaused` so the crew
  pauses at this point and can be resumed after the reset.

A provider refusal is read by :func:`relife.llm_limits.classify_error`: a
throttle halves the learned rate and cools down, a daily quota error marks the
model exhausted — both remembered in the ledger, so every process and every
other member on the same key backs off together.

CrewAI wraps any ``BaseLLM.call`` in its own short retry and skips it for a
nested call, so the inner model's 429 reaches us on the first try. What we
raise must never look like a throttle (or CrewAI would retry *us*), and
:class:`QuotaPaused` / :class:`GovernorStopped` derive from CrewAI's
``HookAborted`` — the one exception its agent passes through rather than
re-running the task (a re-run would just meet the same empty quota).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from crewai import BaseLLM
from crewai.hooks.dispatch import HookAborted
from crewai.llms.base_llm import call_stop_override
from pydantic import PrivateAttr

from .. import llm_usage
from ..llm_limits import ModelLimits, Switch, classify_error, decide

SLICE = 5.0          # longest single sleep, so a stop request is seen quickly
MAX_ATTEMPTS = 20    # provider refusals in one call before giving up on the call

LimitsFor = Callable[[str], ModelLimits]
Factory = Callable[[str], Any]


class QuotaPaused(HookAborted):  # noqa: N818 — a stop, not an error
    """Every model in the chain is out of quota beyond the wait budget."""

    def __init__(self, model: str, reset_at: float, reason: str = "") -> None:
        when = time.strftime("%H:%M", time.localtime(reset_at))
        super().__init__(f"{model} has no quota left until about {when}" + (f" ({reason})" if reason else ""))
        self.model = model
        self.reset_at = reset_at
        self.why = reason


class GovernorStopped(HookAborted):  # noqa: N818
    """The crew was stopped while a call was waiting for its turn."""

    def __init__(self) -> None:
        super().__init__("crew stopped by the user")


def estimate_tokens(messages: Any) -> int:
    """About how many tokens ``messages`` are (chars / 4 — no tokenizer)."""
    if isinstance(messages, str):
        return len(messages) // 4
    n = 0
    for m in messages or []:
        content = m.get("content") if isinstance(m, dict) else m
        if isinstance(content, list):
            n += sum(len(str(p.get("text", "")) if isinstance(p, dict) else str(p)) for p in content)
        else:
            n += len(str(content or ""))
    return n // 4


def _usage_total(llm: Any) -> int:
    try:
        return int(getattr(llm, "_token_usage", {}).get("total_tokens", 0))
    except Exception:  # noqa: BLE001
        return 0


class GovernedLLM(BaseLLM):
    """A CrewAI LLM that is another LLM, called only when its limits allow."""

    llm_type: str = "relife-governed"
    provider: str = "relife"

    _preferred: str = PrivateAttr()
    _current: str = PrivateAttr()
    _inners: dict[str, Any] = PrivateAttr(default_factory=dict)
    _factory: Factory | None = PrivateAttr(default=None)
    _limits_for: LimitsFor = PrivateAttr()
    _ledger: Any = PrivateAttr()
    _sleep: Callable[[float], None] = PrivateAttr()
    _asleep: Callable[[float], Any] = PrivateAttr()
    _now: Callable[[], float] = PrivateAttr()
    _should_stop: Callable[[], bool] | None = PrivateAttr(default=None)
    _on_note: Callable[[str], None] | None = PrivateAttr(default=None)
    _switches: list[Switch] = PrivateAttr(default_factory=list)
    _waited: float = PrivateAttr(default=0.0)

    def __init__(
        self,
        inner: Any,
        *,
        model: str,
        limits_for: LimitsFor | None = None,
        factory: Factory | None = None,
        ledger: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        asleep: Callable[[float], Any] = asyncio.sleep,
        now: Callable[[], float] = time.time,
        should_stop: Callable[[], bool] | None = None,
        on_note: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            stop=list(getattr(inner, "stop", None) or []),
            stream=getattr(inner, "stream", None),
        )
        if limits_for is None:
            store = llm_usage.LimitsStore()
            limits_for = store.match
        self._preferred = self._current = model
        self._inners = {model: inner}
        self._factory = factory
        self._limits_for = limits_for
        self._ledger = ledger or llm_usage
        self._sleep = sleep
        self._asleep = asleep
        self._now = now
        self._should_stop = should_stop
        self._on_note = on_note

    # --- what the rest of ReLife reads --------------------------------------
    @property
    def inner(self) -> Any:
        return self._inners[self._current]

    @property
    def current_model(self) -> str:
        return self._current

    @property
    def switches(self) -> list[Switch]:
        return list(self._switches)

    @property
    def waited_s(self) -> float:
        """Seconds spent waiting for limits so far."""
        return self._waited

    @property
    def cost_usd(self) -> float | None:
        """The inner LLMs' running total (``ClaudeMaxLLM`` reports one); ``None``
        when none of them tracks cost."""
        costs = [getattr(i, "cost_usd", None) for i in self._inners.values()]
        nums = [float(c) for c in costs if isinstance(c, (int, float))]
        return sum(nums) if nums else None

    def chain(self) -> list[str]:
        return [self._preferred, *self._limits_for(self._preferred).fallback]

    # --- choosing a model ---------------------------------------------------
    def _decide(self, model: str, est: int, now: float):
        lim = self._limits_for(model)
        u = self._ledger.usage(model, now=now, day=bool(lim.rpd or lim.tpd))
        learned = self._ledger.learned(model)
        d = decide(lim, u, learned, now, est)
        if d.action == "exhausted" and learned.exhausted_until <= now:
            # Our own count says the day is spent: tell every other process too.
            self._ledger.mark_exhausted(model, d.reset_at, now=now)
        return lim, d

    def _inner_for(self, model: str) -> Any:
        if model not in self._inners:
            if self._factory is None:
                raise QuotaPaused(self._preferred, self._now(), "no way to build its fallback model")
            self._inners[model] = self._factory(model)
        return self._inners[model]

    def _admit(self, est: int, waited: float) -> tuple[str, float]:
        """``("go", 0)`` with ``self._current`` set to the model to call, or
        ``("wait", seconds)``. Raises :class:`QuotaPaused` when waiting can't
        help within the budget."""
        now = self._now()
        budget = self._limits_for(self._preferred).wait_budget
        soonest: tuple[float, str, str] | None = None  # (reset_at, model, reason)
        blocked = ""  # why the models before the chosen one were passed over
        for model in self.chain():
            lim, d = self._decide(model, est, now)
            if d.action == "go" or (d.action == "wait" and d.wait_s <= lim.wait_budget):
                self._use(model, blocked)
                if d.action == "go":
                    return "go", 0.0
                if self._on_note is not None:
                    self._on_note(f"waiting {d.wait_s:.0f}s — {d.reason}")
                return "wait", d.wait_s
            reset = d.reset_at if d.action == "exhausted" else now + d.wait_s
            blocked = blocked or d.reason
            if soonest is None or reset < soonest[0]:
                soonest = (reset, model, d.reason)
        assert soonest is not None
        reset_at, model, reason = soonest
        if reset_at - now <= budget - waited:
            self._use(model, reason)
            if self._on_note is not None:
                self._on_note(f"every model in the chain is spent; waiting {reset_at - now:.0f}s for {model}")
            return "wait", max(0.0, reset_at - now)
        raise QuotaPaused(self._preferred, reset_at, reason)

    def _use(self, model: str, reason: str) -> None:
        if model != self._current:
            self._inner_for(model)
            self._switches.append(Switch(at=self._now(), from_model=self._current, to_model=model, reason=reason))
            if self._on_note is not None:
                self._on_note(f"switching {self._current} → {model}" + (f" ({reason})" if reason else ""))
            self._current = model

    def _check_stop(self) -> None:
        if self._should_stop is not None and self._should_stop():
            raise GovernorStopped()

    # --- the call -----------------------------------------------------------
    def _settle(self, inner: Any, before: int, est: int, result: Any) -> None:
        used = _usage_total(inner) - before
        tokens = used if used > 0 else est + len(str(result or "")) // 4
        self._ledger.record_call(self._current, tokens, "ok", now=self._now())

    def _refused(self, exc: Exception) -> bool:
        """Record a provider refusal; ``False`` when it isn't one (re-raise)."""
        kind, retry_after = classify_error(exc)
        if kind == "other":
            return False
        self._ledger.record_error(self._current, kind, retry_after, now=self._now())
        if self._on_note is not None:
            self._on_note(f"{self._current} refused a request ({kind}); backing off")
        return True

    def _govern(self, run: Callable[[Any], Any], messages: Any) -> Any:
        est = estimate_tokens(messages)
        stop = self.stop_sequences
        waited = 0.0
        for _ in range(MAX_ATTEMPTS):
            while True:
                self._check_stop()
                action, secs = self._admit(est, waited)
                if action == "go":
                    break
                step = min(secs, SLICE) if secs > 0 else 0.0
                self._sleep(step)
                waited += step
                self._waited += step
            inner = self.inner
            before = _usage_total(inner)
            try:
                with call_stop_override(inner, stop):
                    result = run(inner)
            except Exception as e:  # noqa: BLE001 — sorted by _refused
                if not self._refused(e):
                    raise
                continue  # outside the handler: what we raise next carries no 429 context
            self._settle(inner, before, est, result)
            return result
        raise QuotaPaused(self._current, self._now() + 60, "the provider kept refusing requests")

    async def _agovern(self, run: Callable[[Any], Any], messages: Any) -> Any:
        est = estimate_tokens(messages)
        stop = self.stop_sequences
        waited = 0.0
        for _ in range(MAX_ATTEMPTS):
            while True:
                self._check_stop()
                action, secs = self._admit(est, waited)
                if action == "go":
                    break
                step = min(secs, SLICE) if secs > 0 else 0.0
                await self._asleep(step)
                waited += step
                self._waited += step
            inner = self.inner
            before = _usage_total(inner)
            try:
                with call_stop_override(inner, stop):
                    result = await run(inner)
            except Exception as e:  # noqa: BLE001
                if not self._refused(e):
                    raise
                continue
            self._settle(inner, before, est, result)
            return result
        raise QuotaPaused(self._current, self._now() + 60, "the provider kept refusing requests")

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self._govern(
            lambda inner: inner.call(
                messages, tools=tools, callbacks=callbacks, available_functions=available_functions,
                from_task=from_task, from_agent=from_agent, response_model=response_model, **kwargs,
            ),
            messages,
        )

    async def acall(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
        **kwargs: Any,
    ) -> Any:
        return await self._agovern(
            lambda inner: inner.acall(
                messages, tools=tools, callbacks=callbacks, available_functions=available_functions,
                from_task=from_task, from_agent=from_agent, response_model=response_model, **kwargs,
            ),
            messages,
        )

    # --- capabilities are the current model's ---------------------------------
    def supports_function_calling(self) -> bool:
        return bool(self.inner.supports_function_calling())

    def supports_stop_words(self) -> bool:
        return bool(self.inner.supports_stop_words())

    def get_context_window_size(self) -> int:
        return int(self.inner.get_context_window_size())

    def get_token_usage_summary(self) -> Any:
        return self.inner.get_token_usage_summary()


def govern(
    llm: Any,
    model: str,
    *,
    factory: Factory | None = None,
    limits_for: LimitsFor | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_note: Callable[[str], None] | None = None,
    **kw: Any,
) -> GovernedLLM:
    """Wrap ``llm`` (built for ``model``) — idempotent: a governed LLM is
    returned as is."""
    if isinstance(llm, GovernedLLM):
        return llm
    return GovernedLLM(
        llm, model=model, factory=factory, limits_for=limits_for,
        should_stop=should_stop, on_note=on_note, **kw,
    )
