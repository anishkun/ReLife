"""Where a crew member's memory lives, and how it reaches it — decided per model.

Every crew member keeps memory whatever LLM it runs on. What differs is the
cost: a CrewAI agent reaches memory through tool calls, and every tool call is
one more request to its model — cheap on Claude or a local model, the whole
budget on a free tier allowing 10 requests a minute and 250 a day. So, per
member, from the plan and its model's limits (``relife/llm_limits.py``):

- **mode** — where a *new* agent's memory lives. ``individual`` (a space of its
  own) or ``collective`` (the shared pool every agent reads; see
  ``relife/agents.py``). ``auto`` picks collective for a model with *tight*
  declared limits — a small free model's learnings are better pooled than kept
  in a private memory it can barely afford to consult — and individual
  otherwise. A *registered* agent is never moved by a plan: a plan is model
  output, and moving an agent's memory is the user's act
  (``relife agent set NAME --memory …``).
- **access** — ``full`` (memory tools + the recalled block) or ``lean`` (no
  memory tools, a smaller recalled block, and ReLife saves each task's outcome
  for it with no model call). Lean for a tight model, unless
  ``RELIFE_LLM_LEAN_MEMORY`` says ``always``/``never``. ReLife (Claude) members
  are always full: their memory comes through hooks, not requests.

Pure — no CrewAI, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import config
from ..llm_limits import ModelLimits, tightness


@dataclass(frozen=True)
class MemoryPlan:
    mode: str          # individual | collective
    collective: str    # collective name ("" = the global one)
    access: str        # full | lean
    why: str = ""      # why auto chose what it did ("" when nothing was automatic)

    @property
    def label(self) -> str:
        where = "collective" + (f":{self.collective}" if self.collective else "") if self.mode == "collective" else "individual"
        return f"{where} · {self.access}"


def _access(spec_agent: Any, tight: bool) -> str:
    if spec_agent.runtime != "llm":
        return "full"
    lean = config.LLM_LEAN_MEMORY
    if lean == "always":
        return "lean"
    if lean == "never":
        return "full"
    return "lean" if tight else "full"


def resolve(spec_agent: Any, limits: ModelLimits | None) -> MemoryPlan:
    """The memory plan for one spec agent (``AgentSpec``) whose model has
    ``limits`` (``None`` for a ReLife member)."""
    tight = spec_agent.runtime == "llm" and limits is not None and tightness(limits) == "tight"
    access = _access(spec_agent, tight)
    memory = getattr(spec_agent, "memory", "auto") or "auto"
    collective = getattr(spec_agent, "collective", "") or ""
    if memory == "auto":
        if tight:
            return MemoryPlan("collective", collective, access, f"{spec_agent.llm} has tight limits")
        return MemoryPlan("individual", "", access)
    return MemoryPlan(memory, collective if memory == "collective" else "", access)
