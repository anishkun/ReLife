"""A crew member on another model — a CrewAI agent with ReLife memory attached.

``runtime: llm`` agents run CrewAI's own agent loop on whatever model the user
configured (a CrewAI/LiteLLM string: ``ollama/llama3.1``, ``gpt-4.1``,
``gemini/…`` — keys, if any, come from the environment and are CrewAI's to read)
or on Claude through the subscription (``claude-max`` → :class:`ClaudeMaxLLM`).

What they get from ReLife:

- the external memory tools, bound to the agent's scoped client
  (``memory_tools.py``) — saves land in its own space;
- the recalled-context block for each task, prepended deterministically before
  kickoff (``memory/context.build_context``), so it starts from what is known
  even if it never calls a tool;
- every step it takes journaled into its space's event log, so consolidation
  learns its recurring patterns the same way it learns a ReLife agent's;
- **only if the plan names them** (``"tools": ["files", "shell"]``), the gated
  machine tools (``machine.py``). ReLife's permission policy gates the Claude
  SDK's loop, never CrewAI's, so these tools carry the gate themselves: each
  call is checked by the same ``can_use_tool`` a ReLife member gets before it
  acts. No browser and no connectors for other models.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

from crewai import Agent

from .. import config
from .machine import TOOL_NAMES
from .memory_tools import memory_tools
from .spec import AgentSpec

LlmFactory = Callable[[str], Any]

# CrewAI's warning for a callback it can't round-trip through JSON (crewai.types.callback).
_UNSERIALIZABLE_CALLBACK = r".*callbacks cannot be serialized and will prevent checkpointing"


def default_llm(model: str) -> Any:
    """``claude-max`` → Claude via the CLI (no key); anything else → CrewAI's LLM."""
    if model == config.CREW_CLAUDE_LLM:
        from .llm import ClaudeMaxLLM

        return ClaudeMaxLLM()
    from crewai import LLM

    return LLM(model=model)


def journal_step(client: Any, task_id: str) -> Callable[[Any], None]:
    """A CrewAI ``step_callback`` that journals tool steps into the agent's space."""

    def on_step(step: Any) -> None:
        tool = getattr(step, "tool", None)
        if not tool or str(tool) in TOOL_NAMES:
            return  # machine tools journal themselves, under the ReLife tool name
        try:
            client.log_event(str(tool), brief=str(getattr(step, "tool_input", ""))[:120], task_id=task_id)
        except Exception:  # noqa: BLE001 — journaling never breaks a run
            pass

    return on_step


def llm_agent(
    spec: AgentSpec,
    *,
    memory_client: Any,
    llm: Any,
    task_id: str,
    on_step: Callable[[Any], None] | None = None,
    tools: list[Any] | None = None,
) -> Agent:
    """``tools`` are extra CrewAI tools beyond memory — the gated machine tools
    (``machine_tools.py``) when the plan gives the agent any."""
    journal = journal_step(memory_client, task_id)

    def step(s: Any) -> None:
        journal(s)
        if on_step is not None:
            on_step(s)

    # The callback is a closure by necessity (it carries this agent's scoped
    # client and task id), which CrewAI warns can't be serialized for its
    # checkpointing. ReLife never checkpoints a crew (the run record is ours),
    # so that one warning is noise; anything else still surfaces.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_UNSERIALIZABLE_CALLBACK, category=UserWarning)
        return Agent(
            role=spec.role,
            goal=spec.goal,
            backstory=spec.backstory or spec.role,
            llm=llm,
            tools=[*memory_tools(memory_client), *(tools or [])],
            allow_delegation=False,
            verbose=False,
            step_callback=step,
        )
