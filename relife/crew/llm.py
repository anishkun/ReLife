"""``ClaudeMaxLLM`` — a CrewAI LLM that is Claude through the logged-in CLI.

CrewAI needs an LLM to plan, to manage a hierarchical crew, and to drive any
agent that isn't a full ReLife agent. Pointing it at Anthropic's API would need
a metered key; ReLife runs on the Max subscription instead. This adapter answers
CrewAI's ``call`` with :func:`relife.agent.ask_model_oneshot` — the logged-in
``claude`` CLI with every tool hard-denied, no MCP, no hooks: a pure text-in /
text-out call billed to the subscription. No API key is read or needed.

It reports no native function calling, so CrewAI drives tools for it in its
ReAct text format, and it honours CrewAI's stop words (``\\nObservation:``) by
truncating — the CLI has no stop-sequence parameter. Each call spawns one CLI
turn (a few seconds), which is fine for planners, managers and reviewers.

The model call is injectable (``ask_model``) so tests make zero model calls.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from crewai import BaseLLM
from pydantic import PrivateAttr

from .. import config
from .turns import run_sync

AskModel = Callable[[str, str], Awaitable[tuple[str, float | None]]]

_DEFAULT_SYSTEM = "You are a careful, capable assistant working as part of a team of agents."


def _content_text(content: Any) -> str:
    """CrewAI message content is a string or a list of parts; keep the text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type", "text") == "text":
                out.append(str(part.get("text", "")))
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out)
    return "" if content is None else str(content)


def split_messages(messages: Any) -> tuple[str, str]:
    """``(system, prompt)`` for a one-shot call from a CrewAI message list.

    System messages become the system prompt; a lone user message is the
    prompt as-is; a longer exchange is rendered as a labelled transcript
    ending where the assistant should continue (CrewAI's ReAct loop replays
    its own previous turns this way)."""
    if isinstance(messages, str):
        return "", messages
    system: list[str] = []
    convo: list[tuple[str, str]] = []
    for m in messages or []:
        role = (m.get("role") if isinstance(m, dict) else None) or "user"
        text = _content_text(m.get("content") if isinstance(m, dict) else m)
        if role == "system":
            system.append(text)
        else:
            convo.append((role, text))
    if len(convo) == 1 and convo[0][0] == "user":
        prompt = convo[0][1]
    else:
        prompt = "\n\n".join(f"[{role}]\n{text}" for role, text in convo) + "\n\n[assistant]\n"
    return "\n\n".join(s for s in system if s), prompt


class ClaudeMaxLLM(BaseLLM):
    """A CrewAI ``BaseLLM`` answered by Claude on the user's subscription."""

    llm_type: str = "relife-claude-max"
    provider: str = "relife"

    _ask: AskModel | None = PrivateAttr(default=None)
    _cost: float = PrivateAttr(default=0.0)
    _calls: int = PrivateAttr(default=0)

    def __init__(self, *, model: str | None = None, ask_model: AskModel | None = None, **kwargs: Any):
        super().__init__(model=model or config.MODEL, **kwargs)
        self._ask = ask_model

    @property
    def cost_usd(self) -> float:
        """Usage-equivalent cost of every call so far (as the CLI reports it)."""
        return self._cost

    @property
    def calls(self) -> int:
        return self._calls

    async def _answer(self, messages: Any) -> str:
        ask = self._ask
        if ask is None:
            from ..agent import ask_model_oneshot

            ask = ask_model_oneshot
        system, prompt = split_messages(messages)
        text, cost = await ask(system or _DEFAULT_SYSTEM, prompt)
        self._cost += cost or 0.0
        self._calls += 1
        return self._apply_stop_words(text or "")

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
    ) -> str:
        return run_sync(lambda: self._answer(messages))

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
    ) -> str:
        return await self._answer(messages)

    def supports_function_calling(self) -> bool:
        return False  # CrewAI then drives tools in its ReAct text format

    def supports_stop_words(self) -> bool:
        return True  # applied by truncation in _answer

    def get_context_window_size(self) -> int:
        return 200_000
