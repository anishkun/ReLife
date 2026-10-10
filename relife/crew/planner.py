"""The crew planner: a task in, a validated :class:`CrewSpec` out.

One tool-less model call (``agent.ask_model_oneshot`` — Claude through the
logged-in CLI, no API key) asks for a JSON plan, told which agents already exist
(so experienced ones are reused and new ones inherit from them) and which
non-Claude models are configured. The answer is parsed and validated here; an
invalid plan gets **one** retry carrying the validation error, and if that also
fails the plan falls back to a single ReLife agent doing the whole task — a
crew run never dies on a bad plan.

The model is injected (``ask_model``) so all of this is unit-tested with a stub,
the same discipline as the REM pass.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from .. import config
from .spec import CrewSpec, normalize_spec, single_agent_spec

PLANNER_PROMPT = Path(__file__).parent / "prompts" / "planner.md"

AskModel = Callable[[str, str], Awaitable[tuple[str, float | None]]]

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def extract_json(text: str) -> Any:
    """The JSON object in a model answer: a fenced block, or the outermost
    ``{…}``. Raises ``ValueError`` when there is none."""
    m = _FENCE.search(text or "")
    candidate = m.group(1) if m else None
    if candidate is None:
        start, end = (text or "").find("{"), (text or "").rfind("}")
        if start == -1 or end <= start:
            raise ValueError("the answer contains no JSON object")
        candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"the answer's JSON doesn't parse: {e}") from None


def system_prompt() -> str:
    return (
        PLANNER_PROMPT.read_text(encoding="utf-8")
        .replace("{max_agents}", str(config.CREW_MAX_AGENTS))
        .replace("{max_tasks}", str(config.CREW_MAX_TASKS))
    )


def planner_prompt(task: str, roster: list[dict[str, Any]], llms: tuple[str, ...]) -> str:
    """The user turn: available models, existing agents, and the fenced task."""
    models = [f"- `{config.CREW_CLAUDE_LLM}` — Claude via the user's subscription (no tools)"]
    models += [f"- `{m}`" for m in llms]
    if roster:
        agents = "\n".join(
            f"- `{a['name']}` ({a['runtime']}{', ' + a['model'] if a.get('model') else ''}): "
            f"{a.get('description') or 'no description'} — {a.get('memories', 0)} memories"
            for a in roster
        )
    else:
        agents = "(none yet — every agent you define is new)"
    return (
        "## Available models (for runtime llm)\n" + "\n".join(models)
        + "\n\n## Existing agents\n" + agents
        + "\n\n## Task\n<<<TASK\n" + task.strip() + "\nTASK>>>\n\nReturn the JSON plan."
    )


async def plan_crew(
    task: str,
    *,
    roster: list[dict[str, Any]],
    ask_model: AskModel | None = None,
    llms: tuple[str, ...] | None = None,
) -> tuple[CrewSpec, list[str], float]:
    """Plan a crew for ``task``. Returns ``(spec, notes, cost_usd)``; ``notes``
    says if a retry or the single-agent fallback was needed."""
    if ask_model is None:
        from ..agent import ask_model_oneshot

        ask_model = ask_model_oneshot
    llms = config.CREW_LLMS if llms is None else llms
    existing = {a["name"] for a in roster}
    system = system_prompt()
    prompt = planner_prompt(task, roster, llms)
    notes: list[str] = []
    cost = 0.0
    for attempt in range(2):
        text, c = await ask_model(system, prompt)
        cost += c or 0.0
        try:
            spec = normalize_spec(extract_json(text), existing_agents=existing, allowed_llms=llms)
        except ValueError as e:
            notes.append(f"plan attempt {attempt + 1} was invalid: {e}")
            prompt += (
                f"\n\nYour previous answer was invalid: {e}\n"
                "Return a corrected JSON plan only."
            )
            continue
        return spec, notes, cost
    notes.append("falling back to a single ReLife agent")
    return single_agent_spec(task), notes, cost
