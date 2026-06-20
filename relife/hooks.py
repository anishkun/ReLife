"""Agent lifecycle hooks.

Currently: automatic memory recall. On every user prompt, we look up relevant
long-term memories and inject them as additional context — so the agent benefits
from what it has learned without having to remember to call ``memory_recall``
itself. (It still has the tool for explicit lookups.)
"""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import HookMatcher

from .memory import skills, store


async def _recall_hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any):
    prompt = input_data.get("prompt", "")
    sections: list[str] = []

    mem = store.recall(prompt, k=5)
    if mem:
        lines = [f"- [{m.kind}] {m.text}" + (f"  ({m.tags})" if m.tags else "") for m in mem]
        sections.append("Relevant long-term memory (from past sessions):\n" + "\n".join(lines))

    sk = skills.find_skills(prompt, k=2)
    if sk:
        blocks = [f"### {s.name} — {s.when_to_use}\n{s.body}" for s in sk]
        sections.append("Relevant saved skills (reuse these proven procedures):\n\n" + "\n\n".join(blocks))

    if not sections:
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "\n\n".join(sections),
        }
    }


def memory_hooks() -> dict[str, list[HookMatcher]]:
    """Hook config: inject recalled memory on each user prompt."""
    return {"UserPromptSubmit": [HookMatcher(hooks=[_recall_hook])]}
