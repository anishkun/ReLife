"""Agent lifecycle hooks.

Two jobs:

1. **Automatic recall** (``UserPromptSubmit``) — before each prompt we inject the
   most relevant long-term memories, skills, and workflows as extra context, so
   the agent benefits from what it has learned without having to call
   ``memory_recall`` itself. Surfacing a memory *reinforces* it (recall is a
   use), so the things ReLife keeps relying on stay strong.

2. **Event logging** (``PostToolUse``) — every tool the agent uses is journaled
   to the event log. The consolidation pass mines that journal for recurring
   action sequences and turns them into reusable workflows.

Both hook callbacks are plain functions, so they're exercised directly by
deterministic tests (no live agent needed).
"""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import HookMatcher

from .memory import events, skills, store, workflows


async def _recall_hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any):
    prompt = input_data.get("prompt", "")
    sections: list[str] = []

    mem = store.recall(prompt, k=5, reinforce=True)
    if mem:
        lines = [f"- [{m.kind}] {m.text}" + (f"  ({m.tags})" if m.tags else "") for m in mem]
        sections.append("Relevant long-term memory (from past sessions):\n" + "\n".join(lines))

    sk = skills.find_skills(prompt, k=2)
    if sk:
        blocks = [f"### {s.name} — {s.when_to_use}\n{s.body}" for s in sk]
        sections.append("Relevant saved skills (reuse these proven procedures):\n\n" + "\n\n".join(blocks))

    wf = workflows.find_workflows(prompt, k=1)
    if wf:
        blocks = [f"### {w.name} — {w.when_to_use}\n{w.body}" for w in wf]
        sections.append("Relevant saved workflow (a proven multi-step plan):\n\n" + "\n\n".join(blocks))

    if not sections:
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "\n\n".join(sections),
        }
    }


def _brief(tool_input: Any) -> str:
    """One-line hint of what a tool call did (mirrors agent._tool_brief keys)."""
    if not isinstance(tool_input, dict):
        return ""
    for key in ("command", "file_path", "path", "pattern", "url", "query", "name"):
        if key in tool_input:
            return str(tool_input[key])[:120]
    return ""


async def _event_hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any):
    tool = input_data.get("tool_name") or input_data.get("tool") or ""
    if tool:
        task_id = input_data.get("session_id", "") or ""
        try:
            events.log_event(tool, _brief(input_data.get("tool_input", {})), task_id=task_id)
        except Exception:
            pass  # journaling must never break a run
    return {}


def memory_hooks() -> dict[str, list[HookMatcher]]:
    """Hook config: inject recalled context + journal tool use."""
    return {
        "UserPromptSubmit": [HookMatcher(hooks=[_recall_hook])],
        "PostToolUse": [HookMatcher(hooks=[_event_hook])],
    }
