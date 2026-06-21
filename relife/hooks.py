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

from . import config
from .memory import events, skills, workflows
from .memory._text import tokenize as _tokens
from .memory.client import default_client


def _is_dup(text: str, kept: list[set[str]]) -> bool:
    """True if ``text``'s tokens overlap an already-kept block past the Jaccard
    threshold — i.e. it largely repeats something already surfaced."""
    toks = _tokens(text)
    if not toks:
        return False
    for prev in kept:
        union = toks | prev
        if union and len(toks & prev) / len(union) >= config.RECALL_DEDUP_JACCARD:
            return True
    return False


async def _recall_hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any):
    prompt = input_data.get("prompt", "")
    # Remember the prompt so the Stop hook can pair it with the tool approach.
    if prompt:
        _last_prompt[input_data.get("session_id", "") or ""] = prompt

    # Gather candidate blocks in priority order: memory, then skills, then a
    # workflow. Each carries the key text used for cross-section de-duplication.
    candidates: list[tuple[str, str, str]] = []  # (section_label, key_text, rendered)
    for m in default_client().recall(prompt, k=5, reinforce=True):
        rendered = f"- [{m.kind}] {m.text}" + (f"  ({m.tags})" if m.tags else "")
        candidates.append(("memory", m.text + " " + m.tags, rendered))
    for s in skills.find_skills(prompt, k=2):
        candidates.append(("skill", s.name + " " + s.when_to_use, f"### {s.name} — {s.when_to_use}\n{s.body}"))
    for w in workflows.find_workflows(prompt, k=1):
        candidates.append(("workflow", w.name + " " + w.when_to_use, f"### {w.name} — {w.when_to_use}\n{w.body}"))

    # De-duplicate across sections and stay within the injection budget.
    kept_tokens: list[set[str]] = []
    grouped: dict[str, list[str]] = {"memory": [], "skill": [], "workflow": []}
    used = 0
    for label, key, rendered in candidates:
        if _is_dup(key, kept_tokens):
            continue
        if used + len(rendered) > config.RECALL_INJECT_BUDGET:
            continue
        grouped[label].append(rendered)
        kept_tokens.append(_tokens(key))
        used += len(rendered)

    sections: list[str] = []
    if grouped["memory"]:
        sections.append("Relevant long-term memory (from past sessions):\n" + "\n".join(grouped["memory"]))
    if grouped["skill"]:
        sections.append("Relevant saved skills (reuse these proven procedures):\n\n" + "\n\n".join(grouped["skill"]))
    if grouped["workflow"]:
        sections.append("Relevant saved workflow (a proven multi-step plan):\n\n" + "\n\n".join(grouped["workflow"]))

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


# Last user prompt per session, captured at UserPromptSubmit so the Stop hook can
# pair task intent with the tool approach into an episode. Single-process, popped
# on Stop; bounded by the number of concurrent sessions (typically one).
_last_prompt: dict[str, str] = {}


def _episode_text(prompt: str, tools: list[str]) -> str:
    """Deterministic one-line episode: task intent + collapsed tool approach."""
    task_line = (prompt.strip().splitlines() or [""])[0][:140]
    approach = " → ".join(tools[:12]) if tools else "no tools"
    return f"Task: {task_line} | Approach: {approach}"


async def _episode_hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any):
    """On Stop, save an episode (intent + approach) for multi-step runs."""
    sid = input_data.get("session_id", "") or ""
    prompt = _last_prompt.pop(sid, "")
    if not prompt:
        return {}
    evs = events.events_by_task().get(sid, [])
    if len(evs) < config.EPISODE_MIN_EVENTS:
        return {}  # too little happened to be worth remembering
    seq: list[str] = []
    for e in evs:
        short = e.tool.split("__")[-1]
        if not seq or seq[-1] != short:
            seq.append(short)
    try:
        default_client().save(_episode_text(prompt, seq), kind="episode")
    except Exception:
        pass  # episodic capture must never break a run
    return {}


def memory_hooks() -> dict[str, list[HookMatcher]]:
    """Hook config: inject recalled context, journal tool use, capture episodes."""
    return {
        "UserPromptSubmit": [HookMatcher(hooks=[_recall_hook])],
        "PostToolUse": [HookMatcher(hooks=[_event_hook])],
        "Stop": [HookMatcher(hooks=[_episode_hook])],
    }
