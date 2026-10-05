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

3. **Episode capture** (``Stop``) — a deterministic one-liner (task intent +
   this turn's collapsed tool approach) is saved for multi-step runs, so the
   recurring-episode detector has material even when the agent never calls
   ``memory_save``. Scoped to the *turn*, not the session: the prompt hook
   records where the turn starts in the journal.

All three are plain functions exercised directly by deterministic tests (no
live agent). They fail **soft**: a memory error degrades to "no context" /
"nothing captured" and never breaks the run.
"""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import HookMatcher

from . import config
from .memory._text import tokenize as _tokens
from .memory.client import default_client, off_loop


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


def _turn_start_event_id(client: Any, session_id: str) -> int:
    """Id of the session's latest journaled event right now (0 if none).

    The event log is keyed by *session*, and a chat / server session runs many
    turns — so without this watermark every Stop episode would replay every
    earlier turn's tools as if they belonged to this one.
    """
    try:
        evs = client.events_for_task(session_id)
        return max((e.id for e in evs), default=0)
    except Exception:
        return 0  # unknown ⇒ take everything; a partial journal is still an episode


async def _recall_hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any):
    prompt = input_data.get("prompt", "")
    sid = input_data.get("session_id", "") or ""
    client = default_client()
    # Remember the prompt (and where this turn starts in the journal) so the
    # Stop hook can pair intent with *this turn's* tool approach.
    if prompt:
        _last_prompt[sid] = (prompt, await off_loop(_turn_start_event_id, client, sid))

    try:
        # Off the loop: under `relife serve` every session shares it.
        return await off_loop(_recall_context, client, prompt)
    except Exception:
        # Recall is a convenience, not a dependency: a memory daemon that is
        # down (or any store error) must degrade to "no recalled context",
        # never break the prompt. Same fail-soft rule as the other two hooks.
        return {}


def _recall_context(client: Any, prompt: str) -> dict[str, Any]:
    # Gather candidate blocks in priority order: memory, then skills, then a
    # workflow. Each carries the key text used for cross-section de-duplication.
    candidates: list[tuple[str, str, str]] = []  # (section_label, key_text, rendered)
    for m in client.recall(prompt, k=5, reinforce=True):
        rendered = f"- [{m.kind}] {m.text}" + (f"  ({m.tags})" if m.tags else "")
        candidates.append(("memory", m.text + " " + m.tags, rendered))
    for s in client.skill_find(prompt, k=2):
        candidates.append(("skill", s.name + " " + s.when_to_use, f"### {s.name} — {s.when_to_use}\n{s.body}"))
    for w in client.workflow_find(prompt, k=1):
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
            # Routes through the client, so events reach the daemon when
            # RELIFE_MEMORY_URL is set (one loopback POST per tool call — fine at
            # localhost; in-process by default). Consolidate reads them back
            # daemon-side.
            await off_loop(
                default_client().log_event,
                tool, _brief(input_data.get("tool_input", {})), task_id=task_id,
            )
        except Exception:
            pass  # journaling must never break a run
    return {}


# Last user prompt per session (+ the event id the turn started after), captured
# at UserPromptSubmit so the Stop hook can pair task intent with *this turn's*
# tool approach into an episode. Single-process, popped on Stop; bounded by the
# number of concurrent sessions.
_last_prompt: dict[str, tuple[str, int]] = {}


def _episode_text(prompt: str, tools: list[str]) -> str:
    """Deterministic one-line episode: task intent + collapsed tool approach."""
    task_line = (prompt.strip().splitlines() or [""])[0][:140]
    approach = " → ".join(tools[:12]) if tools else "no tools"
    return f"Task: {task_line} | Approach: {approach}"


async def _episode_hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any):
    """On Stop, save an episode (intent + approach) for multi-step runs."""
    sid = input_data.get("session_id", "") or ""
    prompt, start_id = _last_prompt.pop(sid, ("", 0))
    if not prompt:
        return {}
    client = default_client()
    try:
        evs = [e for e in await off_loop(client.events_for_task, sid) if e.id > start_id]
    except Exception:
        return {}  # journal unreachable — nothing to capture, nothing to break
    if len(evs) < config.EPISODE_MIN_EVENTS:
        return {}  # too little happened to be worth remembering
    seq: list[str] = []
    for e in evs:
        short = e.tool.split("__")[-1]
        if not seq or seq[-1] != short:
            seq.append(short)
    try:
        await off_loop(client.save, _episode_text(prompt, seq), kind="episode")
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
