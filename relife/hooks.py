"""Agent lifecycle hooks.

Two jobs:

1. **Automatic recall** (``UserPromptSubmit``) — before each prompt we inject the
   most relevant long-term memories, skills, and workflows as extra context, so
   the agent benefits from what it has learned without having to call
   ``memory_recall`` itself. Surfacing a memory *reinforces* it (recall is a
   use), so the things ReLife keeps relying on stay strong. The block itself is
   built by ``memory/context.py`` (shared with the ``memory_context`` MCP tool).

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

``memory_hooks(client)`` binds them to one memory client — a
``ScopedMemoryClient`` for an agent with its own memory space — so recall,
journaling and episodes all stay inside that agent's scope. Without a client
they resolve this module's ``default_client`` at call time (the main agent).
"""

from __future__ import annotations

from typing import Any, Callable

from claude_agent_sdk import HookMatcher

from . import config
from .memory.client import default_client, off_loop
from .memory.context import build_context


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


def _recall_context(client: Any, prompt: str) -> dict[str, Any]:
    text = build_context(client, prompt)
    if not text:
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": text,
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


# Last user prompt per session (+ the event id the turn started after), captured
# at UserPromptSubmit so the Stop hook can pair task intent with *this turn's*
# tool approach into an episode. Single-process, popped on Stop; bounded by the
# number of concurrent sessions (session ids are unique across agents).
_last_prompt: dict[str, tuple[str, int]] = {}


def _episode_text(prompt: str, tools: list[str]) -> str:
    """Deterministic one-line episode: task intent + collapsed tool approach."""
    task_line = (prompt.strip().splitlines() or [""])[0][:140]
    approach = " → ".join(tools[:12]) if tools else "no tools"
    return f"Task: {task_line} | Approach: {approach}"


def _make_hooks(get_client: Callable[[], Any]):
    """The three hooks over whichever client ``get_client`` returns."""

    async def recall_hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any):
        prompt = input_data.get("prompt", "")
        sid = input_data.get("session_id", "") or ""
        client = get_client()
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

    async def event_hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any):
        tool = input_data.get("tool_name") or input_data.get("tool") or ""
        if tool:
            task_id = input_data.get("session_id", "") or ""
            try:
                # Routes through the client, so events reach the daemon when
                # RELIFE_MEMORY_URL is set (one loopback POST per tool call —
                # fine at localhost; in-process by default). Consolidate reads
                # them back daemon-side.
                await off_loop(
                    get_client().log_event,
                    tool, _brief(input_data.get("tool_input", {})), task_id=task_id,
                )
            except Exception:
                pass  # journaling must never break a run
        return {}

    async def episode_hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any):
        """On Stop, save an episode (intent + approach) for multi-step runs."""
        sid = input_data.get("session_id", "") or ""
        prompt, start_id = _last_prompt.pop(sid, ("", 0))
        if not prompt:
            return {}
        client = get_client()
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

    return recall_hook, event_hook, episode_hook


# The main agent's hooks: ``default_client`` is looked up in this module at
# call time (tests monkeypatch it here).
_recall_hook, _event_hook, _episode_hook = _make_hooks(lambda: default_client())


def memory_hooks(client: Any = None) -> dict[str, list[HookMatcher]]:
    """Hook config: inject recalled context, journal tool use, capture episodes.

    ``client`` binds them to one memory client (an agent's
    ``ScopedMemoryClient``); omitted, they serve the main agent.
    """
    if client is None:
        recall, event, episode = _recall_hook, _event_hook, _episode_hook
    else:
        recall, event, episode = _make_hooks(lambda: client)
    return {
        "UserPromptSubmit": [HookMatcher(hooks=[recall])],
        "PostToolUse": [HookMatcher(hooks=[event])],
        "Stop": [HookMatcher(hooks=[episode])],
    }
