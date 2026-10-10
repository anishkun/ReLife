"""One ReLife agent turn — how a crew member that is a full ReLife agent works.

Each crew task a ReLife (Claude) agent takes is one **fresh** ``ClaudeSDKClient``
turn, the way ``relife build`` delegates each milestone to a fresh-context
builder: the transcript doesn't carry between tasks, memory does. The turn gets
exactly what ``relife do --agent NAME`` gets:

- ReLife's tools and permission policy (``classify()`` via the ``can_use_tool``
  the caller passes — interactive on a terminal, deny-when-unattended
  otherwise), so a crew can't do anything a single agent couldn't;
- the agent's *scoped* memory — recall hook, journaled events, episodes, and
  the ``relife_memory`` tools all bound to its own space and what it inherited;
- plus any extra MCP servers (a crew's tools), never under the trusted prefix.

``result_from_events`` is pure (tested without a model); ``run_relife_turn`` is
the thin SDK wrapper around it.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from .. import config
from ..server.runs import summarize_events

OUTPUT_MAX_CHARS = 20000  # a task's output feeds the next task; keep it bounded

OnEvent = Callable[[dict[str, Any]], None]


@dataclass
class TurnResult:
    output: str
    tool_calls: int = 0
    cost_usd: float | None = None
    error: str | None = None
    denied: list[dict[str, Any]] = field(default_factory=list)


def result_from_events(events: list[dict[str, Any]]) -> TurnResult:
    """The turn's outcome: its closing text in full (what it said after its last
    tool call — or everything, if it used none) plus counts and cost."""
    s = summarize_events(events)
    last_tool = max(
        (i for i, ev in enumerate(events) if ev.get("type") in ("tool_use", "tool_result")),
        default=-1,
    )
    closing = "".join(
        ev["text"] for ev in events[last_tool + 1 :] if ev.get("type") == "text" and ev.get("text")
    ).strip()
    if not closing:  # ended on a tool call: fall back to everything it said
        closing = "\n".join(ev["text"] for ev in events if ev.get("type") == "text").strip()
    if len(closing) > OUTPUT_MAX_CHARS:
        closing = closing[: OUTPUT_MAX_CHARS - 1].rstrip() + "…"
    return TurnResult(
        output=closing,
        tool_calls=s["tool_calls"],
        cost_usd=s["cost_usd"],
        error=s["error"],
        denied=s["denied"],
    )


async def run_relife_turn(
    prompt: str,
    *,
    workspace: Path,
    memory_client: Any,
    can_use_tool: Any,
    extra_mcp: dict[str, Any] | None = None,
    on_event: OnEvent | None = None,
) -> TurnResult:
    """Run one fresh ReLife agent turn and return its outcome."""
    from claude_agent_sdk import ClaudeSDKClient

    from ..agent import build_options, to_event
    from ..hooks import memory_hooks

    options = build_options(
        cwd=workspace,
        can_use_tool=can_use_tool,
        mcp_servers={**config.default_mcp_servers(memory_client), **(extra_mcp or {})},
        hooks=memory_hooks(memory_client),
    )
    events: list[dict[str, Any]] = []
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for msg in client.receive_response():
                for ev in to_event(msg):
                    events.append(ev)
                    if on_event is not None:
                        on_event(ev)
    except Exception as e:  # noqa: BLE001 — a failed member is an outcome, not a crash
        events.append({"type": "error", "message": str(e)})
    return result_from_events(events)


def run_sync(fn: Callable[[], Awaitable[Any]]) -> Any:
    """Run an async callable to completion from sync code (CrewAI calls agents
    and LLMs synchronously). If this thread already runs a loop — CrewAI's own
    async paths — run it on a fresh thread with its own loop instead of
    nesting."""
    import anyio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return anyio.run(fn)
    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["value"] = anyio.run(fn)
        except BaseException as e:  # noqa: BLE001 — re-raised on the caller's thread
            box["error"] = e

    t = threading.Thread(target=worker, name="relife-crew-turn")
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")
