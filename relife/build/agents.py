"""Subagent definitions for orchestrated builds.

The orchestrator delegates each milestone to a fresh ``builder`` subagent via
the Task tool. Running in its own context window is the whole point: the
orchestrator stays small while the builder absorbs the implementation detail.
"""

from __future__ import annotations

from pathlib import Path

from claude_agent_sdk import AgentDefinition

from .. import config

ORCHESTRATOR_PROMPT_FILE = Path(__file__).parent / "prompts" / "orchestrator.md"

_BUILDER_PROMPT = """\
You are a `builder` subagent inside a larger orchestrated build. You implement \
exactly ONE milestone of the project and nothing more.

- Work in the shared workspace (the project the orchestrator is building). Match \
existing conventions, file layout, and interfaces — read what already exists \
before adding to it.
- Implement the milestone you were given, then **verify it**: run the build, run \
the relevant tests, exercise the code. Fix what you broke.
- Stay scoped: don't redesign other milestones or rewrite unrelated code.
- For code and git you act autonomously; for outward-facing actions defer to the \
orchestrator.
- Report back a **concise** summary only: what you built, the key files/entry \
points, and how you verified it (tests run + result). Do NOT paste full file \
contents or a transcript — the orchestrator needs the outcome, not the detail.
"""

# Tools the builder may use. Read/search/edit/shell + browser + memory; it does
# not touch the build-ledger tools (those are the orchestrator's).
_BUILDER_TOOLS = [
    "Read", "Write", "Edit", "MultiEdit", "Glob", "Grep", "LS",
    "Bash", "PowerShell", "BashOutput", "KillShell",
    "WebFetch", "WebSearch", "TodoWrite", "NotebookRead", "NotebookEdit",
]


def build_agents() -> dict[str, AgentDefinition]:
    """The subagents available to the orchestrator (currently just ``builder``)."""
    return {
        "builder": AgentDefinition(
            description=(
                "Implements one milestone of a larger build in the shared "
                "workspace, verifies it with tests, and reports a concise summary. "
                "Delegate each milestone to this agent."
            ),
            prompt=_BUILDER_PROMPT,
            tools=_BUILDER_TOOLS,
            model=config.MODEL,
            effort=config.EFFORT,
        ),
    }
