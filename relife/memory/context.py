"""The recalled-context block: what an agent is shown about a task before it starts.

One builder, two consumers — so every agent sees memory the same way:

- the ``UserPromptSubmit`` hook (``relife/hooks.py``) injects it ahead of each
  prompt for ReLife's own (Claude) agents;
- the ``memory_context`` MCP tool (``tools.py``) returns it on request, for
  agents that have no hook — CrewAI agents on other LLMs, external MCP clients —
  and the crew builder prepends it to a non-Claude agent's task.

The block gathers memories, then skills, then a workflow (that priority order),
drops a block that largely repeats one already kept, and stops at
``RECALL_INJECT_BUDGET`` characters. Recall here **reinforces** what it surfaces
(being shown a memory is using it). A memory from a space other than the
caller's own is labelled ``via <space>``: inherited or shared knowledge stays
visibly someone else's, so it reads as data, not instructions.
"""

from __future__ import annotations

from typing import Any

from .. import config
from ._text import tokenize as _tokens
from .spaces import DEFAULT_SPACE


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


def _own_space(client: Any) -> str:
    scope = getattr(client, "scope", None)
    return getattr(scope, "write", DEFAULT_SPACE)


def build_context(client: Any, query: str) -> str:
    """The recalled-context block for ``query`` ("" when nothing is relevant)."""
    own = _own_space(client)
    # A record without a space (a duck-typed client) counts as the caller's own.
    # Gather candidate blocks in priority order: memory, then skills, then a
    # workflow. Each carries the key text used for cross-section de-duplication.
    candidates: list[tuple[str, str, str]] = []  # (section_label, key_text, rendered)
    for m in client.recall(query, k=5, reinforce=True):
        label = m.kind if getattr(m, "space", own) == own else f"{m.kind} · via {m.space}"
        rendered = f"- [{label}] {m.text}" + (f"  ({m.tags})" if m.tags else "")
        candidates.append(("memory", m.text + " " + m.tags, rendered))
    for s in client.skill_find(query, k=2):
        via = "" if getattr(s, "space", own) == own else f" (via {s.space})"
        candidates.append(
            ("skill", s.name + " " + s.when_to_use, f"### {s.name}{via} — {s.when_to_use}\n{s.body}")
        )
    for w in client.workflow_find(query, k=1):
        via = "" if getattr(w, "space", own) == own else f" (via {w.space})"
        candidates.append(
            ("workflow", w.name + " " + w.when_to_use, f"### {w.name}{via} — {w.when_to_use}\n{w.body}")
        )

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
    return "\n\n".join(sections)
