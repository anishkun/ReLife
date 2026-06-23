"""Consolidation — ReLife's "sleep" pass.

Like a brain consolidating the day's experience, this runs periodically (after
runs, or via ``relife consolidate``) and does four deterministic things:

1. **Decay & forget** — archive memories whose activation has faded below the
   forgetting threshold (see ``cognitive.should_archive``). Finished, unused work
   quietly drops out of recall; preferences and important facts stay.
2. **Dedupe** — merge near-duplicate memories, summing their use_count so the
   surviving copy is appropriately strong.
3. **Detect patterns** — find recurring task *episodes* and recurring *tool
   sequences* (from the event log) and record them as ``pattern`` memories.
4. **Synthesize** — turn a strongly-recurring tool sequence into a reusable
   ``workflow`` automatically, so next time the agent can replay it.

Everything here is deterministic and LLM-free, so it is cheap, safe to run
automatically, and fully unit-testable. (An optional LLM enrichment step to give
synthesized workflows better names/generalization is intentionally deferred — it
would consume Max budget; the deterministic stubs are useful on their own.)
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field

from .. import config
from . import events, store, workflows
from ._text import tokenize as _tokens

_STATE_PATH = config.DATA_DIR / "consolidate_state.json"


@dataclass
class ConsolidationReport:
    archived: int = 0
    deleted: int = 0
    merged: int = 0
    patterns: list[str] = field(default_factory=list)
    workflows_created: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"archived {self.archived}, deleted {self.deleted}, "
            f"merged {self.merged}, patterns {len(self.patterns)}, "
            f"workflows {len(self.workflows_created)}"
        )


# --- throttling state (for automatic per-run consolidation) ----------------
def _read_state() -> dict:
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(state: dict) -> None:
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
    except Exception:
        pass


def should_auto_run() -> bool:
    """True if enough new events have accrued since the last consolidation."""
    if not config.AUTO_CONSOLIDATE:
        return False
    last = _read_state().get("last_event_count", 0)
    return (events.count() - last) >= config.CONSOLIDATE_EVERY


# --- the sweep -------------------------------------------------------------
def _decay_and_archive(now: float, report: ConsolidationReport) -> None:
    from . import cognitive

    # Tier 1: soft-archive faded active memories.
    for m in store.all_memories(include_archived=False):
        if cognitive.should_archive(
            use_count=m.use_count,
            last_used_at=m.last_used_at or m.created_at,
            importance=m.importance,
            kind=m.kind,
            now=now,
        ):
            store.archive(m.id)
            report.archived += 1

    # Tier 2: hard-delete archived memories left idle far longer (bounded store).
    active_ids = {m.id for m in store.all_memories(include_archived=False)}
    for m in store.all_memories(include_archived=True):
        if m.id in active_ids:
            continue  # only already-archived rows are deletion candidates
        if cognitive.should_hard_delete(
            last_used_at=m.last_used_at or m.created_at,
            importance=m.importance,
            kind=m.kind,
            now=now,
        ):
            store.delete(m.id)
            report.deleted += 1


def _dedupe(report: ConsolidationReport) -> None:
    """Merge near-duplicate active memories.

    Two memories are duplicates if their token sets overlap heavily (keyword
    Jaccard >= 0.9) OR — when embeddings are available — their meanings are very
    close (cosine >= ``config.DEDUP_SIM``), which also catches paraphrases that
    share few exact tokens. The survivor is reinforced; the duplicate is dropped.
    """
    from . import embeddings

    mems = [m for m in store.all_memories(include_archived=False)]
    use_sem = embeddings.available()
    vecs: dict[int, list[float] | None] = {}
    if use_sem:
        batch = embeddings.embed([m.text for m in mems]) or []
        vecs = {m.id: v for m, v in zip(mems, batch)}

    seen: list[tuple[set[str], object]] = []
    for m in mems:
        toks = _tokens(m.text)
        if not toks:
            continue
        dup_of = None
        for toks2, keep in seen:
            union = toks | toks2
            kw_dup = bool(union) and len(toks & toks2) / len(union) >= 0.9
            sem_dup = (
                use_sem
                and embeddings.cosine(vecs.get(m.id), vecs.get(keep.id))  # type: ignore[attr-defined]
                >= config.DEDUP_SIM
            )
            if kw_dup or sem_dup:
                dup_of = keep
                break
        if dup_of is None:
            seen.append((toks, m))
        else:
            # Reinforce the survivor, drop the duplicate.
            store.reinforce(dup_of.id)  # type: ignore[attr-defined]
            store.delete(m.id)
            report.merged += 1


def _cluster_episodes(threshold: float = 0.6) -> list[list[object]]:
    """Greedy clusters of episodes that describe the same kind of task."""
    eps = [m for m in store.all_memories(include_archived=True) if m.kind == "episode"]
    clusters: list[list[object]] = []
    cluster_toks: list[set[str]] = []
    for m in eps:
        toks = _tokens(m.text)
        if not toks:
            continue
        placed = False
        for i, ctoks in enumerate(cluster_toks):
            union = toks | ctoks
            if union and len(toks & ctoks) / len(union) >= threshold:
                clusters[i].append(m)
                cluster_toks[i] = ctoks | toks
                placed = True
                break
        if not placed:
            clusters.append([m])
            cluster_toks.append(set(toks))
    return clusters


def _short_tool(tool: str) -> str:
    """Human-friendly short name for a (possibly MCP-namespaced) tool."""
    name = tool.split("__")[-1] if "__" in tool else tool
    return name


_SHELL_TOOLS = {"Bash", "PowerShell"}

# Labels that carry no *procedural* meaning on their own — mechanical file edits
# and task-tracking. A sequence made only of these is a real regularity but not a
# reusable workflow (e.g. Write→Edit), so it is not promoted.
_LOW_SIGNAL_LABELS = {
    "Read", "Write", "Edit", "MultiEdit", "NotebookEdit", "NotebookRead",
    "Glob", "Grep", "LS", "TodoWrite", "TaskCreate", "TaskUpdate",
    "BashOutput", "KillShell", "KillBash", "shell",
}


def _action_label(tool: str, brief: str = "") -> str:
    """Coarse *action* a tool call performed — the unit recurring-sequence
    detection should work over.

    Raw tool names are too coarse for shell tools: three distinct Bash commands
    (git clone, mvn test, git push) would all read as "Bash" and collapse into a
    single node, hiding the real workflow while leaving trivial editor motions as
    the only visible n-grams. So for shell tools we derive the action from the
    command (``brief``); other tools keep their short name.
    """
    name = tool.split("__")[-1] if "__" in tool else tool
    if name not in _SHELL_TOOLS:
        return name
    cmd = (brief or "").strip().lower()
    if not cmd:
        return "shell"
    if "git clone" in cmd:
        return "git-clone"
    if "git push" in cmd:
        return "git-push"
    if "git commit" in cmd:
        return "git-commit"
    if "git checkout -b" in cmd or "git switch -c" in cmd or "git branch" in cmd:
        return "git-branch"
    if cmd.startswith("git "):
        return "git"
    if cmd.startswith("gh "):
        return "gh"
    if "docker" in cmd:
        return "docker"
    if (
        "pytest" in cmd or "go test" in cmd or "npm test" in cmd
        or "gradle test" in cmd or "jest" in cmd
        or ("mvn" in cmd and ("test" in cmd or "verify" in cmd))
    ):
        return "test"
    if (
        ("mvn" in cmd and ("package" in cmd or "install" in cmd or "compile" in cmd))
        or "npm run build" in cmd or "gradle build" in cmd or "go build" in cmd
        or cmd.startswith("make")
    ):
        return "build"
    if "pip install" in cmd or "npm install" in cmd or "npm ci" in cmd or "poetry install" in cmd:
        return "install"
    return "shell"


def _is_meaningful_seq(seq: tuple[str, ...]) -> bool:
    """A recurring sequence is workflow-worthy only if it crosses beyond
    mechanical editing/task-tracking — i.e. contains at least one distinctive
    action (git/test/build/docker, an MCP tool, browsing, …)."""
    return any(label not in _LOW_SIGNAL_LABELS for label in seq)


def _contains(longer: tuple, sub: tuple) -> bool:
    """Whether ``sub`` is a contiguous subsequence of ``longer``."""
    if len(sub) >= len(longer):
        return False
    return any(longer[i : i + len(sub)] == sub for i in range(len(longer) - len(sub) + 1))


def _tool_ngrams(sizes=(2, 3, 4)) -> Counter:
    """Count recurring *action* sequences across tasks (consecutive dups
    collapsed). Works over action labels, not raw tool names, so e.g.
    git-clone→test→git-push is visible instead of collapsing into one "Bash"."""
    counts: Counter = Counter()
    for task_id, evs in events.events_by_task().items():
        if not task_id:
            continue  # only sequences that belong to a known task
        seq: list[str] = []
        for e in evs:
            label = _action_label(e.tool, e.brief)
            if not seq or seq[-1] != label:
                seq.append(label)
        for n in sizes:
            for i in range(len(seq) - n + 1):
                counts[tuple(seq[i : i + n])] += 1
    return counts


def _detect_patterns(report: ConsolidationReport) -> None:
    # Recurring episodes → pattern memory.
    for cluster in _cluster_episodes():
        if len(cluster) >= config.RECUR_THRESHOLD:
            common = sorted(set.intersection(*[_tokens(m.text) for m in cluster]))[:8]
            if not common:
                continue
            desc = (
                f"Recurring task pattern (seen {len(cluster)}x): "
                f"{' '.join(common)}"
            )
            store.save(desc, kind="pattern", tags=",".join(common), importance=0.7)
            report.patterns.append(desc)

    # Recurring tool sequences → pattern memory + a synthesized workflow.
    ngrams = _tool_ngrams()
    accepted: list[tuple[str, ...]] = []
    # Longest first so a full sequence wins over its sub-sequences.
    for seq, n in sorted(ngrams.items(), key=lambda x: (-len(x[0]), -x[1])):
        if n < config.RECUR_THRESHOLD:
            continue
        # Only promote sequences that represent a real procedure — pure
        # editor/task-tracking motions (Write→Edit) are skipped, not turned into
        # noise workflows.
        if not _is_meaningful_seq(seq):
            continue
        # Skip a sequence already contained in a longer accepted one (noise).
        if any(_contains(longer, seq) for longer in accepted):
            continue
        accepted.append(seq)
        shorts = [_short_tool(t) for t in seq]
        name = "auto-" + "-".join(shorts).lower()
        desc = f"Recurring tool sequence (seen {n}x): {' → '.join(shorts)}"
        store.save(desc, kind="pattern", tags=",".join(shorts), importance=0.7)
        report.patterns.append(desc)
        if workflows.read_workflow(name) is None:
            steps = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(seq))
            workflows.write_workflow(
                name=name,
                when_to_use=(
                    "A recurring multi-step sequence ReLife detected itself "
                    f"(observed {n} times). Replay these steps when the task "
                    "matches."
                ),
                steps=steps,
                trigger=",".join(shorts),
            )
            report.workflows_created.append(name)


def run_consolidation(now: float | None = None) -> ConsolidationReport:
    """Run the full deterministic consolidation pass and return a report."""
    now = time.time() if now is None else now
    report = ConsolidationReport()
    store.init_db()
    _decay_and_archive(now, report)
    _dedupe(report)
    _detect_patterns(report)
    _write_state({"last_event_count": events.count(), "last_run": now})
    return report
