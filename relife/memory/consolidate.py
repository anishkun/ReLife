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

Steps 2–4 run **per memory space** (see ``spaces.py``): two agents' memories
are never merged into one another, and a recurring sequence is learned into the
space of the agent that kept doing it — one agent's sweep can never delete or
rewrite what another agent knows.

Everything here is deterministic and LLM-free, so it is cheap, safe to run
automatically, and fully unit-testable. (An optional LLM enrichment step to give
synthesized workflows better names/generalization is intentionally deferred — it
would consume Max budget; the deterministic stubs are useful on their own.)
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field

from .. import config
from . import events, store, workflows
from ._text import tokenize as _tokens
from .spaces import DEFAULT_SPACE

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
    state = _read_state()
    # By event id, not row count: pruning lowers the count, which would stall
    # the throttle. (A pre-pruning state file holds a count; with nothing ever
    # deleted then, the newest id equalled it, so it reads as an id.)
    last = state.get("last_event_id", state.get("last_event_count", 0))
    return (events.max_id() - last) >= config.CONSOLIDATE_EVERY


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


_DEDUP_JACCARD = 0.9


def _valid_mark(mark) -> int:
    """A mark is ``[id, created_at]`` of the newest memory checked. It counts
    only while that memory is still the one with that id — a replaced or
    rebuilt DB restarts ids, and a stale mark would skip new memories."""
    if not (isinstance(mark, list) and len(mark) == 2):
        return 0
    try:
        mid, created = int(mark[0]), float(mark[1])
    except (TypeError, ValueError):
        return 0
    m = store.get(mid)
    return mid if m is not None and abs(m.created_at - created) < 1e-6 else 0


def _dedupe(report: ConsolidationReport, marks: dict | None = None) -> dict:
    """Merge near-duplicate active memories, space by space.

    Two memories are duplicates if their token sets overlap heavily (keyword
    Jaccard >= 0.9) OR — when embeddings are available — their meanings are very
    close (cosine >= ``config.DEDUP_SIM``), which also catches paraphrases that
    share few exact tokens. The survivor (the older one) is reinforced; the
    duplicate is dropped.

    ``marks`` (``{"<mode>:<space>": [id, created_at]}``, from the last pass)
    says up to which memory a space was already deduped: those memories were checked against each other
    then, so only newer ones are compared — against everything. Returns the
    marks for the next pass. Built to scale: the old all-pairs loop took 19 s
    at 3,000 memories and re-embedded every memory on every pass.
    """
    from . import embeddings

    marks = dict(marks or {})
    use_sem = embeddings.available()
    mode = "sem" if use_sem else "kw"
    by_space: dict[str, list] = {}
    for m in store.all_memories(include_archived=False):
        by_space.setdefault(m.space, []).append(m)
    out: dict[str, int] = {}
    for space, mems in by_space.items():
        mems.sort(key=lambda m: m.id)
        # A mark made without embeddings never compared meanings: start over.
        key = f"{mode}:{space}"
        newest = _dedupe_group(mems, report, since=_valid_mark(marks.get(key)), use_sem=use_sem)
        if newest is not None:
            out[key] = [newest.id, newest.created_at]
    return out


def _prefix(toks: list[str]) -> list[str]:
    """Prefix filtering: two sets with Jaccard >= t must share a token among
    each one's first ``|x| - ceil(t·|x|) + 1`` tokens, in one global order."""
    return toks[: len(toks) - math.ceil(_DEDUP_JACCARD * len(toks)) + 1]


def _dedupe_group(mems: list, report: ConsolidationReport, *, since: int = 0, use_sem: bool = False):
    """Dedupe one space's active memories (sorted by id; never across spaces).

    Greedy in id order, as before: a memory is a duplicate of the *oldest*
    surviving memory it matches. Keyword candidates come from an inverted index
    over each survivor's rarest tokens (exact for the Jaccard threshold, not a
    heuristic); meanings are compared as one matrix-vector product per new
    memory over stored embeddings — only memories without one are embedded.
    Returns the newest survivor (the next pass's mark), or None."""
    from . import embeddings

    toks = {m.id: _tokens(m.text) for m in mems}
    df: Counter = Counter(t for ts in toks.values() for t in ts)
    order = {m.id: sorted(toks[m.id], key=lambda t: (df[t], t)) for m in mems}

    vecs: dict[int, list[float]] = {}
    if use_sem:
        vecs = store.stored_embeddings([m.id for m in mems])
        missing = [m for m in mems if m.id not in vecs]
        if missing:
            for m, v in zip(missing, embeddings.embed([m.text for m in missing]) or []):
                if v:
                    vecs[m.id] = v
    sem = _SemIndex(len(mems)) if use_sem else None

    index: dict[str, list] = {}  # token → survivors whose prefix holds it
    newest = None

    def keep(m) -> None:
        nonlocal newest
        newest = m
        for t in _prefix(order[m.id]):
            index.setdefault(t, []).append(m)
        if sem is not None:
            sem.add(m, vecs.get(m.id))

    for m in mems:
        ts = toks[m.id]
        if not ts:
            continue
        if m.id <= since:  # checked last pass: a survivor by definition
            keep(m)
            continue
        dup_of = None
        seen_ids: set[int] = set()
        for t in _prefix(order[m.id]):
            for other in index.get(t, ()):
                if other.id in seen_ids:
                    continue
                seen_ids.add(other.id)
                o = toks[other.id]
                if len(ts & o) / len(ts | o) >= _DEDUP_JACCARD and (dup_of is None or other.id < dup_of.id):
                    dup_of = other
        if sem is not None:
            other = sem.first_match(vecs.get(m.id), config.DEDUP_SIM)
            if other is not None and (dup_of is None or other.id < dup_of.id):
                dup_of = other
        if dup_of is None:
            keep(m)
        else:
            # Reinforce the survivor, drop the duplicate.
            store.reinforce(dup_of.id)
            store.delete(m.id)
            report.merged += 1
    return newest


class _SemIndex:
    """Survivors' unit vectors, searched with one product per query (numpy,
    which fastembed brings), or a plain loop when numpy is absent."""

    def __init__(self, capacity: int) -> None:
        self._items: list = []
        self._vecs: list[list[float]] = []
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - embeddings imply numpy
            self._np = None
        else:
            self._np = np
            self._mat = None
            self._cap = max(capacity, 1)
            self._n = 0

    def add(self, m, vec: list[float] | None) -> None:
        if not vec:
            return
        if self._np is None:
            self._items.append(m)
            self._vecs.append(vec)
            return
        np = self._np
        v = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return
        if self._mat is None:
            self._mat = np.zeros((self._cap, v.shape[0]), dtype=np.float32)
        if v.shape[0] != self._mat.shape[1]:
            return  # a vector from another model; can't be compared
        self._mat[self._n] = v / norm
        self._items.append(m)
        self._n += 1

    def first_match(self, vec: list[float] | None, threshold: float):
        """The oldest survivor at or above ``threshold`` cosine, or None."""
        if not vec or not self._items:
            return None
        if self._np is None:
            from . import embeddings

            for m, v in zip(self._items, self._vecs):
                if embeddings.cosine(vec, v) >= threshold:
                    return m
            return None
        np = self._np
        q = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm == 0.0 or q.shape[0] != self._mat.shape[1]:
            return None
        sims = self._mat[: self._n] @ (q / norm)
        hits = np.nonzero(sims >= threshold)[0]
        return self._items[int(hits[0])] if hits.size else None


def _cluster_episodes(threshold: float = 0.6, space: str = DEFAULT_SPACE) -> list[list[object]]:
    """Greedy clusters of ``space``'s episodes that describe the same kind of task."""
    eps = [
        m
        for m in store.all_memories(include_archived=True, spaces=(space,))
        if m.kind == "episode"
    ]
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
    # The generic edit → test → commit loop is in nearly every coding task, so
    # a sequence made only of it says nothing reusable — and its labels collide
    # with ordinary prompt words ("write", "test"), dragging the workflow into
    # unrelated tasks' recall. Same for the agent's own bookkeeping tools.
    "test", "git", "git-commit", "ToolSearch", "Task", "Agent",
    "build_status", "build_milestone_update", "build_plan_set",
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


def _tool_ngrams(sizes=(2, 3, 4), space: str = DEFAULT_SPACE, by_task=None) -> Counter:
    """Count recurring *action* sequences across ``space``'s tasks (consecutive
    dups collapsed). Works over action labels, not raw tool names, so e.g.
    git-clone→test→git-push is visible instead of collapsing into one "Bash"."""
    counts: Counter = Counter()
    if by_task is None:
        by_task = events.events_by_task(space=space)
    for task_id, evs in by_task.items():
        if not task_id:
            continue  # only sequences that belong to a known task
        seq: list[str] = []
        for e in evs:
            if e.space != space:
                continue
            label = _action_label(e.tool, e.brief)
            if not seq or seq[-1] != label:
                seq.append(label)
        for n in sizes:
            for i in range(len(seq) - n + 1):
                counts[tuple(seq[i : i + n])] += 1
    return counts


def _detect_patterns(report: ConsolidationReport) -> None:
    """Detect patterns space by space, learning each into its own space."""
    spaces = {m.space for m in store.all_memories(include_archived=True)}
    spaces |= events.spaces()
    for space in sorted(spaces or {DEFAULT_SPACE}):
        # Each space mines its own recent window (see events.recent_events).
        _detect_patterns_in(space, events.events_by_task(space=space), report)


def _detect_patterns_in(space: str, by_task, report: ConsolidationReport) -> None:
    # Report lines name the space unless it's the default one (the main agent).
    label = "" if space == DEFAULT_SPACE else f"[{space}] "

    # Recurring episodes → pattern memory.
    for cluster in _cluster_episodes(space=space):
        if len(cluster) >= config.RECUR_THRESHOLD:
            common = sorted(set.intersection(*[_tokens(m.text) for m in cluster]))[:8]
            if not common:
                continue
            desc = (
                f"Recurring task pattern (seen {len(cluster)}x): "
                f"{' '.join(common)}"
            )
            store.save(desc, kind="pattern", tags=",".join(common), importance=0.7, space=space)
            report.patterns.append(label + desc)

    # Recurring tool sequences → pattern memory + a synthesized workflow.
    ngrams = _tool_ngrams(space=space, by_task=by_task)
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
        store.save(desc, kind="pattern", tags=",".join(shorts), importance=0.7, space=space)
        report.patterns.append(label + desc)
        if workflows.read_workflow(name, space) is None:
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
                space=space,
            )
            report.workflows_created.append(label + name)


def run_consolidation(now: float | None = None) -> ConsolidationReport:
    """Run the full deterministic consolidation pass and return a report."""
    now = time.time() if now is None else now
    report = ConsolidationReport()
    store.init_db()
    _decay_and_archive(now, report)
    marks = _dedupe(report, _read_state().get("dedupe_marks"))
    _detect_patterns(report)
    events.prune(config.EVENTS_KEEP)
    _write_state({"last_event_id": events.max_id(), "last_run": now, "dedupe_marks": marks})
    return report
