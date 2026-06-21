"""Central configuration: paths, model, and (later) permission allowlists.

Kept deliberately small. Anything that needs tuning as ReLife grows lives here
so the rest of the code reads from one place.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# --- Model -----------------------------------------------------------------
# Runs on the Claude Code Max subscription (the logged-in `claude` CLI), NOT a
# metered API key. See memory: auth-via-max-subscription.
MODEL = "claude-opus-4-8"
EFFORT = "high"  # low | medium | high | xhigh | max  — agentic work wants high+

# --- Paths -----------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"          # gitignored: db + logs
BUILDS_DIR = DATA_DIR / "builds"          # one subdir per orchestrated `relife build`
SKILLS_DIR = DATA_DIR / "skills"          # one Markdown file per learned skill
WORKFLOWS_DIR = DATA_DIR / "workflows"    # one Markdown file per learned workflow
PROMPTS_DIR = PACKAGE_DIR / "prompts"
SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system.md"


# --- Cognitive memory model ------------------------------------------------
# ReLife's memory behaves like a brain: each item's relevance rises when it is
# used (reinforcement) and decays when ignored (forgetting). These tunables
# control that curve and how recall fuses meaning + keywords + activation.
#
# Activation (ACT-R-inspired):
#   activation = ln(1 + use_count) - DECAY*ln(1 + age_days(last_used))
#                + IMPORTANCE_BOOST*importance
DECAY = float(os.environ.get("RELIFE_DECAY", "0.35"))          # forgetting rate
IMPORTANCE_BOOST = 1.5     # how strongly explicit importance lifts activation

# Forgetting (archival) — applied by the consolidation sweep, never by recall.
FORGET_THRESHOLD = 0.20    # archive when activation falls below this …
MIN_FORGET_AGE_DAYS = 14.0 # … and the item has been idle at least this long …
PIN_THRESHOLD = 0.80       # … and importance is under this (>= is "pinned").

# Default importance by kind, used when the caller doesn't specify one. Durable
# kinds (preferences, learned patterns) start more salient so they resist fading;
# episodes start lower because most task outcomes are transient. An explicit
# `importance` argument always overrides these.
DEFAULT_IMPORTANCE = {
    "preference": 0.7,
    "pattern": 0.65,
    "fact": 0.5,
    "episode": 0.45,
}

# Hard-forgetting (deletion) — a second, slower tier. An already-archived memory
# that has stayed idle this long is permanently deleted by consolidation, so the
# store doesn't accumulate dead rows forever. Preferences and pinned items are
# exempt (they are never archived in the first place).
HARD_DELETE_AGE_DAYS = 90.0

# Fused recall score = weighted sum of the four signals.
# NOTE: `importance` deliberately influences recall through TWO paths — it lifts
# `activation()` (so salient memories resist decay) AND contributes directly via
# W_IMP. This double counting is intentional: importance should both slow
# forgetting and act as a standalone relevance signal.
W_SEM = 0.45               # semantic similarity (local embeddings)
W_KW = 0.30                # keyword overlap (FTS5 / tokens)
W_ACT = 0.15               # cognitive activation (decay/reinforcement)
W_IMP = 0.10               # explicit importance

# Gentle per-kind prior added to the fused score: stable kinds (preferences,
# learned patterns) rank slightly higher than transient ones at equal evidence,
# mirroring how durable knowledge stays more accessible than one-off episodes.
KIND_RECALL_BOOST = {"preference": 0.05, "pattern": 0.02}

# Absolute relevance floor: a candidate must clear this fused score to surface.
# Keeps weak (esp. semantic-only) matches out of recall; keyword hits clear it
# easily. The candidate gate already drops fully unrelated rows; this trims the
# long tail of barely-relevant ones.
RECALL_FLOOR = float(os.environ.get("RELIFE_RECALL_FLOOR", "0.12"))

# Max characters of recalled context the auto-recall hook injects per prompt, so
# memory never floods the agent's window. Sections are added highest-priority
# first (memory → skills → workflow) until the budget is reached.
RECALL_INJECT_BUDGET = int(os.environ.get("RELIFE_RECALL_INJECT_BUDGET", "2400"))
# Two recalled blocks whose token Jaccard is at/above this are treated as saying
# the same thing; the later (lower-priority) one is dropped.
RECALL_DEDUP_JACCARD = 0.8

# Stage-1 candidate generation (keeps recall cheap on large stores).
CANDIDATE_TOPN = 50        # max candidates pulled before fuse-ranking
SEM_CANDIDATE_THRESHOLD = 0.60  # a zero-keyword row is a candidate only above this

# Semantic de-duplication thresholds (only used when embeddings are available).
# Calibrated for the local bge-small model: near-identical rewordings score
# ~0.95, genuinely distinct-but-related facts ~0.75, unrelated ~0.4 — so a
# threshold in the high-0.8s/low-0.9s catches paraphrases without false merges.
DEDUP_SIM = 0.90           # consolidation merges active memories above this cosine
SAVE_DEDUP_SIM = 0.93      # save reinforces an existing near-duplicate above this

# Embeddings — a LOCAL ONNX model (no API key, runs offline). Soft-optional:
# if unavailable, recall degrades to keyword + activation.
EMBED_MODEL = os.environ.get("RELIFE_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDINGS_ENABLED = os.environ.get("RELIFE_EMBEDDINGS", "auto")  # auto|on|off

# Episodic capture: on Stop, record a deterministic episode (task intent + tool
# approach) when a run did at least this many tool calls — so genuinely
# multi-step tasks accrue episodic memory even when the agent never calls
# memory_save itself. These feed the recurring-episode → pattern detector.
EPISODE_MIN_EVENTS = int(os.environ.get("RELIFE_EPISODE_MIN_EVENTS", "3"))

# Consolidation ("sleep" pass).
RECUR_THRESHOLD = 3        # a signature must repeat this many times to be a pattern
AUTO_CONSOLIDATE = os.environ.get("RELIFE_AUTO_CONSOLIDATE", "1") != "0"
CONSOLIDATE_EVERY = 5      # auto-run after this many new episodes/events

# Default place the agent builds projects, unless --workspace overrides it.
DEFAULT_WORKSPACE = PROJECT_ROOT / "workspace"


def ensure_dirs() -> None:
    """Create runtime directories that must exist before a run."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BUILDS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_WORKSPACE.mkdir(parents=True, exist_ok=True)


# --- Subprocess environment ------------------------------------------------
# Candidate install dirs for tools that may not be on PATH yet (e.g. gh just
# installed via winget — new on PATH only for fresh shells).
_EXTRA_PATH_DIRS = [
    r"C:\Program Files\GitHub CLI",
]


def agent_env() -> dict[str, str]:
    """Environment overrides handed to the agent subprocess.

    Ensures CLIs we rely on (currently `gh`) are reachable even if the parent
    shell predates their installation. Only prepends dirs that exist and only
    when the tool isn't already resolvable.
    """
    env: dict[str, str] = {}
    if shutil.which("gh") is None:
        existing = os.environ.get("PATH", "")
        prepend = [d for d in _EXTRA_PATH_DIRS if Path(d, "gh.exe").exists()]
        if prepend:
            env["PATH"] = os.pathsep.join([*prepend, existing])
    return env


# --- MCP servers -----------------------------------------------------------
def default_mcp_servers() -> dict:
    """MCP servers attached to every run.

    'browser' = Microsoft's Playwright MCP (navigate/read/click/fill). Launched
    on demand via npx; first run downloads the package and a Chromium build.
    Tools surface to the agent as ``mcp__browser__*`` and are allowed by the
    permission policy (browsing is a core v1 capability).
    """
    # Lazy import: server.py imports the SDK; keep config import-light.
    from .memory.server import memory_server

    return {
        "browser": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@playwright/mcp@latest"],
        },
        "relife_memory": memory_server(),
    }


# --- Permission mode -------------------------------------------------------
# 'default' routes tool-permission decisions to our `can_use_tool` policy
# (see relife/permissions.py): auto-allow code/git/workspace edits, always-ask
# for outward-facing or out-of-workspace actions.
DEFAULT_PERMISSION_MODE = os.environ.get("RELIFE_PERMISSION_MODE", "default")
