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

# Fused recall score = weighted sum of the four signals.
W_SEM = 0.45               # semantic similarity (local embeddings)
W_KW = 0.30                # keyword overlap (FTS5 / tokens)
W_ACT = 0.15               # cognitive activation (decay/reinforcement)
W_IMP = 0.10               # explicit importance

# Stage-1 candidate generation (keeps recall cheap on large stores).
CANDIDATE_TOPN = 50        # max candidates pulled before fuse-ranking
SEM_CANDIDATE_THRESHOLD = 0.60  # a zero-keyword row is a candidate only above this

# Embeddings — a LOCAL ONNX model (no API key, runs offline). Soft-optional:
# if unavailable, recall degrades to keyword + activation.
EMBED_MODEL = os.environ.get("RELIFE_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDINGS_ENABLED = os.environ.get("RELIFE_EMBEDDINGS", "auto")  # auto|on|off

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
