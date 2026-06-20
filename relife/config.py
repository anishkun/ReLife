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
PROMPTS_DIR = PACKAGE_DIR / "prompts"
SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system.md"

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
