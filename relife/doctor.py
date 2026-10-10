"""``relife doctor`` — check the environment before the first (or next) run.

ReLife leans on things outside the package: the Claude Code CLI and its login,
Node for the browser MCP, ``gh`` for GitHub, SQLite with FTS5, a few optional
extras, and the claude.ai connectors (Gmail/Calendar/Drive) for outward work.
Each of those fails *late* and cryptically when missing — mid-run, inside a
subprocess, behind the SDK. This module checks them up front and says what to do.

It also looks at the always-on side, which fails *silently* rather than late:
schedules only fire while ``relife serve`` is running, a scheduled run that
needed an approval nobody was there to give just records a denial, and a
non-loopback bind without a token is refused at start. ``doctor`` surfaces
each of those as a check with a fix.

Structure follows the repo's policy modules: :func:`run_checks` is pure over an
injected :class:`Probes` bundle (no subprocess, filesystem, or network of its own),
so the whole matrix is unit-tested with fakes; :func:`default_probes` is the one
place that touches the real machine.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from . import config

Status = str  # "ok" | "warn" | "fail" | "skip"


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    fix: str = ""


@dataclass
class Probes:
    """Everything :func:`run_checks` needs to know about the machine.

    Every field is a value or a callable so tests can script any environment.
    """

    which: Callable[[str], str | None]
    run: Callable[[list[str]], tuple[int, str]]  # → (returncode, combined output)
    env: Mapping[str, str]
    python_version: tuple[int, int, int]
    import_ok: Callable[[str], bool]
    bundled_cli: Path | None
    fts5_ok: Callable[[], bool]
    data_dir: Path
    data_dir_writable: Callable[[Path], bool]
    memory_url: str | None
    http_get: Callable[[str], tuple[int, str]]  # → (status, body); raises on no connection
    extra_path_dirs: list[str] = field(default_factory=list)
    # --- always-on side (defaults keep older probe bundles valid) ---
    model: str = ""
    effort: str = ""
    agent_host: str = "127.0.0.1"
    agent_port: int = 8600
    agent_token: str | None = None
    workspace_root: Path | None = None
    schedules: list[dict] = field(default_factory=list)  # raw records from data/schedules.json
    schedules_problem: str | None = None  # the store couldn't fully read the file
    sidecar_exists: bool = False
    # --- agents / crews (defaults keep older probe bundles valid) ---
    agents: list[dict] = field(default_factory=list)  # public agent records
    agents_problem: str | None = None  # the registry couldn't fully read agents.json


# --- the real machine --------------------------------------------------------
def default_probes() -> Probes:
    import importlib.util
    import os
    import shutil
    import sqlite3
    import subprocess

    def run(cmd: list[str]) -> tuple[int, str]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        except (OSError, subprocess.TimeoutExpired) as e:  # noqa: PERF203
            return 127, str(e)
        return p.returncode, (p.stdout or "") + (p.stderr or "")

    def bundled_cli() -> Path | None:
        # The SDK ships its own CLI and prefers it; mirror its lookup so we
        # report on the binary that will actually run.
        try:
            import claude_agent_sdk

            name = "claude.exe" if sys.platform == "win32" else "claude"
            p = Path(claude_agent_sdk.__file__).parent / "_bundled" / name
            return p if p.is_file() else None
        except Exception:  # noqa: BLE001
            return None

    def fts5_ok() -> bool:
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
            return True
        except sqlite3.OperationalError:
            return False

    def writable(p: Path) -> bool:
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".doctor-write-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return True
        except OSError:
            return False

    def http_get(url: str) -> tuple[int, str]:
        import urllib.request

        with urllib.request.urlopen(url, timeout=3) as r:  # noqa: S310
            return r.status, r.read().decode("utf-8", "replace")

    def schedules() -> tuple[list[dict], str | None]:
        # Raw records, read the same way the server does (no fastapi needed).
        try:
            from .server.schedules import ScheduleStore

            store = ScheduleStore(config.AGENT_SCHEDULES_PATH)
            return [s.to_dict() for s in store.list()], store.problem
        except Exception as e:  # noqa: BLE001
            return [], f"could not load ({e})"

    schedule_records, schedule_problem = schedules()

    def agents() -> tuple[list[dict], str | None]:
        try:
            from .agents import AgentStore

            store = AgentStore(config.AGENTS_PATH)
            return [a.public() for a in store.list()], store.problem
        except Exception as e:  # noqa: BLE001
            return [], f"could not load ({e})"

    agent_records, agents_problem = agents()

    return Probes(
        which=shutil.which,
        run=run,
        env=os.environ,
        python_version=sys.version_info[:3],
        import_ok=lambda mod: importlib.util.find_spec(mod) is not None,
        bundled_cli=bundled_cli(),
        fts5_ok=fts5_ok,
        data_dir=config.DATA_DIR,
        data_dir_writable=writable,
        memory_url=config.MEMORY_URL,
        http_get=http_get,
        extra_path_dirs=list(config._EXTRA_PATH_DIRS),
        model=config.MODEL,
        effort=config.EFFORT,
        agent_host=config.AGENT_HOST,
        agent_port=config.AGENT_PORT,
        agent_token=config.AGENT_TOKEN,
        workspace_root=config.AGENT_WORKSPACE_ROOT,
        schedules=schedule_records,
        schedules_problem=schedule_problem,
        sidecar_exists=config.MEMORY_SIDECAR_PATH.exists(),
        agents=agent_records,
        agents_problem=agents_problem,
    )


# --- checks -------------------------------------------------------------------
def find_claude_cli(p: Probes) -> str | None:
    """The CLI the SDK will run: bundled first, then PATH (same order as the SDK)."""
    if p.bundled_cli is not None:
        return str(p.bundled_cli)
    return p.which("claude")


def _check_python(p: Probes) -> Check:
    v = ".".join(map(str, p.python_version))
    if p.python_version >= (3, 11):
        return Check("python", "ok", f"{v}")
    return Check("python", "fail", f"{v} (need ≥ 3.11)", "install Python 3.11+ and reinstall: pip install -e .")


def _check_cli(p: Probes) -> tuple[Check, str | None]:
    cli = find_claude_cli(p)
    if not cli:
        return (
            Check(
                "claude cli", "fail", "not found (bundled or on PATH)",
                "pip install -U claude-agent-sdk  (bundles the CLI), or npm install -g @anthropic-ai/claude-code",
            ),
            None,
        )
    rc, out = p.run([cli, "--version"])
    version = out.strip().splitlines()[0] if out.strip() else "unknown version"
    where = "bundled with the SDK" if p.bundled_cli and cli == str(p.bundled_cli) else cli
    if rc != 0:
        return Check("claude cli", "fail", f"{where}: exits {rc}", "reinstall the Claude Code CLI"), None
    return Check("claude cli", "ok", f"{version} — {where}"), cli


def _check_login(p: Probes, cli: str | None) -> Check:
    if not cli:
        return Check("claude login", "skip", "no CLI to ask")
    rc, out = p.run([cli, "auth", "status"])
    info: dict = {}
    m = re.search(r"\{.*\}", out, re.S)
    if m:
        try:
            info = json.loads(m.group(0))
        except json.JSONDecodeError:
            info = {}
    if info.get("loggedIn"):
        who = info.get("email") or "?"
        sub = info.get("subscriptionType") or "?"
        method = info.get("authMethod") or "?"
        return Check("claude login", "ok", f"{who} · {sub} subscription via {method}")
    return Check(
        "claude login", "fail", "not logged in",
        "run `claude` once and log in (ReLife rides the subscription — no API key)",
    )


def _check_api_key(p: Probes) -> Check:
    if p.env.get("ANTHROPIC_API_KEY"):
        return Check(
            "api key", "warn", "ANTHROPIC_API_KEY is set",
            "unset it — ReLife is meant to run on the subscription, and a key in the env can bill metered usage instead",
        )
    return Check("api key", "ok", "ANTHROPIC_API_KEY unset (subscription auth)")


def _check_node(p: Probes) -> Check:
    node = p.which("node")
    npx = p.which("npx")
    if node and npx:
        rc, out = p.run([node, "--version"])
        return Check("node / npx", "ok", f"{out.strip() or 'node'} (browser MCP via npx)")
    return Check(
        "node / npx", "fail", "not on PATH",
        "install Node.js ≥ 18 — the browser MCP is launched with `npx @playwright/mcp`",
    )


def _check_gh(p: Probes) -> Check:
    gh = p.which("gh")
    if not gh:
        for d in p.extra_path_dirs:
            cand = Path(d) / "gh.exe"
            if cand.exists():
                gh = str(cand)
                break
    if not gh:
        return Check(
            "github cli", "warn", "gh not found",
            "install GitHub CLI and `gh auth login` — needed for `gh repo create`; plain git push still works",
        )
    rc, out = p.run([gh, "auth", "status"])
    if rc == 0:
        acct = re.search(r"account (\S+)", out)
        return Check("github cli", "ok", f"authenticated as {acct.group(1) if acct else '?'}")
    return Check("github cli", "warn", "installed but not authenticated", "gh auth login")


def _check_fts5(p: Probes) -> Check:
    if p.fts5_ok():
        return Check("sqlite fts5", "ok", "keyword recall is indexed")
    return Check(
        "sqlite fts5", "warn", "this SQLite build lacks FTS5",
        "recall falls back to a full keyword scan (fine for small stores); a newer Python usually fixes it",
    )


def _check_data_dir(p: Probes) -> Check:
    if p.data_dir_writable(p.data_dir):
        return Check("data dir", "ok", str(p.data_dir))
    return Check("data dir", "fail", f"cannot write {p.data_dir}", "fix permissions or move the checkout")


_EXTRAS = [
    # (module, extra name, what it unlocks)
    ("fastembed", "embeddings", "semantic recall (local ONNX, no API key)"),
    ("sqlite_vec", "vector", "ANN vector index for large stores"),
    ("fastapi", "server", "`relife serve` web console + memory daemon"),
    ("uvicorn", "server", "`relife serve` web console + memory daemon"),
    ("httpx", "daemon", "client for a remote memory daemon"),
]


def _check_extras(p: Probes) -> list[Check]:
    out: list[Check] = []
    seen: set[str] = set()
    for mod, extra, what in _EXTRAS:
        if extra in seen:
            continue
        seen.add(extra)
        mods = [m for m, e, _ in _EXTRAS if e == extra]
        missing = [m for m in mods if not p.import_ok(m)]
        if not missing:
            out.append(Check(f"extra [{extra}]", "ok", what))
        else:
            out.append(
                Check(
                    f"extra [{extra}]", "skip", f"not installed — {what}",
                    f'pip install -e ".[{extra}]"',
                )
            )
    return out


def _check_crews(p: Probes) -> Check:
    """`relife crew` needs CrewAI, which doesn't support Python 3.14 yet."""
    if p.import_ok("crewai"):
        return Check("crews", "ok", "CrewAI installed — `relife crew` is available")
    if p.python_version[:2] >= (3, 14):
        return Check(
            "crews", "skip",
            f"CrewAI needs Python ≤ 3.13 (this is {'.'.join(map(str, p.python_version))})",
            r'py -3.12 -m venv .venv  then  .venv\Scripts\pip install -e ".[crewai]"',
        )
    return Check("crews", "skip", "CrewAI not installed — `relife crew` unavailable", 'pip install -e ".[crewai]"')


def _check_agents(p: Probes) -> Check:
    if p.agents_problem:
        return Check(
            "agents", "warn", f"agents.json: {p.agents_problem}",
            "fix or remove the unreadable records; `relife agent list` shows what loaded",
        )
    if not p.agents:
        return Check("agents", "skip", "none registered — only the main agent (default memory space)")
    tokens = sum(1 for a in p.agents if a.get("has_token"))
    kinds: dict[str, int] = {}
    for a in p.agents:
        kinds[a.get("runtime", "?")] = kinds.get(a.get("runtime", "?"), 0) + 1
    mix = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
    return Check(
        "agents", "ok",
        f"{len(p.agents)} registered ({mix})" + (f" · {tokens} with an HTTP MCP token" if tokens else ""),
    )


def _check_model(p: Probes) -> Check:
    if not p.model:
        return Check("model", "skip", "unknown")
    src = " (RELIFE_MODEL)" if p.env.get("RELIFE_MODEL") else ""
    return Check("model", "ok", f"{p.model} · effort {p.effort or '?'}{src}")


def _check_memory_daemon(p: Probes) -> Check:
    if not p.memory_url:
        if p.sidecar_exists:
            return Check(
                "memory daemon", "warn",
                "a daemon's sidecar file exists but RELIFE_MEMORY_URL is unset — this process would write the DB around it",
                "set RELIFE_MEMORY_URL=http://127.0.0.1:8787 (or stop the daemon and delete data/relife.db.daemon if it crashed)",
            )
        return Check("memory daemon", "skip", "RELIFE_MEMORY_URL unset — memory runs in-process (default)")
    url = p.memory_url.rstrip("/") + "/health"
    try:
        status, body = p.http_get(url)
    except Exception as e:  # noqa: BLE001
        return Check(
            "memory daemon", "fail", f"unreachable at {p.memory_url}: {e}",
            "start it with `relife memory serve`, or unset RELIFE_MEMORY_URL",
        )
    if status == 200:
        return Check("memory daemon", "ok", f"{p.memory_url} healthy")
    return Check("memory daemon", "fail", f"{url} → HTTP {status}", "check the daemon's logs")


_CONNECTORS = ("Gmail", "Google Calendar", "Google Drive")


def parse_mcp_list(out: str) -> dict[str, str]:
    """``claude mcp list`` lines → {server name: status text}."""
    found: dict[str, str] = {}
    for line in out.splitlines():
        m = re.match(r"^(.+?):\s+\S+\s+-\s+(.+)$", line.strip())
        if m:
            found[m.group(1).strip()] = m.group(2).strip()
    return found


def _check_connectors(p: Probes, cli: str | None, logged_in: bool) -> list[Check]:
    if not cli or not logged_in:
        return [Check("connectors", "skip", "needs a logged-in CLI")]
    rc, out = p.run([cli, "mcp", "list"])
    servers = parse_mcp_list(out)
    checks: list[Check] = []
    for name in _CONNECTORS:
        key = f"claude.ai {name}"
        status = servers.get(key)
        label = f"connector {name.lower()}"
        if status is None:
            checks.append(
                Check(
                    label, "warn", "not enabled on this account",
                    f"enable {name} at claude.ai → Settings → Connectors, then link your Google account "
                    "from a ReLife session (say “connect my " + name.split()[-1].lower() + "”)",
                )
            )
        elif "connected" in status.lower():
            checks.append(
                Check(
                    label, "ok",
                    "enabled — reads run on their own, sends/changes ask for approval "
                    "(Google account linking happens in-session on first use)",
                )
            )
        elif "auth" in status.lower():
            short = name.split()[-1].lower()
            checks.append(
                Check(
                    label, "warn", "enabled, but the Google account isn't linked yet",
                    f"link it from a ReLife session — say “connect my {short}” and approve the browser sign-in "
                    "(reads then run on their own; sends/changes still ask)",
                )
            )
        else:
            checks.append(Check(label, "warn", status, "re-enable the connector at claude.ai → Settings → Connectors"))
    return checks


# --- the always-on side -------------------------------------------------------
_BAD_RUN = re.compile(r"^(error|timeout|interrupted|skipped)")


def _check_workspace_root(p: Probes) -> Check:
    if p.workspace_root is None:
        return Check("workspace root", "skip", "unknown")
    if p.data_dir_writable(p.workspace_root):
        return Check("workspace root", "ok", f"{p.workspace_root} (server sessions are confined here)")
    return Check(
        "workspace root", "fail", f"cannot write {p.workspace_root}",
        "fix permissions or point RELIFE_AGENT_WORKSPACE_ROOT somewhere writable",
    )


def _agent_health_url(p: Probes) -> str:
    host = p.agent_host.strip("[]")
    probe = "127.0.0.1" if host in {"0.0.0.0", "", "::"} else host
    return f"http://{probe}:{p.agent_port}/health"


def _check_agent_server(p: Probes, enabled_schedules: int) -> Check:
    """Bind/token sanity first (serve() refuses a bad bind), then reachability.

    Not running is normal for CLI use — but not when schedules exist: they only
    fire inside the server process, so that case is a warning with the fix.
    """
    from .server.security import guard_bind, is_loopback

    try:
        guard_bind(p.agent_host, p.agent_token)
    except ValueError:
        return Check(
            "agent server", "fail", f"RELIFE_AGENT_HOST={p.agent_host} with no RELIFE_AGENT_TOKEN — serve() will refuse",
            "set RELIFE_AGENT_TOKEN (and put TLS/a reverse proxy in front), or bind 127.0.0.1",
        )
    notes = []
    if p.agent_token and len(p.agent_token) < 16:
        notes.append("token is short — use 16+ random characters")
    if not is_loopback(p.agent_host):
        notes.append("bound beyond loopback with no TLS — only behind a reverse proxy")
    url = _agent_health_url(p)
    try:
        status, body = p.http_get(url)
    except Exception:  # noqa: BLE001 - connection refused = not running
        if enabled_schedules:
            return Check(
                "agent server", "warn",
                f"not running at {p.agent_host}:{p.agent_port} — {enabled_schedules} enabled schedule(s) will not fire",
                "start it: `relife serve` (schedules only run inside the server process)",
            )
        return Check("agent server", "skip", f"not running at {p.agent_host}:{p.agent_port} (start with `relife serve`)")
    if status != 200:
        return Check("agent server", "fail", f"{url} → HTTP {status}", "check the server's logs")
    info: dict = {}
    try:
        info = json.loads(body)
    except (ValueError, TypeError):
        pass
    detail = f"running at {p.agent_host}:{p.agent_port}"
    if info:
        detail += f" · {info.get('sessions', '?')} session(s), {info.get('schedules', '?')} schedule(s)"
        detail += " · auth on" if info.get("auth_required") else " · no token (loopback only)"
    if notes:
        return Check("agent server", "warn", detail + " — " + "; ".join(notes), "see RELIFE_AGENT_TOKEN / RELIFE_AGENT_HOST")
    return Check("agent server", "ok", detail)


def _check_schedules(p: Probes) -> Check:
    """What the schedules did last time, in one line — the panel shows it too,
    but doctor is where someone looks when \"nothing seems to happen\"."""
    if p.schedules_problem:
        return Check(
            "schedules", "warn", f"{config.AGENT_SCHEDULES_PATH.name}: {p.schedules_problem}",
            "fix or restore the file before adding schedules — the next save rewrites it "
            "(a copy of the original is kept beside it as .corrupt-<time>)",
        )
    if not p.schedules:
        return Check("schedules", "skip", "none defined (add one in the web console)")
    enabled = [s for s in p.schedules if s.get("enabled", True)]
    failing = []
    needed = []
    for sc in enabled:
        last = sc.get("runs") or []
        entry = last[-1] if last else {}
        status = str(entry.get("status") or sc.get("last_status") or "")
        if _BAD_RUN.match(status):
            failing.append(f"{sc.get('name')} ({status})")
        if entry.get("denied"):
            needed.append(f"{sc.get('name')} ({entry['denied']} denied)")
    head = f"{len(enabled)} enabled, {len(p.schedules) - len(enabled)} paused"
    if failing or needed:
        parts = []
        if failing:
            parts.append("last run failed: " + ", ".join(failing))
        if needed:
            parts.append("needed you: " + ", ".join(needed))
        return Check(
            "schedules", "warn", head + " — " + "; ".join(parts),
            "open the console → schedules → runs for each one's summary and the exact actions that were denied",
        )
    return Check("schedules", "ok", head + (" — last runs fine" if any(s.get("runs") for s in enabled) else " — none run yet"))


def run_checks(p: Probes) -> list[Check]:
    checks: list[Check] = [_check_python(p)]
    cli_check, cli = _check_cli(p)
    checks.append(cli_check)
    login = _check_login(p, cli)
    checks.append(login)
    checks.append(_check_api_key(p))
    checks.append(_check_model(p))
    checks.append(_check_node(p))
    checks.append(_check_gh(p))
    checks.append(_check_fts5(p))
    checks.append(_check_data_dir(p))
    checks.append(_check_workspace_root(p))
    checks.extend(_check_extras(p))
    checks.append(_check_crews(p))
    checks.append(_check_agents(p))
    checks.append(_check_memory_daemon(p))
    enabled = sum(1 for s in p.schedules if s.get("enabled", True))
    checks.append(_check_agent_server(p, enabled))
    checks.append(_check_schedules(p))
    checks.extend(_check_connectors(p, cli, login.status == "ok"))
    return checks


def to_json(checks: list[Check]) -> dict:
    return {
        "overall": worst(checks),
        "checks": [{"name": c.name, "status": c.status, "detail": c.detail, "fix": c.fix} for c in checks],
    }


def worst(checks: list[Check]) -> Status:
    order = {"ok": 0, "skip": 0, "warn": 1, "fail": 2}
    return max(checks, key=lambda c: order[c.status]).status if checks else "ok"
