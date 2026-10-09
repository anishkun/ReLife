# ReLife

A personal agent that acts on the world through MCP servers and **learns over time** —
accumulating facts and reusable skills so it gets better at recurring tasks.

Built on the [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk) (Python),
running on the **Claude Code Max subscription** (no metered API key).

## Status

**1.0** — release-tested end-to-end (see `RELEASE_TESTING.md` and `CHANGELOG.md`):

- **CLI** — `relife do "<task>"` (one-shot) and `relife chat` (interactive).
- **GitHub work items** — `relife work` lists the issues assigned to you; `relife work owner/repo#12`
  clones, branches, fixes, tests and pushes, then asks before opening the PR. Issue text is treated
  as untrusted data (a release test planted a prompt injection in an issue; it was ignored).
- **Agent loop** — Claude Opus 4.8 with the Claude Code coding preset + ReLife persona,
  full built-in toolset (read/write/edit/bash/web), streaming output.
- **Permissions** — auto-allow reading, in-workspace edits, build/test, and git
  (incl. push); ask before outward/destructive actions, writes or deletes outside the
  workspace, and package installs into your global environment (use a project `.venv`).
  Fails closed when unattended. This is a policy against *accidental* outward actions,
  not a sandbox — code the agent writes and runs can do anything your user account can.
- **MCP** — Playwright **browser** server (navigate/read/click) + an in-process
  **memory/skills** server. GitHub via `gh` (build → commit → create repo → push).
- **Email / calendar / files** — the claude.ai **Gmail, Google Calendar and Google Drive**
  connectors ride the subscription (no keys, no setup beyond enabling them at claude.ai).
  Reads run on their own; anything that changes something asks you first, in the
  terminal or the web console. `relife doctor` shows whether they're enabled. The Gmail
  connector can send, reply and forward as well as draft — every one of those asks unless a
  schedule pre-approved it — and what actually goes through also depends on the Google
  permissions you granted when linking the account.
- **Memory that grows (like a brain)** — facts/skills/workflows recalled automatically
  before each task; relevance **rises with use and fades when ignored**; an automatic
  LLM-free **"sleep" pass** forgets stale notes, merges duplicates, and learns workflows
  from repeated actions; and an opt-in, AI-driven **"dream" pass** (`relife dream`)
  reversibly critiques and tidies memory when you have budget to spare. What it learned is
  inspectable and correctable (`relife memory search|list|show|forget`).
- **Always-on** — `relife serve` runs a long-lived agent server with a self-contained web
  console (streaming transcript, approval cards for outward actions) and **schedules**:
  tasks that fire on a cadence in their own session, with each run's outcome — summary,
  cost, and anything it needed you for — recorded durably. A **work schedule** runs
  `relife work` on your newest assigned issue each time; a schedule may carry narrow
  pre-approvals (email / create calendar events to listed addresses, or open the
  one PR for the issue it worked) so an unattended run can finish.
- **Memory as a service (optional)** — `relife memory serve` runs the memory layer as a
  standalone daemon so several ReLife processes share one brain (`RELIFE_MEMORY_URL`).
- **`relife doctor`** — checks the CLI login, Node, `gh`, FTS5, extras, connectors, the memory
  daemon, the agent server and whether your schedules will actually fire.

## Setup

Requires Python ≥ 3.11, Node.js (for MCP servers), and a logged-in Claude Code CLI
(`claude`) on a Max subscription.

```sh
pip install -e .                 # core
pip install -e ".[server]"       # + `relife serve` (agent server + web console)
pip install -e ".[daemon]"       # + `relife memory serve` (memory as a standalone daemon)
pip install -e ".[embeddings]"   # + local semantic recall (fastembed, offline, no API key)
pip install -e ".[vector]"       # + sqlite-vec index for very large memory stores
```

Where state lives: in a source checkout, `data/` (memory, schedules, run records) and
`workspace/` sit next to the code; an installed (non-editable) copy uses `~/.relife/`.
Set `RELIFE_HOME` to put them anywhere else.

The web console binds `127.0.0.1` and, without a token, only answers requests addressed to
`localhost`/`127.0.0.1`. Set `RELIFE_AGENT_TOKEN` to require a token (mandatory for any
non-loopback bind — the server refuses otherwise). There is no TLS: put a reverse proxy in
front before exposing it beyond your machine.

Use the interpreter ReLife is installed into for everything below (if plain `python` lacks
`claude_agent_sdk`, try `py -3`).

## Usage

```sh
relife do "scaffold a Python CLI that prints the weather for a city"
relife chat
relife work                # your open assigned GitHub issues; `relife work owner/repo#12` to work one
relife build "<spec>"      # large, multi-milestone builds (resumable: --resume)
relife serve               # always-on agent server + web console on http://127.0.0.1:8600
relife doctor              # check the environment first (CLI login, node, gh, connectors, server…)
relife consolidate         # run the LLM-free memory "sleep" pass now
relife dream               # opt-in AI deep review of memory (spends Max budget)
relife memory stats        # what's remembered and what has faded
relife memory search "<q>" # what the agent would be shown for a query (doesn't reinforce)
relife memory list         # everything it has learned (--kind, --archived, --sort strong)
relife memory show <id>    # one memory in full
relife memory forget <id>  # archive a wrong or stale memory (reversible; --query works too)
relife memory serve        # optional: run memory as a standalone daemon (then set RELIFE_MEMORY_URL)
relife memory ping         # is the daemon up?
```

`do`/`chat`/`build` accept `--workspace PATH` (default: `./workspace`) — the directory
the agent works in.

See **`HOW_IT_WORKS.md`** for a friendly, top-to-bottom walkthrough of the whole system.

## Layout

```
relife/
  cli.py        CLI entry (Typer): do / chat / build / serve / doctor / consolidate / dream / memory *
  agent.py      builds ClaudeAgentOptions, runs the SDK loop, the shared event taxonomy
  config.py     model, paths, permission mode, MCP servers, every tunable and env knob
  permissions.py allow/ask policy gating every tool call (shell, files, connectors)
  hooks.py      auto-recall before each prompt, tool journaling, episode capture on Stop
  doctor.py     `relife doctor`: environment + always-on checks, each with a fix
  prompts/      system prompt (persona + safety rules) + the REM critic prompt
  web/          the self-contained web console served by `relife serve`
  memory/       cognitive memory + skills/workflows, exposed as an in-process MCP server
    remote/     the optional memory daemon (FastAPI) + its HTTP client
  build/        orchestrated, resumable large builds (decompose → delegate → resume)
  server/       always-on agent server: sessions, SSE, approvals, security, scheduler, run outcomes
data/           runtime db, schedules, run outcomes, logs (gitignored)
tests/          648 tests (no live model calls; CI runs them on Windows + Linux, py3.11-3.13)
```
