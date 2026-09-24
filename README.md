# ReLife

A personal agent that acts on the world through MCP servers and **learns over time** —
accumulating facts and reusable skills so it gets better at recurring tasks.

Built on the [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk) (Python),
running on the **Claude Code Max subscription** (no metered API key).

## Status

v1 working end-to-end:

- **CLI** — `relife do "<task>"` (one-shot) and `relife chat` (interactive).
- **Agent loop** — Claude Opus 4.8 with the Claude Code coding preset + ReLife persona,
  full built-in toolset (read/write/edit/bash/web), streaming output.
- **Permissions** — auto-allow reading, in-workspace edits, build/test, and git
  (incl. push); ask before outward/destructive actions. Fails closed when unattended.
- **MCP** — Playwright **browser** server (navigate/read/click) + an in-process
  **memory/skills** server. GitHub via `gh` (build → commit → create repo → push).
- **Email / calendar / files** — the claude.ai **Gmail, Google Calendar and Google Drive**
  connectors ride the subscription (no keys, no setup beyond enabling them at claude.ai).
  Reads run on their own; anything that sends or changes something asks you first, in the
  terminal or the web console. `relife doctor` shows whether they're enabled.
- **Memory that grows (like a brain)** — facts/skills/workflows recalled automatically
  before each task; relevance **rises with use and fades when ignored**; an automatic
  LLM-free **"sleep" pass** forgets stale notes, merges duplicates, and learns workflows
  from repeated actions; and an opt-in, AI-driven **"dream" pass** (`relife dream`)
  reversibly critiques and tidies memory when you have budget to spare. What it learned is
  inspectable and correctable (`relife memory search|list|show|forget`).
- **Always-on** — `relife serve` runs a long-lived agent server with a self-contained web
  console (streaming transcript, approval cards for outward actions) and **schedules**:
  tasks that fire on a cadence in their own session, with each run's outcome — summary,
  cost, and anything it needed you for — recorded durably.
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
```

Use the interpreter ReLife is installed into for everything below (if plain `python` lacks
`claude_agent_sdk`, try `py -3`).

## Usage

```sh
relife do "scaffold a Python CLI that prints the weather for a city"
relife chat
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
tests/          252 deterministic tests (no live model calls)
```
