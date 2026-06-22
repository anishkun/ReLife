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
- **Memory that grows (like a brain)** — facts/skills/workflows recalled automatically
  before each task; relevance **rises with use and fades when ignored**; an automatic
  LLM-free **"sleep" pass** forgets stale notes, merges duplicates, and learns workflows
  from repeated actions; and an opt-in, AI-driven **"dream" pass** (`relife dream`)
  reversibly critiques and tidies memory when you have budget to spare.

Next (future phases): standalone MCP memory server (the seam is already in place), an
always-on daemon + UI, and the outward capabilities (email/calendar/work-items) — gated
by approval.

## Setup

Requires Python ≥ 3.11, Node.js (for MCP servers), and a logged-in Claude Code CLI
(`claude`) on a Max subscription.

```sh
pip install -e .
```

## Usage

```sh
relife do "scaffold a Python CLI that prints the weather for a city"
relife chat
relife build "<spec>"      # large, multi-milestone builds (resumable: --resume)
relife consolidate         # run the LLM-free memory "sleep" pass now
relife dream               # opt-in AI deep review of memory (spends Max budget)
relife memory stats        # what's remembered and what has faded
```

`do`/`chat`/`build` accept `--workspace PATH` (default: `./workspace`) — the directory
the agent works in.

See **`HOW_IT_WORKS.md`** for a friendly, top-to-bottom walkthrough of the whole system.

## Layout

```
relife/
  cli.py        CLI entry (Typer)
  agent.py      builds ClaudeAgentOptions, runs the SDK loop, renders output
  config.py     model, paths, permission mode, MCP servers
  permissions.py allow/ask policy gating every tool call
  hooks.py      auto-recall of memory/skills/workflows before each prompt
  prompts/      system prompt (persona + safety rules) + the REM critic prompt
  memory/       cognitive memory + skills/workflows, exposed as an in-process MCP server
  build/        orchestrated, resumable large builds (decompose → delegate → resume)
data/           runtime db + logs (gitignored)
tests/          77 deterministic tests (no live model calls)
```
