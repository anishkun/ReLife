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
- **Memory that grows** — facts/preferences recalled automatically before each task;
  reusable **skills** the agent writes after succeeding and reuses later.

Next (future phases): standalone MCP memory server, an always-on daemon + UI, and
the outward capabilities (email/calendar/work-items) — gated by approval.

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
```

Both accept `--workspace PATH` (default: `./workspace`) — the directory the agent
works in.

## Layout

```
relife/
  cli.py        CLI entry (Typer)
  agent.py      builds ClaudeAgentOptions, runs the SDK loop, renders output
  config.py     model, paths, permission mode
  prompts/      system prompt (persona + safety rules)
  memory/       (upcoming) retrieval + skills, exposed as an in-process MCP server
data/           runtime db + logs (gitignored)
```
