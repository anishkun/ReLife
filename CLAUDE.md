# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ReLife is a personal agent built on the **Claude Agent SDK** (Python) that acts through MCP servers and **learns over time** (facts + reusable skills). It runs on the **Claude Code Max subscription**, NOT a metered API key — `ANTHROPIC_API_KEY` is intentionally unset and the SDK drives the logged-in `claude` CLI. Do not introduce code that requires an API key. See `PROJECT_CONTEXT.md` for the full design rationale, locked decisions, and current status — read it before making architectural changes.

## Commands

```sh
pip install -e .                  # install (editable); creates the `relife` entry point
python -m pytest tests/           # run all tests (17, deterministic — no live agent calls)
python -m pytest tests/test_permissions.py::test_name -v   # single test

relife do "<task>"                # one-shot: run a task to completion
relife chat                       # interactive multi-turn session
# both accept --workspace/-w PATH (default ./workspace) — the dir the agent works in
```

Prereqs to actually *run* the agent (not needed for tests): Python ≥3.11, Node.js (`npx` for the browser MCP), a logged-in `claude` CLI on Max, and authenticated `gh`.

## Architecture

The flow is `cli.py` → `agent.build_options()` → `ClaudeSDKClient` (streaming). Each `do`/`chat` invocation wires together three pluggable pieces, all passed into `build_options`:

- **Permissions** (`permissions.py`) — `classify()` is a pure function returning `("allow"|"ask", reason)`; `make_permission_callback()` wraps it as the SDK `can_use_tool`. Autonomy model: auto-allow read-only tools, file writes **inside the workspace**, shell build/test/git (incl. `git push`), and ReLife's own MCP tools (`mcp__relife*`, `mcp__browser`). Always-ask for outward/destructive actions (the `_OUTWARD_BASH` regex catches email senders, `gh pr|issue|release|api|gist`, uploads, remote shells, package publish, `sudo`, root `rm -rf`), writes outside the workspace, and any unrecognized tool (fail-closed default). Non-interactive runs **deny** ask-cases rather than block.
- **MCP servers** (`config.default_mcp_servers()`) — `browser` (Playwright via `npx @playwright/mcp`) and `relife_memory` (in-process SDK server). Memory is shipped as an MCP server *even though in-process* so the agent-facing contract is unchanged when it's later split out.
- **Memory hooks** (`hooks.py`) — a `UserPromptSubmit` hook auto-injects relevant recalled memories + skills as additional context before each prompt, so the agent benefits without explicitly calling recall.

**Per-task loop:** recall (hook injects memory+skills) → act (tools, gated by `classify`) → reflect (agent calls `memory_save`/`skill_write` for durable lessons).

### Memory layer (`relife/memory/`)
Two retrieval stores, both keyword + recency, **no embeddings** in v1 (the public `save`/`recall` API stays stable when a vector index is added later):
- `store.py` — SQLite facts/preferences/episodes at `data/relife.db`. Exact-duplicate text refreshes recency instead of inserting.
- `skills.py` — reusable procedures as one Markdown-with-frontmatter file each under `data/skills/`. Recall weights name matches 2×.
- `server.py` — exposes `memory_save`/`memory_recall`/`skill_write`/`skill_find` as MCP tools. Server name `relife_memory` → tools surface as `mcp__relife_memory__*` (matched by the trusted `mcp__relife` permission prefix).
- `_text.py` — shared stopword tokenizer used by both stores' recall.

The system prompt uses the **`claude_code` preset** with `prompts/system.md` appended (persona + safety + memory/skill instructions). `setting_sources=None` deliberately prevents inheriting the surrounding repo's Claude Code settings.

## Non-obvious constraints

- **`can_use_tool` requires streaming mode.** Use `ClaudeSDKClient` + `receive_response()`; the one-shot `query(prompt=str)` helper raises with a permission callback.
- **Windows console encoding.** `agent.py` reconfigures stdout/stderr to UTF-8 and uses `Console(legacy_windows=False)` — Rich otherwise crashes on glyphs like `→`/`✓` under cp1252.
- **Two shell tools.** On Windows the agent has both `Bash` and `PowerShell`; `permissions._SHELL_TOOLS` gates them identically. New shell tools must be added there.
- **`gh` on PATH.** `config.agent_env()` prepends the GitHub CLI install dir to the agent subprocess PATH only when `gh` isn't already resolvable (it was winget-installed mid-session).
- **Max session limits.** Live agent runs consume the same Max usage budget as interactive Claude Code. Prefer the deterministic test suite for verification; don't burn budget hammering live runs.
