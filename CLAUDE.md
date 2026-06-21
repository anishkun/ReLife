# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ReLife is a personal agent built on the **Claude Agent SDK** (Python) that acts through MCP servers and **learns over time** (facts + reusable skills). It runs on the **Claude Code Max subscription**, NOT a metered API key — `ANTHROPIC_API_KEY` is intentionally unset and the SDK drives the logged-in `claude` CLI. Do not introduce code that requires an API key. See `PROJECT_CONTEXT.md` for the full design rationale, locked decisions, and current status — read it before making architectural changes.

## Commands

```sh
pip install -e .                  # install (editable); creates the `relife` entry point
pip install -e ".[embeddings]"    # + optional LOCAL semantic recall (fastembed; no API key)
python -m pytest tests/           # run all tests (50, deterministic — no live agent calls)
python -m pytest tests/test_permissions.py::test_name -v   # single test

relife do "<task>"                # one-shot: run a task to completion
relife chat                       # interactive multi-turn session
relife build "<spec>"             # orchestrated large build (decompose → delegate → resume)
relife build --resume [ID]        # continue a build (most recent for the workspace if no ID)
relife consolidate                # run a memory "sleep" pass now (decay/dedupe/learn workflows)
relife memory stats               # counts, strongest memories, what has faded
# do/chat/build accept --workspace/-w PATH (default ./workspace) — the dir the agent works in
```

Prereqs to actually *run* the agent (not needed for tests): Python ≥3.11, Node.js (`npx` for the browser MCP), a logged-in `claude` CLI on Max, and authenticated `gh`.

## Architecture

The flow is `cli.py` → `agent.build_options()` → `ClaudeSDKClient` (streaming). Each `do`/`chat` invocation wires together three pluggable pieces, all passed into `build_options`:

- **Permissions** (`permissions.py`) — `classify()` is a pure function returning `("allow"|"ask", reason)`; `make_permission_callback()` wraps it as the SDK `can_use_tool`. Autonomy model: auto-allow read-only tools, file writes **inside the workspace**, shell build/test/git (incl. `git push`), and ReLife's own MCP tools (`mcp__relife*`, `mcp__browser`). Always-ask for outward/destructive actions (the `_OUTWARD_BASH` regex catches email senders, `gh pr|issue|release|api|gist`, uploads, remote shells, package publish, `sudo`, root `rm -rf`), writes outside the workspace, and any unrecognized tool (fail-closed default). Non-interactive runs **deny** ask-cases rather than block.
- **MCP servers** (`config.default_mcp_servers()`) — `browser` (Playwright via `npx @playwright/mcp`) and `relife_memory` (in-process SDK server). Memory is shipped as an MCP server *even though in-process* so the agent-facing contract is unchanged when it's later split out.
- **Memory hooks** (`hooks.py`) — a `UserPromptSubmit` hook auto-injects relevant recalled memories + skills + workflows before each prompt (and *reinforces* what it surfaces); a `PostToolUse` hook journals every tool call to the event log (raw material for pattern detection). The agent benefits without explicitly calling recall.

**Per-task loop:** recall (hook injects memory+skills+workflows, reinforcing them) → act (tools, gated by `classify`, journaled to events) → reflect (agent calls `memory_save`/`skill_write`/`workflow_save`) → **consolidate** (a "sleep" pass after the run: decay/forget, dedupe, learn workflows from recurring tool sequences).

### Memory layer (`relife/memory/`) — cognitive model
Memory behaves like a brain: each item's relevance **rises when used** and **fades when ignored**, recall **fuses four signals**, and a consolidation pass periodically forgets and generalizes. The public `save`/`recall` API stays backward-compatible with v1.
- `cognitive.py` — **pure, deterministic, no I/O** heart of the model: ACT-R-inspired `activation()` (frequency + recency decay + importance), `fused_score()`, `should_archive()`. All tunables in `config.py`.
- `store.py` — SQLite at `data/relife.db`, schema v2 (idempotent migration adds `importance/last_used_at/use_count/status/embedding`). **Two-stage recall** keeps it cheap at scale: Stage 1 pulls bounded candidates from an **FTS5** index (+ a semantic scan when embeddings are on), Stage 2 fuse-ranks only those. `recall(..., reinforce=True)` strengthens hits; `archive()` soft-forgets (reversible). Re-saving identical text reinforces instead of duplicating. `_DB_PATH` is a reassignable module global (tests rely on it).
- `embeddings.py` — **soft-optional** local semantic vectors via `fastembed` (ONNX, offline, **no API key**). `available()/embed()/cosine()`; everything degrades to keyword+activation if absent. Never a hard dependency; tests never require it.
- `skills.py` — single reusable procedures (Markdown+frontmatter under `data/skills/`).
- `workflows.py` — **multi-step** procedures (ordered chains of skills/actions) under `data/workflows/`, same file format as skills + a `trigger` field.
- `events.py` — tool-event log (own table in `relife.db`); `_DB_PATH` reassignable for tests.
- `consolidate.py` — the **"sleep" pass** (`run_consolidation()`, deterministic, LLM-free): (1) decay→archive via `should_archive`; (2) dedupe near-duplicate memories; (3) detect recurring episodes + recurring tool **n-grams**, recording `pattern` memories; (4) synthesize a **workflow** from each *maximal* recurring sequence. Auto-runs after a run when enough events accrue (`should_auto_run`, gated by `config.AUTO_CONSOLIDATE`); state in `data/consolidate_state.json`.
- `server.py` — MCP tools `memory_save` (now takes `importance`), `memory_recall`, `memory_forget`, `skill_write`/`skill_find`, `workflow_save`/`workflow_find`, `memory_consolidate`. Server `relife_memory` → `mcp__relife_memory__*` (trusted `mcp__relife` prefix → auto-allowed; **no permission changes**).
- `_text.py` — shared stopword tokenizer used by every keyword path.

The system prompt uses the **`claude_code` preset** with `prompts/system.md` appended (persona + safety + memory/skill/workflow instructions). `setting_sources=None` deliberately prevents inheriting the surrounding repo's Claude Code settings.

### Build orchestration (`relife/build/`)
`relife build` scales to large projects the single-context `do` loop can't: it **decomposes → delegates → resumes**.
- `ledger.py` — `BuildLedger`: durable plan + progress at `data/builds/<id>/ledger.json` (+ a `plan.md` mirror). Source of truth for resume. Pure/deterministic.
- `server.py` — in-process MCP server `relife_build` (tools `build_plan_set`/`build_milestone_update`/`build_status`), bound to one ledger per run via closure. Surfaces as `mcp__relife_build__*` → already auto-allowed by the trusted `mcp__relife` prefix (no permission change).
- `agents.py` — the `builder` `AgentDefinition` the orchestrator delegates each milestone to via the **Task** tool, so each milestone runs in a *fresh context* (the orchestrator stays small). Parallel milestones are deliberately deferred.
- `orchestrator.py` — `run_build()`: wires ledger + server + builder + the `prompts/orchestrator.md` persona, streams the run, and persists `ResultMessage.session_id` so `--resume` continues the same session. Resume also re-injects the ledger state, so it's robust even if the CLI session is gone.
- `build_options` (agent.py) gained `system_prompt`/`agents`/`resume`/`max_budget_usd` params to support this; `do`/`chat` are unchanged.

## Non-obvious constraints

- **`can_use_tool` requires streaming mode.** Use `ClaudeSDKClient` + `receive_response()`; the one-shot `query(prompt=str)` helper raises with a permission callback.
- **Windows console encoding.** `agent.py` reconfigures stdout/stderr to UTF-8 and uses `Console(legacy_windows=False)` — Rich otherwise crashes on glyphs like `→`/`✓` under cp1252.
- **Two shell tools.** On Windows the agent has both `Bash` and `PowerShell`; `permissions._SHELL_TOOLS` gates them identically. New shell tools must be added there.
- **`gh` on PATH.** `config.agent_env()` prepends the GitHub CLI install dir to the agent subprocess PATH only when `gh` isn't already resolvable (it was winget-installed mid-session).
- **Max session limits.** Live agent runs consume the same Max usage budget as interactive Claude Code. Prefer the deterministic test suite for verification; don't burn budget hammering live runs.
- **FTS5 external-content `COUNT(*)` lies.** `memories_fts` uses `content='memories'`, so `SELECT COUNT(*) FROM memories_fts` reads the *content table*, not the index — it can't tell you the index is empty. `store._init_fts` rebuilds based on whether the FTS table is being created for the first time (sqlite_master check), not on a count. FTS5 is also feature-detected: if the SQLite build lacks it, recall falls back to a full keyword scan.
- **Embeddings are soft-optional and OFF in tests.** `fastembed` isn't a hard dep; `embeddings.available()` is `False` unless installed, so recall stays deterministic (keyword + activation) in CI. The candidate gate (keyword overlap **or** strong semantic sim) preserves the "unrelated query → nothing" guarantee even with embeddings on.
- **Consolidation must stay LLM-free.** `consolidate.run_consolidation()` is deterministic by design (cheap, safe to auto-run, testable). Don't add live model calls to it; LLM enrichment of synthesized workflows is intentionally deferred.
