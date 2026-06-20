# ReLife — Project Context & Handoff

> Durable reference for future sessions. Captures *why* things are the way they are,
> what's built and verified, the non-obvious gotchas, and what's next.
> Last updated: 2026-06-20.

## 1. Vision

A personal agent that can eventually "do anything I (the user) can," acts on the world
through **MCP servers**, and has a **long-term memory that grows over time** so it gets
measurably better at recurring tasks.

- **v1 (built):** terminal agent that builds projects, pushes to git, drives a browser —
  autonomous for code/git, asks approval for outward actions. Memory of facts + reusable
  skills.
- **Future:** complete assigned work items, reply to email/messages, plan the calendar;
  an always-on daemon + UI.

## 2. Locked decisions (with rationale)

| Area | Choice | Why |
|---|---|---|
| Foundation | **Claude Agent SDK** (Python, `claude-agent-sdk` v0.2.105) | Same engine as Claude Code: inherits the production agent loop, native MCP, hooks, permissions. The model is the same across any foundation, so effort goes into memory (the real edge), not plumbing. |
| Language | **Python** (≥3.11; dev machine has 3.14) | Best ecosystem for the memory/embeddings side. |
| Model | **`claude-opus-4-8`**, effort `high` | Strong agentic work. |
| **Auth / billing** | **Claude Code Max subscription** — NOT a metered API key | User explicitly does not want to buy API tokens. The SDK drives the logged-in `claude` CLI, so it uses the subscription. Verified: `ANTHROPIC_API_KEY` unset, queries still succeed. **Caveat:** agent runs consume the same Max usage limits as interactive Claude Code (we hit the limit once mid-build). |
| Memory | **A + B: retrieval (facts/episodes) + procedural skills.** No fine-tuning. | Retrieval shrinks re-derivation; skills replace re-planning → improvement without training. |
| Interface | **Terminal CLI** now; daemon + UI later | |
| Security | Code + git push **autonomous**; outward actions (mail/messages/etc.) **require approval** | User's stated autonomy model. |

## 3. Architecture

```
relife (CLI, Typer)
  └─ ClaudeSDKClient (streaming)  ── Claude Opus 4.8, Claude Code preset + ReLife persona
       ├─ built-in tools: Read/Write/Edit/Bash/PowerShell/Glob/Grep/WebFetch/WebSearch
       ├─ can_use_tool  → permission policy (auto-allow code/git, ask for outward)
       ├─ UserPromptSubmit hook → auto-inject recalled memories + skills
       └─ MCP servers:
            ├─ browser        (Playwright, npx @playwright/mcp)  → mcp__browser__*
            └─ relife_memory  (in-process SDK MCP server)        → mcp__relife_memory__*
                 ├─ memory_save / memory_recall   (facts/preferences/episodes)
                 └─ skill_write / skill_find       (reusable procedures)
GitHub: via `gh` CLI (build → commit → gh repo create → push)
```

**Why memory is an MCP server (even in-process):** the agent-facing contract stays identical
when we later split it into a standalone server — the "own memory layer via MCP" upgrade is
pre-wired. (`create_sdk_mcp_server` + `@tool`.)

**Control loop per task:** recall (hook injects relevant memory+skills) → act (tools, gated by
policy) → reflect (agent calls `memory_save` / `skill_write` for durable lessons).

## 4. Current status — ALL v1 TASKS DONE & VERIFIED

| # | Task | Status / proof |
|---|---|---|
| 1 | Env + subscription auth | ✅ `ANTHROPIC_API_KEY` unset, query returns AUTH_OK |
| 2 | Skeleton CLI + agent | ✅ built a hello-world project end-to-end |
| 3 | Permission model | ✅ wrote in-workspace file (allow), blocked `mail` (ask→deny); 8 unit tests |
| 4 | Git + browser MCP | ✅ built+committed+created private repo `anishkun/relife-demo`+pushed; navigated example.com |
| 5 | Memory (retrieval A) | ✅ taught ruff+gitignore in run A; **unrelated** run B applied both unprompted |
| 6 | Skills (B) | ✅ agent wrote `push-new-github-repo` skill live; recall hook surfaces skills (deterministic test) |

**Tests:** 17 passing (`python -m pytest tests/`). Covers permission classify, store
save/recall, skills, and the recall hook injecting memory+skills.

## 5. File map (`D:\ReLife`)

```
pyproject.toml              # deps: claude-agent-sdk, typer, rich
PROJECT_CONTEXT.md          # this file
README.md
relife/
  cli.py                    # Typer: `relife do "<task>"`, `relife chat`  (--workspace)
  agent.py                  # build_options + run_task/run_chat (ClaudeSDKClient, streaming)
  config.py                 # MODEL, EFFORT, paths, agent_env() (gh PATH), default_mcp_servers()
  permissions.py            # classify() + make_permission_callback() (can_use_tool)
  hooks.py                  # UserPromptSubmit recall hook (memory + skills)
  prompts/system.md         # persona + safety + memory/skill instructions
  memory/
    store.py                # SQLite facts/episodes, keyword+recency recall
    skills.py               # Markdown skill files, keyword recall
    server.py               # in-process MCP server (memory_save/recall, skill_write/find)
    _text.py                # shared tokenizer w/ stopwords
data/                       # gitignored runtime: relife.db, skills/, logs
tests/                      # 17 unit/integration tests
```

## 6. Setup / run

Prereqs: Python ≥3.11, Node.js (npx for browser MCP), logged-in `claude` CLI (Max), and
`gh` authenticated.

```sh
pip install -e .
relife do "scaffold a Python CLI that prints the weather for a city"
relife chat
# --workspace PATH chooses the dir the agent works in (default ./workspace)
```

## 7. Non-obvious gotchas (learned the hard way)

- **`can_use_tool` requires streaming mode.** `query(prompt=str)` raises; we use
  `ClaudeSDKClient` + `receive_response()` for both `do` and `chat`.
- **Windows console encoding.** Rich crashed on `→`/`✓` under cp1252. Fixed in `agent.py`:
  reconfigure stdout/stderr to utf-8 + `Console(legacy_windows=False)`.
- **PowerShell tool.** On Windows the agent has a separate `PowerShell` tool (not just
  `Bash`). `permissions._SHELL_TOOLS` gates both identically.
- **`gh` PATH.** Installed via winget mid-session → not on the parent shell PATH. `config.agent_env()`
  prepends `C:\Program Files\GitHub CLI` to the agent subprocess PATH when `gh` isn't found.
- **Max session limits.** Heavy multi-step agent runs can hit the subscription limit
  ("You've hit your session limit · resets …"). Prefer cheap/deterministic verification;
  don't re-burn budget hammering live runs.
- **Inherited claude.ai MCP connectors.** The subscription surfaces Gmail/Calendar/Drive MCP
  servers (status `needs-auth`). Useful later; our policy gates them (not in trusted prefixes).
- **git author identity.** This machine's *global* git config is `tezoo2002@live.com` /
  "ReLife" (the system email) — so commits ReLife makes are authored as that unless changed.
  See open items.

## 8. Open items / next steps

- **Git identity fix — ✅ DONE.** Global git config is now `anishkun` /
  `anish03anish@gmail.com` (= the GitHub account). Commits author correctly. The leftover
  `data/_recommit` temp dir has been removed.
- **ReLife git repo — ✅ DONE.** `git init` + initial commit (`480e1b7`); pushed to the
  **public** repo `anishkun/ReLife` (https://github.com/anishkun/ReLife). Added a proprietary
  "All Rights Reserved" `LICENSE` (custom → GitHub shows no license badge, by design).
  `.claude/settings.local.json` is gitignored (machine-local).
- **Test repo `anishkun/relife-demo` — DELETE PENDING.** User chose to delete it, but `gh`
  lacks the `delete_repo` scope. Grant once with
  `gh auth refresh -h github.com -s delete_repo`, then `gh repo delete anishkun/relife-demo --yes`.
- **Full live skills round-trip** (scaffold twice, second run reuses skill) was deferred by
  the session limit; each half is proven separately. Run when budget is comfortable.
- **Phase 2/3:** standalone MCP memory server; always-on daemon + UI; outward capabilities
  (email/calendar/work-items) — Anthropic **Managed Agents** is the natural host (hosted
  memory stores, MCP vaults, GitHub mounting, scheduled deployments).

## 9. Key facts to remember

- GitHub account: **`anishkun`** (= anish03anish@gmail.com). `gh` authed, `repo` scope, HTTPS.
- Approved plan lives at `C:\Users\HP\.claude\plans\witty-enchanting-fountain.md`.
- Memory about auth constraint: `…/.claude/projects/D--ReLife/memory/auth-via-max-subscription.md`.
