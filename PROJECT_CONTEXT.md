# ReLife — Project Context & Handoff

> Durable reference for future sessions. Captures *why* things are the way they are,
> what's built and verified, the non-obvious gotchas, and what's next.
> Last updated: 2026-06-22.

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

**Tests:** 50 passing (`python -m pytest tests/`). Covers permission classify, store
save/recall, skills, the recall hook injecting memory+skills+workflows, the build
ledger + ledger MCP tools, and the **cognitive memory v2** layer — activation/decay
math, schema migration + two-stage fused recall + reinforcement/archival, workflows,
the event log, and the consolidation pass (deterministic — no live agent).

**Cognitive memory v2 — added post-v1.** The memory layer now behaves like a brain:
relevance **rises with use and fades when idle** (ACT-R-style activation in
`cognitive.py`), recall **fuses semantic + keyword + activation + importance** and is
**two-stage** (FTS5 candidates → fuse-rank) so it scales, and a **consolidation
("sleep") pass** (`consolidate.py`) auto-runs after tasks to forget stale memories,
dedupe, and **synthesize workflows from recurring tool sequences** it observes via a
new event log. Semantic recall uses a **local** embedding model (`fastembed`, ONNX,
no API key) and is soft-optional — absent it degrades to keyword + activation. New
modules: `cognitive.py`, `embeddings.py`, `workflows.py`, `events.py`,
`consolidate.py`; new MCP tools (`memory_forget`, `workflow_save/find`,
`memory_consolidate`, `importance` on `memory_save`); new CLI (`relife consolidate`,
`relife memory stats`). Consolidation is deliberately deterministic/LLM-free to
protect the Max budget (LLM enrichment of synthesized workflows deferred).
Verified deterministically + a temp-dir smoke (reinforcement reorders recall, stale
memory archived, a recurring Read→Edit→Bash sequence learned as a workflow).

**Large builds (`relife build`) — added post-v1.** Orchestration layer for projects too big
for one context: the orchestrator decomposes the spec into milestones (persisted in a
`BuildLedger` at `data/builds/<id>/`), delegates each to a fresh-context `builder` subagent via
the Task tool (keeps the orchestrator's context small), and is **resumable** — `relife build
--resume` continues after a Max session limit using the ledger + persisted `session_id`. See
`relife/build/`. Parallel milestones deferred. **Exercised live end-to-end & verified:**
build `20260620-…-0c1b` (multi-service FastAPI+CLI+tests todo app, 7 milestones, 60 tests)
and build `20260621-…-a556` (`tempconv` CLI, 4 milestones, 41 tests, $1.39 usage-equiv). Both
decomposed → delegated each milestone to a fresh-context `builder` → completed within budget;
deterministic tests still cover the persistence layer. Resume path not yet triggered live (no
session limit hit), but `session_id` is persisted for it.

## 5. File map (`D:\ReLife`)

```
pyproject.toml              # deps: claude-agent-sdk, typer, rich
PROJECT_CONTEXT.md          # this file
README.md
relife/
  cli.py                    # Typer: `relife do`, `relife chat`, `relife build` (--workspace)
  agent.py                  # build_options + run_task/run_chat (ClaudeSDKClient, streaming)
  config.py                 # MODEL, EFFORT, paths, agent_env() (gh PATH), default_mcp_servers()
  permissions.py            # classify() + make_permission_callback() (can_use_tool)
  hooks.py                  # UserPromptSubmit recall hook (memory + skills)
  prompts/system.md         # persona + safety + memory/skill instructions
  memory/                   # cognitive memory: relevance rises w/ use, fades when idle
    cognitive.py            # pure ACT-R-style activation/fused_score/should_archive/should_hard_delete
    store.py                # injectable MemoryStore: SQLite (user_version migrations), two-stage fused recall, decay
    vector_index.py         # VectorIndex seam: BruteForceIndex + soft-optional SqliteVecIndex (self-tested)
    service.py              # MemoryService — in-process facade over MemoryStore
    client.py               # MemoryClient seam; LocalMemoryClient (default transport) + default_client()
    embeddings.py           # soft-optional LOCAL semantic vectors (fastembed; no API key)
    skills.py               # single reusable procedures (Markdown files)
    workflows.py            # multi-step procedures (ordered skill/action chains)
    events.py               # injectable EventLog (tool-event log for pattern detection)
    consolidate.py          # "sleep" pass: decay/forget + hard-delete, (semantic) dedupe, learn workflows
    rem.py                  # opt-in "dream" pass (`relife dream`): LLM adversarial critic; prunes/reweights, reversibly
    server.py               # MCP server (tools route through default_client()): memory/skill/workflow tools
    _text.py                # shared tokenizer w/ stopwords
  build/                    # `relife build`: orchestrated, resumable large builds
    ledger.py               # BuildLedger — durable plan+progress (data/builds/<id>/)
    server.py               # relife_build MCP server (plan_set/milestone_update/status)
    agents.py               # `builder` subagent definition (Task-delegated milestones)
    orchestrator.py         # run_build(): decompose → delegate → resume
    prompts/orchestrator.md # orchestrator persona (architect/PM, delegates building)
data/                       # gitignored runtime: relife.db, skills/, builds/, logs
scripts/bench_recall.py     # non-CI recall scaling benchmark (10k+ memories)
tests/                      # 67 tests (63 deterministic + 4 semantic, embeddings forced off)
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
- **Test repo `anishkun/relife-demo` — ✅ DELETED.** Removed via the GitHub web UI
  (the `gh` token lacked the `delete_repo` scope). API confirms 404.
- **Full live skills round-trip** (scaffold twice, second run reuses skill) was deferred by
  the session limit; each half is proven separately. Run when budget is comfortable.
- **Long-term memory deepening — ✅ DONE (this phase).** Four tracks landed behind the unchanged
  `mcp__relife_memory__*` contract: (A) smarter recall/forgetting — `RECALL_FLOOR`, kind-aware
  `fused_score`, semantic dedup, tiered hard-delete; (B) better save/surface — kind-based default
  importance, recall-hook de-dup + size budget, deterministic episode capture on `Stop`;
  (D) scale — `VectorIndex` seam (brute-force + self-tested optional `sqlite-vec`), `user_version`
  migrations, 10k+ benchmark; (C) the memory-only **service seam** (`MemoryService` + `MemoryClient`/
  `LocalMemoryClient`, all consumers via `default_client()`). 67 tests green. See
  `.claude/plans/keen-wiggling-octopus.md` for the design.
- **REM "dream" pass — ✅ DONE (this phase).** Added `relife dream` (and `memory_dream` MCP
  tool): the **only** LLM-driven memory path, deliberately **opt-in / never auto-run** so it
  never silently spends Max budget. The model is an *adversarial critic* over a bounded,
  watermarked "replay buffer" of recent memories; it can only **prune (archive, reversible)** or
  **reweight (importance)** — never edit text. Every verdict is confidence-gated + prune-capped +
  journaled (`data/rem_journal.jsonl`), so a misbehaving critic can't corrupt memory. The
  deterministic `consolidate.py` stays LLM-free (guard tests enforce both invariants). New:
  `memory/rem.py`, `prompts/rem.md`, `agent.ask_model_oneshot`, `store.set_importance`,
  `MemoryService/Client.dream`. 77 tests green (was 67). **Honest expectation:** this is a
  *qualitative/safety* improvement (contradiction/hallucination/alignment pruning) with
  diminishing returns — it does not change recall ranking, so it is not "exponentially" more
  accurate; budget is gated for *risk* reasons, not just cost. Live `relife dream` smoke deferred
  to a comfortable-budget window.
- **Phase 2/3 (next):** flip `LocalMemoryClient` → an `HttpMemoryClient` against a standalone
  long-lived memory **daemon** (the seam now makes this a drop-in); always-on agent + UI; outward
  capabilities (email/calendar/work-items) — Anthropic **Managed Agents** is the natural host
  (hosted memory stores, MCP vaults, GitHub mounting, scheduled deployments).

## 9. Key facts to remember

- GitHub account: **`anishkun`** (= anish03anish@gmail.com). `gh` authed, `repo` scope, HTTPS.
- Approved plan lives at `C:\Users\HP\.claude\plans\witty-enchanting-fountain.md`.
- Memory about auth constraint: `…/.claude/projects/D--ReLife/memory/auth-via-max-subscription.md`.
