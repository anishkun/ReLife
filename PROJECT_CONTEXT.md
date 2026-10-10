# ReLife — Project Context & Handoff

> Durable reference for future sessions. Captures *why* things are the way they are,
> what's built and verified, the non-obvious gotchas, and what's next.
> Last updated: 2026-10-10.

## 1. Vision

A personal agent that can eventually "do anything I (the user) can," acts on the world
through **MCP servers**, and has a **long-term memory that grows over time** so it gets
measurably better at recurring tasks.

- **v1 (built):** terminal agent that builds projects, pushes to git, drives a browser —
  autonomous for code/git, asks approval for outward actions. Memory of facts + reusable
  skills.
- **Since v1 (built, released as 1.0.0 on 2026-10-08):** cognitive memory (decay,
  consolidation, opt-in "dream"), orchestrated resumable builds, an always-on server +
  web console with schedules and run records, GitHub work items (`relife work`, work
  schedules), Gmail/Calendar/Drive via the claude.ai connectors with narrow
  pre-approvals, `relife doctor`.
- **Platform pass (built 2026-10-10, unreleased):** agents with their own memory spaces,
  memory handoff (inherit/fork/promote/packs), ReLife memory over MCP for any LLM, and
  `relife crew` (CrewAI plans a team, ReLife staffs it).
- **Platform in the server (built 2026-10-10, unreleased):** `/agents` `/spaces` `/crews`
  routes, crews run under `relife serve` with members' approvals in the browser, crew
  schedules, Agents + Crews panels in the web console.
- **Next:** ReLife-gated machine tools for non-Claude agents; hosting (Managed Agents) later.

## 2. Locked decisions (with rationale)

| Area | Choice | Why |
|---|---|---|
| Foundation | **Claude Agent SDK** (Python, `claude-agent-sdk` ≥ 0.2.105) | Same engine as Claude Code: inherits the production agent loop, native MCP, hooks, permissions. The model is the same across any foundation, so effort goes into memory (the real edge), not plumbing. |
| Language | **Python** (≥3.11; dev machine has 3.14) | Best ecosystem for the memory/embeddings side. |
| Model | **`claude-opus-4-8`**, effort `high` | Strong agentic work. |
| **Auth / billing** | **Claude Code Max subscription** — NOT a metered API key | User explicitly does not want to buy API tokens. The SDK drives the logged-in `claude` CLI, so it uses the subscription. Verified: `ANTHROPIC_API_KEY` unset, queries still succeed. **Caveat:** agent runs consume the same Max usage limits as interactive Claude Code (we hit the limit once mid-build). |
| Memory | **A + B: retrieval (facts/episodes) + procedural skills.** No fine-tuning. | Retrieval shrinks re-derivation; skills replace re-planning → improvement without training. |
| Interface | **Terminal CLI** + an always-on local server and web console (`relife serve`) | CLI first; the server came once the core was proven. |
| Security | Code + git push **autonomous**; outward actions (mail/messages/etc.) **require approval** | User's stated autonomy model. Narrow per-schedule pre-approvals are the only exception. |
| Other LLMs (2026-10-10) | **Any provider, keys optional** — via CrewAI/LiteLLM, keys read from the env; Claude always via Max (`ClaudeMaxLLM`) | ReLife never stores or requires a key. |
| Memory handoff (2026-10-10) | **Own space + inherit read-only**; fork, promote and packs on top | An agent can never write the user's memory; trusting it is an explicit `promote`. |
| Crews' Python (2026-10-10) | **Python 3.12 `.venv`** for `[crewai]` | CrewAI 1.15 requires `<3.14`; ReLife's core stays 3.11–3.14. |

## 3. Architecture

```
relife (CLI, Typer)  ·  relife serve (FastAPI + web console, schedules)
  └─ ClaudeSDKClient (streaming)  ── Claude Opus 4.8, Claude Code preset + ReLife persona
       ├─ built-in tools: Read/Write/Edit/Bash/PowerShell/Glob/Grep/WebFetch/WebSearch
       ├─ can_use_tool  → permission policy (auto-allow code/git, ask for outward; grants)
       ├─ hooks: UserPromptSubmit (inject recalled memory + skills + workflows),
       │         PostToolUse (journal), Stop (episode)
       └─ MCP servers:
            ├─ browser        (Playwright, npx @playwright/mcp)  → mcp__browser__*
            ├─ relife_memory  (in-process SDK MCP server)        → mcp__relife_memory__*
            │    memory_save/recall/forget, skill_write/find, workflow_save/find,
            │    memory_consolidate, memory_dream
            └─ claude.ai connectors (Gmail / Calendar / Drive, from the account)
memory: MemoryClient seam → Local (default) | Http daemon (RELIFE_MEMORY_URL);
        spaces per agent (ScopedMemoryClient); also served to ANY MCP client:
        `relife mcp --agent NAME` (stdio) and /mcp on the daemon (agent token)
crews:  `relife crew` → CrewAI Crew of ReLifeAgent (a ReLife turn per task) +
        CrewAI agents on other models (ReLife memory, no machine tools)
GitHub: via `gh` CLI (build → commit → gh repo create → push; `relife work` → PR)
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

**Tests:** ~810 passing — 800 on Python 3.14 (the CrewAI modules skip), 811 in the 3.12 `.venv`
with `[crewai]` (`python -m pytest tests/`; later phases below added the daemon, server,
scheduler, run-outcome, doctor, work-item, grant, spaces, agents, MCP and crew suites). The original set covers permission classify, store
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
pyproject.toml              # deps: claude-agent-sdk, typer, rich; extras embeddings/vector/daemon/server/crewai
PROJECT_CONTEXT.md          # this file
README.md / CHANGELOG.md / RELEASE_TESTING.md / CLAUDE.md
HOW_IT_WORKS.md             # plain-English guide;  MODULE_DEEP_DIVE.md — architecture + reasoning (M1–M16)
relife/
  cli.py                    # Typer: do/chat (--agent)/work/build/serve/doctor/memory */agent */mcp/crew/crews
  __main__.py               # `python -m relife` (what MCP client configs launch)
  agents.py                 # agent registry: identity → MemoryScope; inherit/fork/promote; hashed MCP tokens
  agent.py                  # build_options + run_task/run_chat (ClaudeSDKClient, streaming), to_event
  config.py                 # MODEL, EFFORT, paths (RELIFE_HOME), agent_env() (gh PATH), default_mcp_servers()
  permissions.py            # classify() + callbacks (TTY / UI); gh + connector verb policy; schedule grants
  workitems.py              # `relife work`: GitHub issue plumbing (parse/list/fetch/checkout/branch/prompt)
  doctor.py                 # `relife doctor`: pure run_checks() over injected Probes
  hooks.py                  # recall (UserPromptSubmit), journal (PostToolUse), episode (Stop); per-client factory
  prompts/system.md         # persona + safety + memory/skill instructions
  web/index.html            # self-contained web console served by `relife serve`
  memory/                   # cognitive memory: relevance rises w/ use, fades when idle
    cognitive.py            # pure ACT-R-style activation/fused_score/should_archive/should_hard_delete
    store.py                # injectable MemoryStore: SQLite (user_version migrations), two-stage fused recall, decay
    vector_index.py         # VectorIndex seam: BruteForceIndex + soft-optional SqliteVecIndex (self-tested)
    spaces.py               # memory spaces: names, MemoryScope(read, write, source), per-space dirs
    tools.py                # the memory tools, defined once (ToolSpec) for every transport
    context.py              # the recalled-context block (recall hook + memory_context tool)
    mcp_server.py           # ReLife memory over MCP for any agent: stdio + the daemon's /mcp
    service.py              # MemoryService — in-process facade over MemoryStore
    client.py               # MemoryClient seam; default_client() picks Local (default) or Http by RELIFE_MEMORY_URL
    remote/                 # opt-in out-of-process daemon ([daemon] extra): wire.py, daemon.py (FastAPI), http_client.py
    embeddings.py           # soft-optional LOCAL semantic vectors (fastembed; no API key)
    skills.py               # single reusable procedures (Markdown files)
    workflows.py            # multi-step procedures (ordered skill/action chains)
    events.py               # injectable EventLog (tool-event log for pattern detection)
    consolidate.py          # "sleep" pass: decay/forget + hard-delete, (semantic) dedupe, learn workflows
    rem.py                  # opt-in "dream" pass (`relife dream`): LLM adversarial critic; prunes/reweights, reversibly
    server.py               # MCP server (tools route through default_client()): memory/skill/workflow tools
    _text.py                # shared tokenizer w/ stopwords
  crew/                     # `relife crew` ([crewai] extra, Python <= 3.13)
    spec.py / planner.py    # pure: validated CrewSpec; Claude plans it (one tool-less call)
    agent.py                # ReLifeAgent(BaseAgentAdapter): a crew member = a fresh ReLife turn
    llm.py                  # ClaudeMaxLLM(BaseLLM): Claude via the CLI for CrewAI (no API key)
    native.py / memory_tools.py  # CrewAI agents on other models + ReLife memory as CrewAI tools
    build.py / runner.py / record.py  # profiles+handoff → Crew → kickoff → data/crews/<id>/
  build/                    # `relife build`: orchestrated, resumable large builds
    ledger.py               # BuildLedger — durable plan+progress (data/builds/<id>/)
    server.py               # relife_build MCP server (plan_set/milestone_update/status)
    agents.py               # `builder` subagent definition (Task-delegated milestones)
    orchestrator.py         # run_build(): decompose → delegate → resume
    prompts/orchestrator.md # orchestrator persona (architect/PM, delegates building)
  server/
    session.py              # AgentSession (busy flag, per-turn grants) / ApprovalBroker / SessionManager
    app.py                  # create_app (sessions, SSE, approvals, /schedules) + serve()
    security.py             # pure auth/CSRF/bind/DNS-rebinding/workspace-confinement policy
    schedules.py            # Schedule record (grants, work), spec parsing, next_run(), JSON ScheduleStore
    scheduler.py            # lifespan tick loop: fire due schedules (incl. work schedules) + record outcomes
    runs.py                 # RunRecord / summarize_events / per-schedule RunStore (data/runs/)
data/                       # gitignored runtime: relife.db, skills/, workflows/, spaces/, agents.json,
                            #   builds/, crews/, runs/, schedules.json
scripts/bench_recall.py     # non-CI recall scaling benchmark (10k+ memories)
tests/                      # ~800 tests (deterministic + 4 semantic, embeddings forced off)
.venv/                      # gitignored Python 3.12 env for `[crewai]` (CrewAI needs < 3.14)
```

## 6. Setup / run

Prereqs: Python ≥3.11, Node.js (npx for browser MCP), logged-in `claude` CLI (Max), and
`gh` authenticated.

```sh
pip install -e .                  # extras: ".[embeddings]", ".[vector]", ".[daemon]", ".[server]"
relife doctor                     # what's missing, with the fix for each
relife do "scaffold a Python CLI that prints the weather for a city"
relife chat
# --workspace PATH chooses the dir the agent works in (default ./workspace)

# crews (CrewAI needs Python <= 3.13):
py -3.12 -m venv .venv
.venv\Scripts\pip install -e ".[crewai]"
.venv\Scripts\relife crew --plan-only "<task>"
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
- **Inherited claude.ai MCP connectors.** The subscription attaches Gmail/Calendar/Drive MCP
  servers to every session (`setting_sources=None` doesn't exclude them). Linked in-session
  on first use; gated by the verb-based connector policy (see MVP pass 2 and post-1.0 below).
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
- **Phase 2 — memory daemon split — ✅ DONE.** `default_client()` now returns an
  `HttpMemoryClient` against a standalone long-lived daemon (`relife memory serve`) **iff
  `RELIFE_MEMORY_URL` is set**; unset, memory stays fully in-process (default, no regressions).
  Opt-in, **zero consumer changes** — the `MemoryClient` seam made it a drop-in. New
  `relife/memory/remote/` (`wire`/`daemon`/`http_client`), optional `[daemon]` extra
  (fastapi/uvicorn/httpx), `relife memory serve`/`ping`, `RELIFE_MEMORY_URL`/`_TOKEN`/`_HOST`/
  `_PORT`. Transport = loopback HTTP/REST (behind the protocol, so swappable; remote-ready in
  phase 3). **Key invariant — do not "fix" this:** the daemon binds the *module-level*
  `store._DB_PATH` **and** `events._DB_PATH` and uses the **default** `MemoryService()`, NOT an
  injected store — because `consolidate()`/`dream()` mine the module default; injecting a store
  would silently split save/recall from upkeep onto different DBs. 93 tests green (was 81):
  wire round-trip + a parametrized Local-vs-Http conformance suite (Http driven by FastAPI
  `TestClient`, no real socket). See `.claude/plans/so-http-is-the-snoopy-pinwheel.md`.
- **Phase 3 — skills+workflows server-side — ✅ DONE (this phase).** Skills and workflows now
  route through the `MemoryClient` seam just like long-term memory: `MemoryService` gained
  `skill_write/find/count` + `workflow_write/find/count`, the daemon gained `/skills/*` +
  `/workflows/*` routes, and `HttpMemoryClient` implements them (with 400→`ValueError` mapping so
  both transports fail identically). Consumers (recall hook, MCP `skill_*`/`workflow_*` tools,
  `memory stats`) now call `default_client()`; **MCP tool names/schemas unchanged**. **Key
  invariant — do not "fix":** `consolidate.py` still calls `workflows.write_workflow` *directly*
  (module function), never the client — it always runs where the data lives (in-process locally,
  *inside* the daemon under `POST /consolidate`); routing it through the client would make the
  daemon's `async` handler issue a blocking HTTP call back into its own event loop → deadlock.
  Instead `create_app` binds `skills._SKILLS_DIR`/`workflows._WORKFLOWS_DIR` daemon-side (mirroring
  the `_DB_PATH` binding), which is what closes the "daemon synthesizes workflows nobody sees"
  split. 105 tests green (was 93): wire round-trip for Skill/Workflow, parametrized Local-vs-Http
  conformance (incl. a regression proving daemon-side consolidate output is client-visible, unicode
  round-trip, and shared write-validation). See `.claude/plans/composed-twirling-shore.md`.
  Upgrade daemon and client **together** (an old daemon 404s the new routes → the hooks fail).
- **Phase 3 — event log server-side — DONE (this phase).** Closed the last in-process split: the
  PostToolUse hook now writes via `client.log_event` and the Stop hook reads one task's events via
  `client.events_for_task` (both were direct `events.*` calls), so in http mode client-logged tool
  events reach the daemon and daemon-side consolidation actually mines them — instead of relying on
  the agent and daemon *coincidentally* resolving the same `data/relife.db` path (which broke the
  moment the daemon lived elsewhere). `MemoryService`/`MemoryClient` gained
  `log_event`/`events_for_task`/`event_count`; daemon routes `POST /events/log`, `GET
  /events/by-task`, `GET /events/count` (events share the DB, so `_bind_db` already covered them —
  no dir binding); `wire.py` gained `Event` helpers; `/health` now reports the event count.
  **`consolidate.py` still reads the log directly** (module-level) — it always runs where the data
  lives (daemon-side under `POST /consolidate`), same invariant as workflows. 109 tests green (was
  105): `Event` wire round-trip, event-log conformance across both transports, and the consolidate
  regression now logs *through the client* end-to-end. The recall/event/episode hooks pass
  **unedited** (they resolve the module defaults at call time via `LocalMemoryClient`).
- **Auto-consolidate throttle fix — ✅ DONE (this phase).** A scan for the *same* bug class turned up
  one more: `agent._maybe_consolidate` gated on `consolidate.should_auto_run()` (a **local** read of
  the event count + watermark) but then ran `default_client().consolidate()` **remotely** — so under
  a real remote daemon the gate reads the agent's empty local event log and auto-consolidation never
  fires (it "worked" on localhost only by the same `data/relife.db` path coincidence). Fixed by
  moving the gate to where the data lives: new `MemoryService.maybe_consolidate()` (checks
  `should_auto_run()` **and** runs, together), exposed on the client + `POST /consolidate/maybe`; the
  agent now calls `client.maybe_consolidate()`. 111 tests green (was 109): a both-transports test
  proving the throttle runs/skips server-side and advances its watermark. The rule is now a CLAUDE.md
  gotcha: a "has enough accrued?" decision about server-side work must live server-side.
- **Always-on agent + web UI (first cut) — ✅ DONE (this phase).** `relife serve` (optional
  `[server]` extra) turns the cold one-shot loop into a **long-lived process** hosting persistent
  agent sessions, streaming their work to a **self-contained web UI** over SSE, with outward-action
  approvals **routed to the browser** (approve/deny, timeout→deny) — closing the "non-interactive =
  silent deny" gap for a human-at-the-UI. Mirrors the memory-daemon patterns: side-effect-free
  `create_app()` + lazy-uvicorn `serve()` + bearer auth. New `relife/server/` (`session.py`:
  `AgentSession`/`ApprovalBroker`/`SessionManager`; `app.py`: `create_app`/`serve`), `relife/web/index.html`
  (vanilla JS, no build step), a pure `agent.to_event()` shared by the terminal renderer and the SSE
  stream, `permissions.make_approval_callback` (reuses `classify()` verbatim), and `RELIFE_AGENT_*`
  config (default `:8600`). 131 tests green (was 111): `to_event` unit tests + a model-free HTTP/SSE/
  approval suite driven by an **injected fake session factory** (no `ClaudeSDKClient` opened in any
  test — same discipline as `rem.ask_model`). Live smoke: server boots over a real socket, serves the
  UI + `/health`, and `POST /sessions` connects a real client — no turn run (Max budget preserved).
  **Local-only for now (by design):** the server binds loopback (`127.0.0.1:8600`) and the browser UI
  runs **tokenless** — `RELIFE_AGENT_TOKEN` gates the API (bearer header) but browser `EventSource`
  can't send custom headers, so token auth is for programmatic clients, not the web UI. That's fine for
  single-user local use; **exposing it beyond localhost is a deliberate future step** (needs a
  cookie/query-token scheme for the SSE stream, TLS, and multi-user auth — revisit then).
  **Deferred to a later phase:** scheduler / autonomous triggers, pre-authorized outward allowlist,
  non-loopback exposure + browser-compatible auth, multi-user, React SPA.
- **Agent-server hardening — ✅ DONE (this phase).** The first cut shipped with a caveat
  ("local-only, the UI runs tokenless"); a read of the code found that caveat was hiding a
  real privilege hole, so this phase closed it before anything autonomous is built on top.
  (1) **The UI can authenticate now.** A browser `EventSource` can't set an `Authorization`
  header, so `GET /sessions/{id}/events` — the route carrying the whole transcript *and the
  approval prompts* — was effectively unauthenticatable. `POST /auth` now exchanges the token
  for an **HttpOnly, SameSite=Strict** cookie; every route accepts bearer **or** cookie
  (constant-time compare), the UI prompts once (`GET /auth/status`), and mutating routes also
  require a same-origin `Origin` (CSRF), with failed `/auth` attempts throttled per client.
  (2) **Workspace confinement.** `POST /sessions` took an *arbitrary* path while
  `permissions.classify()` auto-allows writes **inside the session workspace** — i.e. the
  request body chose the auto-allow blast radius. Server-created workspaces are now confined
  to `AGENT_WORKSPACE_ROOT` (resolve-then-check, so `..`/symlink escapes 400). The CLI's
  `--workspace` is deliberately untouched (that's the local user speaking directly).
  (3) **Fail-closed bind.** `serve()` now *refuses* a non-loopback bind with no token instead
  of warning — this process runs shell commands and edits files.
  (4) **Ceilings + lifecycle** for a long-lived process where every session owns a
  `ClaudeSDKClient` subprocess: `AGENT_MAX_SESSIONS` (429), an idle reaper under the app
  **lifespan** that spares sessions with a live SSE stream (the heartbeat touches them),
  bounded turn queue (429), message size (413), subscriber count (429), drop-oldest on a
  lagging subscriber, `DELETE /sessions/{id}`, and `manager.aclose()` on shutdown — previously
  a server stop orphaned every agent subprocess.
  All policy lives in a new pure `relife/server/security.py` (no I/O, no framework — the
  `cognitive.py` discipline), so `app.py` only wires it to routes. **154 tests green** (was
  131): policy unit tests, resource-ceiling tests against the real session machinery, and an
  HTTP suite covering cookie auth, the throttle, CSRF, confinement and the 4xx/429 mapping —
  still **zero model calls**. Live socket smoke: guard refuses `0.0.0.0`, UI + `/health` +
  `/auth/status` serve, wrong token 401 → right token sets the cookie → SSE authenticates,
  escaping workspace 400s, clean lifespan shutdown.
  **Still not a public server:** no TLS, no multi-user. Beyond `127.0.0.1` needs a token *and*
  a reverse proxy; that remains a deliberate future step.
- **MVP pass 1 — permission holes + a mute, leaky console — ✅ DONE (this phase).** An audit
  ahead of the MVP work found the two headline promises were not actually held.
  (1) **The permission policy stopped at the shell.** `classify()` gated `Bash`/`PowerShell`
  with a *Bash-flavoured* denylist, so on the platform ReLife runs on, `Send-MailMessage`,
  `Invoke-RestMethod -Method POST`, `Invoke-WebRequest -OutFile` and `Start-Process -Verb
  RunAs` were all **auto-allowed** — and because containment applied only to `Write`/`Edit`,
  so were `echo pwned > ~/.bashrc`, `cp secrets.env /etc/app.env` and `rm -rf ~/Documents`.
  Fixed by (a) `_OUTWARD_SHELL`, which covers both shells (PowerShell outward verbs, remote
  sessions, publish, elevation, `mkfs`/`dd of=/dev/`/`Format-Volume`, and a download piped
  into an interpreter), and (b) a **containment rule for the shell**:
  `_write_targets`/`_delete_targets` extract redirect/copy/delete destinations and `_escapes`
  asks unless every one is provably inside the workspace. `_under()` now `expanduser()`s, or
  `~/Documents` reads as a *relative* path inside the workspace. The extraction is knowingly
  heuristic and tuned to err toward asking; for deletes, an unevaluable target (`rm -rf
  "$DIR"`) counts as outside. Verified no new friction on ordinary work (`pytest -q > out.txt
  2>&1`, `rm -rf node_modules`, `git push`, loopback GETs all still auto-allow).
  (2) **The web console never showed a tool result.** `to_event` walked only
  `AssistantMessage`, but the CLI delivers results on a `user` frame — so the UI streamed
  every tool call and no output. The existing test built the shape the runtime never emits,
  which is why it was green; it now asserts the real carrier.
  (3) **Every page reload forked a new agent.** The UI unconditionally `POST`ed `/sessions`,
  so a refresh spawned a second `ClaudeSDKClient` subprocess, orphaned the first for an hour
  (the idle reaper), threw away the transcript, and wedged at 429 after `AGENT_MAX_SESSIONS`
  reloads with no way to clear it. The page now remembers its session in `localStorage` and
  reattaches via a new `GET /sessions/{id}`, replaying the ring with `?last_id=0`; a header
  **"new session"** button `DELETE`s the old one; a 404 mid-send self-heals. **167 tests
  green** (was 154), still zero model calls. Real-socket smoke (fake session, no model):
  create → stream → probe 200 → reconnect with `last_id=0` replays the full transcript
  *including* `tool_result` → DELETE → 404, sessions back to 0.
- **MVP pass 2 — a real outward capability + `relife doctor` — ✅ DONE (this phase).**
  (1) **Gmail (and Calendar/Drive) via the claude.ai connectors.** A one-turn live probe
  confirmed the connectors are attached to every ReLife session by the CLI (`mcp_servers`
  at init: `claude.ai Gmail`, `claude.ai Google Calendar`, `claude.ai Google Drive`,
  alongside `browser` and `relife_memory`) — `setting_sources=None` doesn't exclude them,
  they come from the account (OAuth scope `user:mcp_servers`), so there are no keys and no
  local server to run. Until now they fell into the "unrecognized tool → ask" default, so
  even a search prompted. `permissions.classify()` now has a **verb-based, fail-closed
  connector policy** for `mcp__claude_ai_*`: read verbs auto-allow, write verbs ask,
  neither asks, write beats read — deliberately independent of the exact tool names Google
  ships (which only appear after the user links their Google account in-session via the
  connector's `authenticate` tool). The TTY prompt and the UI approval card now show the
  call's leading fields (`_tool_brief` fallback), so what the user approves is a concrete
  `to=… subject=…`, not a blank line. `prompts/system.md` tells the agent the rules: show
  recipient/subject/body before sending, a denial is final, never route mail elsewhere.
  (2) **`relife doctor`.** Every first-run dependency failed late and cryptically inside a
  subprocess. `doctor.py` is a pure `run_checks(Probes)` (the `security.py`/`cognitive.py`
  discipline — no subprocess/fs/network of its own; `default_probes()` is the one real-machine
  seam), checking: Python ≥3.11; the CLI the SDK will actually run (bundled first, PATH
  second, same order as the SDK) and `claude auth status` (parsed JSON → who/subscription);
  `ANTHROPIC_API_KEY` *unset* (warn if set — it would bill metered usage); node/npx (fail —
  the browser MCP needs it); `gh` incl. the winget dir `agent_env()` prepends (warn only);
  SQLite FTS5 (warn — recall degrades); data dir writable; each optional extra with its
  `pip install -e ".[x]"` hint; the memory daemon's `/health` only when `RELIFE_MEMORY_URL`
  is set; and the three connectors via `claude mcp list`. Exit 1 on a blocker. 17 scripted
  tests; the real run on this machine is all-green (note: this account is a **pro**
  subscription per `auth status`, not Max — same auth path either way).
- **MVP pass 3 — memory is inspectable and correctable — ✅ DONE (this phase).** The
  differentiating feature had one read-only view (`memory stats`). Now: `relife memory
  search` (the hook's ranking, **without reinforcing** — inspecting must never change what
  the agent is shown next), `list` (`--kind`, `--archived`, `--sort recent|strong|oldest`),
  `show <id>` (text, tags, importance, activation, use history), and `forget <id>… |
  --query` (archive, reversible, confirms unless `--yes`). Forgetting by *id* needed
  precise access the seam lacked — query-based `forget()` archives whatever matches best —
  so `MemoryService`/`MemoryClient` gained `archive(id) -> bool` and `get(id)`, with daemon
  routes `POST /archive` and `GET /memories/{id}` and a Local-vs-Http conformance case.
  11 CliRunner tests over an isolated store. 203 tests green (was 190).
- **MVP pass 4 — consolidation off the server's event loop — ✅ DONE (this phase).**
  `AgentSession._run` called the sync `_maybe_consolidate()` inline on the server's single
  loop, so a sweep (SQLite scans, dedupe, ONNX inference with embeddings on — seconds on a
  big store) froze every session's SSE stream and every pending approval for its duration.
  `agent.maybe_consolidate()` now returns the report (no printing) behind a process-wide
  **non-blocking lock** (two sessions finishing together don't sweep twice; the loser gets
  `None` and the throttle fires next turn); `maybe_consolidate_off_loop()` runs it via
  `anyio.to_thread` and returns a one-line note the session publishes as a `note` event
  (rendered dim in the UI). The CLI's `_maybe_consolidate()` still runs inline and prints —
  nothing else is on that loop. Thread-safety checked: the store and event log open a fresh
  SQLite connection per call, nothing is shared across threads. Tests drive the **real**
  `AgentSession._run` with a one-turn fake client and a 0.3s fake pass: the pass ran on
  another thread and a 10ms ticker kept turning throughout (≥10 ticks), a silent pass
  publishes nothing, and the lock serializes concurrent callers. 206 tests green (was 203).
- **Scheduler / autonomous triggers — ✅ DONE (this phase).** The first thing built on top
  of the hardened server: `relife/server/schedules.py` (pure cadence logic + `Schedule`
  record + JSON-file `ScheduleStore` at `data/schedules.json`) and `scheduler.py` (one
  lifespan task that fires each due schedule as a turn in a per-schedule `AgentSession`).
  Design choices, each a deliberate answer to "what does unattended mean here": a run goes
  through the **same** session/approval machinery as a typed turn, so with nobody watching
  an outward action times out to deny (and the prompt tells the agent so); a slot missed
  during downtime fires **once** (advance from now, never a catch-up burst); a session still
  busy from the previous run **skips** the slot instead of queueing behind it; failures to
  start are recorded in the schedule's bounded history and the schedule still advances; the
  interval form is floored (`AGENT_SCHEDULE_MIN_INTERVAL`, 5 min) because every run spends
  Max budget; workspaces are confined exactly like `POST /sessions`. Two spec forms:
  `every: 30m|2h|1d` and `at: HH:MM` (+ weekdays) in local wall-clock time (DST-safe via
  naive-local arithmetic). Routes `GET/POST /schedules`, `GET/PATCH/DELETE /schedules/{id}`,
  `POST /schedules/{id}/run`; the web UI gained a schedules panel (add / pause / run now /
  delete / **watch** = attach the console to the schedule's session). `AgentSession.busy` is
  new. Smoke-tested in a real browser against a scripted fake session: add → run now →
  console attached and streamed the run; the lifespan tick then fired the next slot on its
  own. 232 tests green (was 206): cadence math at fixed instants, scheduler policy against a
  fake manager, the routes over `TestClient` (`run_scheduler=False`, tmp store).
- **MVP pass 6 — scheduled runs deliver unattended — ✅ DONE (this phase).** The scheduler
  exposed the gap it created: a 09:00 run fired with nobody watching left its transcript
  only in the session ring buffer, which the idle reaper wiped an hour later — the panel
  said `submitted` (= queued) and nothing else. Now `relife/server/runs.py` holds a durable
  `RunRecord` per firing (closing summary, tool count, cost, **every approval denied because
  no one was there**, the event stream) under `data/runs/<schedule>/`, written by a recorder
  task that subscribes to the session *before* the turn is submitted and takes exactly that
  turn (matched by the `user` echo; a turn the user typed into the same session is ignored),
  ending on `result`/`error`, a run timeout, or shutdown (`interrupted`). The schedule's inline
  history entry is upgraded in place from `submitted` to the outcome. Routes
  `GET /schedules/{id}/runs[/{run_id}]`; the panel shows the last summary on the card and a
  per-run list with a "needed you — denied unattended" block. Browser smoke with a scripted
  session that hits a denied approval: card read `done · 1 tool · $0.010 · 1 denied` with the
  agent's closing note and the exact `gh pr create` it could not run. 239 tests (was 232).
- **MVP pass 7 — `relife doctor` knows the always-on side — ✅ DONE (this phase).** The
  scheduler introduced a class of failure that is *silent* rather than late: schedules only
  fire inside `relife serve`, an unattended run just records a denial, a non-loopback bind
  without a token is refused at start. `doctor` now checks the model/effort in effect, the
  workspace root (writable — it's the auto-allow radius), the agent server (bind/token sanity
  via the same `guard_bind`, then `/health`: **skip** when not running, **warn** when not
  running *and* enabled schedules exist — "N schedules will not fire"), the schedules'
  last outcomes in one line ("last run failed: …; needed you: inbox (2 denied)"), and a
  stray memory-daemon sidecar with `RELIFE_MEMORY_URL` unset. Connector advice was wrong for
  the real `Needs authentication` state (it said re-enable; it now says link the Google
  account in-session). `--json` for scripts. Same pure `run_checks(Probes)` shape — new
  probe fields default so older bundles stay valid. 248 tests (was 239).
- **MVP pass 8 — hardening sweep — ✅ DONE (this phase).** (1) Stop-hook **episodes are scoped
  to the turn**, not the session: the event log is keyed by session id and a chat/server session
  runs many turns, so the prompt hook now stashes the session's latest event id and the Stop hook
  only takes events after it (every later episode used to replay the whole session's tools and
  skew the pattern detector). (2) **All three hooks fail soft** — daemon down / store error ⇒ no
  injected context / nothing captured, never an exception into the SDK. (3) **Run-id path
  traversal closed**: `RunStore.get` refuses any id not shaped like `RunRecord.new_id`
  (`YYYYMMDD-HHMMSS-mmm`) before it names a file. (4) A scheduled run with every SSE slot taken
  still goes out but records `submitted (unrecorded: …)` instead of a `submitted` no recorder
  would ever upgrade. (5) A `[tool.ruff]` baseline (`python -m ruff check relife tests scripts`,
  zero findings). 252 tests (was 248).
- **MVP pass 9 — schedule grants (pre-authorized outward actions) — ✅ DONE (this phase).** An
  unattended run had no one at the approval card, so "summarize my inbox and email me" could
  never finish. A schedule may now carry **grants**: `email` (Gmail send/reply/forward/draft) and
  `calendar` (create-event), each bound to ≤5 listed addresses. Policy is pure and fail-closed in
  `permissions.py` (`normalize_grants`/`grant_allows`): connector ask-cases only (never shell/file/
  unknown tools), destructive ops excluded even if the op also matches, every address outside
  free-text fields must be listed, a recipient field holding a non-address fails, and email needs
  a visible recipient (thread-inferred reply / raw MIME ⇒ ask). Grants ride **one turn**
  (`AgentSession.submit(text, grants=)`, so typing into the same session gets none), are capped
  per run by `AGENT_GRANT_MAX_USES` (3; bounds a loop or a prompt-injected "mail me 50 times"),
  and every use is an `approval_auto` event → the run record's `acted` list ("done for you —
  pre-approved" in the UI). The scheduled preamble tells the agent what it may do. Schedules
  accept `grants` on POST/PATCH; an invalid hand-edited grant is dropped on load, never widened.
  UI: pre-approve fieldset on the add form, grants line on the card. 286 tests (was 252).
  **Not yet exercised live** against the real Gmail/Calendar connectors — their input field
  names aren't known here, so the recipient check is deliberately schema-agnostic; a first live
  run may show a legitimate send falling back to ask (the safe direction).
- **MVP pass 10 — verb-based GitHub CLI policy — ✅ DONE (this phase).** `gh` was gated by
  *group* (`gh pr|issue|release|api|gist` → ask), wrong in both directions: reading the
  user's own work items (`gh issue list --assignee @me`, `gh pr view`) prompted — so a
  scheduled "triage my issues" run was denied before it could look — while `gh repo delete`,
  `gh repo edit --visibility public`, `gh secret set`, `gh workflow run` and `gh auth login`
  ran **unasked**. Now `_gh_outward` (pure, in `permissions.py`) requires *every* `gh`
  occurrence in the command — wrapped in `bash -c`, after `time`/`xargs`, a full `gh.exe`
  path — to be a known read (`_GH_READ`) or `gh repo create|clone` (the v1 create→push flow,
  kept autonomous like `git push`); everything else asks. `gh api` allows only a REST GET (no
  method and no field flags, or an explicit `-X GET`); `graphql` always asks (it can mutate).
  The system prompt tells the agent what reads freely and to show what it will post. 291 tests
  (was 286).
- **MVP pass 11 — memory off the server's event loop — ✅ DONE (this phase).** Pass 4 moved
  auto-consolidation to a thread, but the rest of memory still ran inline: the three hooks and
  the memory MCP tools are `async` callers of the *sync* client, and under `relife serve` they
  share one loop with every session — each recall (FTS + ONNX with embeddings), each
  journaled tool call (a loopback POST in daemon mode) and an agent-called
  `memory_consolidate` froze every other session's stream and pending approval. New
  `memory.client.off_loop(fn, …)` (`anyio.to_thread`) wraps every such call; thread safety is
  the pass-4 argument (fresh SQLite connection per call, locked model load, thread-safe
  `httpx.Client`). This delivers the practical goal of the planned async `MemoryClient`
  variants without a second client surface. Tests drive the real hooks/tools against a
  blocking fake client with a 10ms ticker on the same loop (fail on the old code: 6/6).
  297 tests (was 291).
- **MVP pass 12 — work items: GitHub issues end-to-end — ✅ DONE (this phase).** The vision's
  "complete assigned work items" now exists: `relife work` lists open issues assigned to the
  user; `relife work owner/repo#12` (or an issue URL, or `12 --repo …`) fetches the issue,
  clones the repo once into `<workspace>/<owner>__<name>`, and runs the agent *in that checkout*
  (the auto-allow radius is one repo) on a branch `relife/issue-12-<slug>` → implement → test →
  commit → `git push` (autonomous) → `gh pr create` with "Closes #12" (asks). The issue text is
  third-party input, so the prompt fences it as untrusted data and bounds it; the policy is the
  backstop. An existing checkout is never reset by our code (uncommitted work → the agent stops
  and says so); a closed issue exits before cloning; `--dry-run` prints the task without a model
  call. All plumbing in `relife/workitems.py`, tested against a fake `gh` (22 tests). Smoke:
  `relife work` against the real account (read-only) returned cleanly. A live end-to-end run on
  a real issue is pending budget. 319 tests (was 297).
- **MVP pass 13 — work schedules — ✅ DONE (this phase).** A schedule can now *be* `relife work`
  on a cadence: `work: {}` (or `{"repo"}`/`{"label"}`) makes each firing take the next open
  assigned issue it hasn't attempted, check it out under the schedule's workspace, and run the
  issue prompt in a fresh session in that checkout. Selection/fetch/clone are deterministic and
  run on a worker thread; an empty poll is `skipped` **before** a session is created (no budget
  spent). An issue counts as attempted only once its turn is submitted. Unattended, the PR
  creation is denied by timeout and shows up in the run's "needed you" list — the branch is
  pushed. Web UI: a "work my assigned GitHub issues" toggle (repo/label, task becomes optional)
  and a `⑂` line on the card. Verified in Chrome against an isolated server (fake sessions, temp
  files). 324 tests (was 319).
- **MVP pass 14 — the pull-request grant — ✅ DONE (this phase).** An unattended work schedule
  could push its branch but not open the PR (the ask timed out to deny). A new grant kind,
  `pull_request`, is the first (and only) grant that touches the shell, so it is the narrowest:
  allowed only on a work schedule, stored *unbound*, bound per turn by the scheduler to that
  issue's repo + branch, and then it pre-approves one exact shape — a single `gh pr create` with
  matching `--repo`/`--head`, only title/body/base/draft/fill flags, no `$`/backtick, no operator,
  no extra argument (no `--body-file`, reviewers, labels, `--web`). Everything else still asks;
  the per-turn use cap and `approval_auto` journaling apply as for email/calendar. The agent is
  told the exact shape in the scheduled preamble. UI: "may open the PR without asking" under the
  work toggle. 36 new tests, mostly bypass attempts (chains, newlines, `$(…)`, `$env:`, backticks,
  `--body-file`, wrong repo/head, `bash -c` wrapping). 360 tests (was 324).
- **Post-1.0 — grants checked against the real Google connectors — ✅ DONE.** Linking Gmail +
  Calendar in a Claude Code session (same account connectors ReLife gets) exposed the real tool
  names and schemas without a ReLife turn. Findings: Gmail **can** send (`send_message`, `reply`,
  `forward` — the "drafts only" note was stale), and two real grant bypasses: `send_message(draftId=…)`
  sends a stored draft ignoring the call's `to`, and `reply(replyAll=true)` keeps the thread's CC —
  both passed the address check with `to=[me]`. Both now ask. Also fixed false asks: camelCase
  content fields (`htmlBody`, `forwardText`) were scanned as recipients, `get_draft`/`suggest_time`
  asked; and an attendee `{email: "ops-team"}` is now junk. The scheduled preamble names the
  covered email shape. Tests pin every real tool name and shape (fail 7/18 on the old code).
- **Post-1.0 — live budget smokes (2026-10-09), all in scratch `RELIFE_HOME`s — ✅ PASSED.**
  (1) Email grant on a schedule via `relife serve` + `POST /schedules/{id}/run`: draft to the
  listed address created with no card (`acted`), draft to another address asked → timed out →
  `denied`; Gmail shows only the first draft ($0.39). (2) `relife dream --max 10` on a copy of real
  data: 2 duplicate patterns archived + journaled ($0.21). (3) Skills round-trip: run 1 scaffolded a
  CLI and wrote `scaffold-python-cli` ($0.92); run 2 found and followed it ($0.37). (4) `relife
  build` killed after milestone 1, `--resume` (no persisted session yet → fresh-session path)
  finished 2–3 without redoing 1, 39 tests ($0.76). The smokes surfaced a real shell-gate hole:
  `rm -rf /d` was auto-allowed (`_FLAG` read `/d` as a cmd switch for every verb; in Git Bash it is
  drive D:) — fixed, slash switches only for cmd built-ins; plus a false ask: Git Bash `/d/…` paths
  inside the workspace now translate (`_msys_to_windows`, `Bash` tool on Windows only).
- **Platform pass — agents, memory handoff, memory for any LLM, CrewAI crews — ✅ DONE (2026-10-10).**
  The user asked for ReLife to become a platform: CrewAI spins up ReLife agents for a task, memory
  can be handed from old agents to new ones, and other LLMs connect as agents with memory attached.
  Decisions (asked, all the recommended options): other models via any provider with keys optional
  (Claude stays on Max — `ClaudeMaxLLM`, never an API key); handoff default = own space + inherit
  read-only, with fork / promote / packs; this pass = foundation + MCP + CrewAI CLI (server/UI next);
  crews run from a Python 3.12 `.venv` because CrewAI 1.15 requires `<3.14`.
  (1) **Memory spaces** (store schema v3): every memory/event has a `space` (+ `source` provenance);
  skills/workflows per space; recall, save-dedupe, doc-freq, vector search and consolidation are all
  space-scoped — consolidation never merges or learns across spaces. Found + fixed on the way: a
  fresh DB stamped the *current* schema version before later steps ran (now stamps v2, then migrates).
  (2) **Agents** (`relife/agents.py`, `data/agents.json`): a profile resolves to a `MemoryScope` —
  writes only its own space (never `default`), reads own + inherited (transitive) + `default` unless
  `--isolated`. `ScopedMemoryClient` enforces it: no space arguments exist, admin ops and `dream`
  refuse. Handoff: inherit (live, read-only), fork (snapshot copy, strength + provenance kept),
  promote (the explicit path into the user's memory), export/import packs (validated, `import:<space>`).
  (3) **Memory over MCP**: tools defined once (`memory/tools.py`); the in-process SDK server is
  unchanged for Claude; a standalone server (`memory/mcp_server.py`) serves the external set (no
  consolidate/dream, plus `memory_context`) over stdio (`relife mcp --agent NAME`) and streamable HTTP
  at `/mcp` on the memory daemon (agent bearer token, re-read per request; DNS-rebinding protection).
  Real-stdio smoke on Windows found a **deadlock**: with embeddings on, the first `tools/call` hung
  (initialize/list answered) because fastembed/onnxruntime was constructed on a worker thread while the
  stdio reader blocked on stdin — fixed by warming the model before serving (regression test pins the
  order; the smoke is in RELEASE_TESTING).
  (4) **Crews** (`relife/crew/`): planner → validated `CrewSpec` (caps, backward-only task refs, known
  lineage; one retry, then a single-agent fallback) → record → shown + confirmed → profiles created
  with the plan's inherit/fork → `Crew.kickoff()`. ReLife members are `ReLifeAgent(BaseAgentAdapter)`
  (each task a fresh ReLife turn in `<workspace>/crews/<id>/`, ReLife permissions, scoped memory; CrewAI
  tools exposed as `crew_tools`, outside the trusted prefix); other-model members are CrewAI agents with
  ReLife memory tools and the recall block, and no machine-touching tools. CrewAI memory/planning/
  telemetry off. Two CrewAI 1.15.27 contract surprises, both from reading the pinned source: all of
  `execute_task/aexecute_task/create_agent_executor/get_delegation_tools/get_platform_tools/get_mcp_tools`
  are abstract (its own OpenAI adapter predates that), and a crew member must expose
  `function_calling_llm`, `step_callback` and `last_messages`.
  Tests: 790 on 3.14 (crew module skips), 797 on the 3.12 venv with `[crewai]` — including a real
  `Crew.kickoff()` with a ReLife agent and a CrewAI agent on `ClaudeMaxLLM`, zero model calls. CI's
  `full` job adds `[crewai]`. Smokes (no budget): stdio MCP via the SDK's own client on Windows;
  `relife crew --spec crew.yaml --plan-only` + `relife crews ID`; doctor on both interpreters.
  **Live crew smokes (2026-10-10, scratch `RELIFE_HOME`, 3.12 venv) — ✅ PASSED.** (1) `relife crew
  --plan-only` on a "slugify module + tests + edge-case review" task: valid plan first try (a ReLife
  `python-dev` + a `claude-max` `code-reviewer`, review ← implement), 18 s, $0.26. (2) That plan as
  `--spec`, with `inherit: [python-dev]` added to the reviewer, confirmed via the real `[y/N]` prompt:
  agents created with the lineage; the builder wrote `slugify.py` + `test_slugify.py` in
  `workspace/crews/<id>/` (14 tests pass); the reviewer got the output as context, called
  `memory_context`, and wrote a substantive review (found real bugs: `ß/ø/ł` dropped, em-dash words
  merged); `relife crews <id>` showed both outcomes; status `done`, $0.47. Isolation held: the builder's
  events + episode in `python-dev`, the reviewer's journaled step in `code-reviewer`, `default`
  untouched. Fixed from what it surfaced: CrewAI's "function callbacks cannot be serialized" warning on
  every run (our step callback is necessarily a closure; ReLife never checkpoints — silenced at agent
  construction only), and crew episodes all reading "Task: You are working on a crew as …" (the crew
  prompt now leads with the task, which is what the episode keeps).
- **Platform pass 2 — the platform in the server (2026-10-10, branch `feat/platform-server`).**
  (1) **Routes:** `GET /spaces`; `GET/POST /agents`, `GET/DELETE /agents/{name}`,
  `POST /agents/{name}/attach|detach|promote` (the `relife agent` operations; registry re-read per
  request, memory work off the loop; token minting stays CLI-only); `GET/POST /crews` (plan a task —
  one model call — or record a spec; over HTTP a spec may only name `RELIFE_CREW_LLMS` + `claude-max`),
  `GET /crews/{id}`, `POST /crews/{id}/run|stop`, `DELETE /crews/{id}` (discard a plan). Same auth +
  same-origin guard as every route. (2) **Crews hosted like sessions** (`server/crews.py`): a
  `CrewHost` publishes through the `EventStream` base now shared with `AgentSession`, and is adopted
  by the `SessionManager` — so `/sessions/{id}/events`, the approval route, the UI's watch and the
  scheduler's recorder all work on a crew unchanged, and a crew counts against the session ceiling,
  reaper (never reaped while running) and shutdown. `kickoff()` runs on a daemon thread; events cross
  back with `run_coroutine_threadsafe`; a member's `can_use_tool` is the ordinary
  `make_approval_callback` over a `LoopBroker` that hops each ask onto the server loop's broker — so an
  outward action is an approval card, and unattended it times out to deny. Members' `result`/`error`
  become `member_done`/`member_error` (one `result` per crew). Stop is cooperative: pending approvals
  denied at once, no new member task starts. `AGENT_MAX_CREWS` (default 1). `run_crew` split into
  `prepare_crew` + `execute_crew` for this. (3) **Crew schedules:** a schedule may carry a validated
  crew spec (`crew`); each firing re-validates it against the live registry (no model call), hosts it
  and records the run like any other; no grants on a crew schedule, never both crew and work.
  (4) **UI:** Agents panel (list with memory counts + what each reads, add with inherit/fork/isolated,
  attach/detach chips, promote, delete) and Crews panel (plan → review → run/discard, watch, stop, run
  again, per-task outcomes, schedule it); crew events name the member; leaving a crew's stream with
  "new session" never stops it. Found on the way: `asyncio.to_thread` copies context variables, so
  `anyio.run` inside the worker believed it was already in the loop ("Already running asyncio in this
  thread") — the planner now runs via `run_in_executor`, the crew on a plain thread; and a fast second
  `run` could read the record as still `planned` before the worker saved `running` — the live host is
  checked first. Tests: 14 new (`tests/test_server_platform.py`, 4 need CrewAI): 811 on the 3.12 venv,
  800 on 3.14 (full suite 10× in a row clean after the race fix below).
  **Live check from the browser (2026-10-10, scratch `RELIFE_HOME`, 3.12 venv) — ✅ PASSED** (~$1.15 in all).
  Planned in the crews panel (valid first try, a ReLife `python-builder` + a `claude-max` `code-reviewer`,
  $0.25); run: the console switched to the crew's stream, events labelled per member; the builder wrote
  greet.py + tests (pass) and its `curl -X POST http://127.0.0.1:8612/health` came up as an approval
  card naming the member — approved, it ran (405, as expected); the reviewer called `memory_context` and
  reviewed the builder's output; status `done`, $0.65. Memory: the builder's episode in its own space,
  `default` untouched. Found and fixed: (a) **stop didn't stop a CrewAI member** — it only refused the
  next *ReLife* turn, so the reviewer still ran and the crew ended `done`; the crew's `task_callback`
  now raises once stop is set (`build_crew(should_stop=)`), so no task of any kind starts after the
  one in flight — re-checked live: card denied, builder finished, reviewer never called, record
  `error: crew stopped by the user` with only the builder's task; (b) the crew card read `planned`
  (offering *run*) until the worker saved `running` — the live host now overrides the status;
  (c) the full suite then exposed a **Windows read/replace race**: a route reading `record.json` while
  the worker `os.replace`s it raises `PermissionError` on one side — `CrewRunStore` retries both
  briefly. Also: the panel buttons now appear once auth passes, not when the chat stream opens.
- **Phase 3 (next):** async `MemoryClient` variants are no longer needed (pass 11 moved every
  async caller off the loop via `off_loop`); the live smokes are done (above); Anthropic **Managed Agents** is the natural host (hosted memory
  stores, MCP vaults, GitHub mounting, scheduled deployments).

## 9. Key facts to remember

- GitHub account: **`anishkun`** (= anish03anish@gmail.com). `gh` authed, `repo` scope, HTTPS.
- Approved plan lives at `C:\Users\HP\.claude\plans\witty-enchanting-fountain.md`.
- Memory about auth constraint: `…/.claude/projects/D--relife/memory/auth-via-max-subscription.md`.
