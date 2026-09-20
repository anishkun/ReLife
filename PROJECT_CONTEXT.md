# ReLife — Project Context & Handoff

> Durable reference for future sessions. Captures *why* things are the way they are,
> what's built and verified, the non-obvious gotchas, and what's next.
> Last updated: 2026-07-04.

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

**Tests:** 131 passing (`python -m pytest tests/`). Covers permission classify, store
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
  permissions.py            # classify() + make_permission_callback() (can_use_tool); connector verb policy
  doctor.py                 # `relife doctor`: pure run_checks() over injected Probes
  hooks.py                  # UserPromptSubmit recall hook (memory + skills)
  prompts/system.md         # persona + safety + memory/skill instructions
  memory/                   # cognitive memory: relevance rises w/ use, fades when idle
    cognitive.py            # pure ACT-R-style activation/fused_score/should_archive/should_hard_delete
    store.py                # injectable MemoryStore: SQLite (user_version migrations), two-stage fused recall, decay
    vector_index.py         # VectorIndex seam: BruteForceIndex + soft-optional SqliteVecIndex (self-tested)
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
  build/                    # `relife build`: orchestrated, resumable large builds
    ledger.py               # BuildLedger — durable plan+progress (data/builds/<id>/)
    server.py               # relife_build MCP server (plan_set/milestone_update/status)
    agents.py               # `builder` subagent definition (Task-delegated milestones)
    orchestrator.py         # run_build(): decompose → delegate → resume
    prompts/orchestrator.md # orchestrator persona (architect/PM, delegates building)
data/                       # gitignored runtime: relife.db, skills/, builds/, logs
scripts/bench_recall.py     # non-CI recall scaling benchmark (10k+ memories)
tests/                      # 111 tests (107 deterministic + 4 semantic, embeddings forced off)
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
- **Phase 3 (next):** scheduler / autonomous triggers on top of the agent server; async
  `MemoryClient` variants (today the sync recall/save/log briefly block an async caller's loop —
  acceptable at loopback, only `dream` is offloaded; events now add one loopback POST per tool call
  in http mode); outward capabilities
  (email/calendar/work-items) — Anthropic **Managed Agents** is the natural host (hosted memory
  stores, MCP vaults, GitHub mounting, scheduled deployments).

## 9. Key facts to remember

- GitHub account: **`anishkun`** (= anish03anish@gmail.com). `gh` authed, `repo` scope, HTTPS.
- Approved plan lives at `C:\Users\HP\.claude\plans\witty-enchanting-fountain.md`.
- Memory about auth constraint: `…/.claude/projects/D--ReLife/memory/auth-via-max-subscription.md`.
