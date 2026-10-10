# ReLife — Module-by-Module Deep Dive (Architectural Study Guide)

> A complete, ground-up explanation of the ReLife codebase organized as the **M1–M16
> tutoring curriculum**: what the system is, how data flows, *why* each decision was
> made over the obvious alternative, and where the trade-offs and weak points live.
> Written to be *read and re-read* — every module is covered in detail, with the
> causal reasoning and failure modes spelled out, not just the facts.
>
> Companions: `HOW_IT_WORKS.md` is the friendly plain-English walkthrough;
> `CLAUDE.md` is the working ruleset; `PROJECT_CONTEXT.md` is the locked decisions.
> This file explains the *reasoning behind the architecture*.
>
> **Reading order.** M1–M7 are the core (the agent, its safety gate, its brain).
> M8–M11 are the *always-on* layer that was added afterwards (memory as a daemon,
> the agent server + web console, scheduled unattended runs, and the operator
> surface). M12 is the agent doing the user's own work unattended (GitHub issues,
> pre-approved actions). M13–M15 are the *platform* layer (many agents with their own
> memory, memory for any LLM over MCP, CrewAI crews staffed with ReLife agents). M16
> is the synthesis — read it last, and re-read it whenever you make an architectural
> change. Line references are to the code as of the platform pass (branch
> `feat/platform-crewai`, commit `a53ef91`); the test suite is **~800 deterministic
> tests** — 790 on Python 3.14, 796 in the 3.12 `.venv` where CrewAI is installed
> (`python -m pytest tests/` — use the interpreter ReLife is installed into; on this
> machine that is `py -3`, or `.venv\Scripts\python` for the crew tests).

---

## Table of contents

- [M1. The 30,000-ft view — the central bet](#m1)
- [M2. The agent runner & the SDK seam](#m2)
- [M3. The permission model](#m3)
- [M4. The cognitive core (`cognitive.py`)](#m4)
- [M5. The memory store (`store.py`) + the seams around it](#m5)
- [M6. Hooks & the learning loop (`hooks.py`, `consolidate.py`, `rem.py`)](#m6)
- [M7. Build orchestration (`relife/build/`)](#m7)
- [M8. Memory as a process — the daemon split (`memory/remote/`)](#m8)
- [M9. The always-on agent server + web console (`relife/server/`)](#m9)
- [M10. Autonomy that delivers — scheduler & run outcomes](#m10)
- [M11. The operator surface — connectors, `doctor`, memory inspection](#m11)
- [M12. Doing the user's work — GitHub issues and pre-approved actions](#m12)
- [M13. Many agents, one store — memory spaces & handoff](#m13)
- [M14. Memory for any LLM — the MCP surface](#m14)
- [M15. Crews — CrewAI plans, ReLife staffs](#m15)
- [M16. Trade-offs, failure modes & "why not X"](#m16)
- [Appendix A. The complete request lifecycle (three traces)](#appendix-a)
- [Appendix B. File-by-file index](#appendix-b)
- [Appendix C. Every tunable in `config.py`](#appendix-c)

---

<a name="m1"></a>
## M1. The 30,000-ft view — the central bet

### What ReLife is

ReLife is a **personal agent** built on the **Claude Agent SDK** (Python). It does
two things ordinary scripted agents don't:

1. It **acts** in the world through tools and MCP servers (shell, files, a browser,
   git, GitHub — and, via the claude.ai connectors, Gmail / Calendar / Drive).
2. It **learns over time** — it accumulates *facts* (long-term memory) and
   *procedures* (skills + workflows), and it reshapes that knowledge with brain-like
   maintenance passes.

And since the always-on layer landed, it does a third thing: it **keeps running** —
a long-lived server hosts persistent sessions, streams them to a browser console,
and fires **scheduled tasks unattended**, recording what each one did and what it
needed from you — including working the user's assigned GitHub issues (M12).

And since the platform pass, a fourth: it is **more than one agent.** Registered
agents each get their own memory *space*; a new agent can be handed an old agent's
memory (read it live, or start from a copy); agents on *other* LLMs — Cursor, Gemini
CLI, a CrewAI agent on Ollama — attach to ReLife memory over MCP; and `relife crew`
lets CrewAI plan a team for a task that ReLife then staffs and runs (M13–M15). The
bet doesn't change — experience is the moat — it just stops being locked to one
agent.

### The one big idea (the central bet)

The model itself is a **commodity**. Anyone can call Claude. What can't be copied is
**the accumulated, personalized experience** wrapped around the model: what *this*
agent has learned about *this* user and *their* projects. ReLife bets that the
durable value is in the **memory and procedural layers**, not the raw model.

Everything else follows from that bet. If the moat is accumulated experience, then:

- memory must be **cheap to maintain** (you'll run maintenance constantly),
- it must **degrade gracefully** (a learning system that corrupts itself is worse
  than useless),
- and it must be **portable across model versions** (the commodity underneath will
  change).

### The economic constraint that shapes everything: Max, not API

ReLife runs on the **Claude Code Max subscription** — the logged-in `claude` CLI —
**not** a metered `ANTHROPIC_API_KEY`. The key is *intentionally unset*; the SDK
drives the CLI that is already authenticated against Max. (`relife doctor` even
**warns if the key is set**, `doctor.py:233` — a key in the env would silently bill
metered usage instead.)

This is not a minor deployment detail. It is the single biggest force on the
architecture:

- **Every model call spends the same finite session budget the user's real work
  spends.** There is no separate "background" budget. A token spent on bookkeeping
  is a token *not* available for the task.
- Therefore: **anything that runs automatically and frequently must be LLM-free.**
  This is why consolidation (the constant "sleep" pass) is pure deterministic code,
  and the only LLM-driven memory pass (REM/"dream") is **opt-in and never
  auto-runs**. (See M6.)
- It also forbids hosted embeddings — so semantic recall uses a **local, offline**
  ONNX model with no API key (see M5).
- It even shapes the scheduler: an interval schedule is **floored** at
  `AGENT_SCHEDULE_MIN_INTERVAL` (300s) because "every 1m" is a budget mistake, not a
  wish (see M10).
- And it shapes the platform: when CrewAI needs Claude, it gets it **through the CLI**
  (`ClaudeMaxLLM`, M15), never an API key; agents on other models bring *their own*
  keys through CrewAI/LiteLLM's environment variables, and ReLife never reads or
  stores them. A crew's plan is shown and confirmed before it spends anything.

> If you remember one thing from M1: *"runs on the subscription, not an API key"*
> is the reason half the rest of the system looks the way it does.

### The per-task loop (the four beats)

Every task ReLife runs follows the same rhythm:

```
recall  →  act  →  reflect  →  consolidate
```

- **recall** — before the task, relevant memories, skills, and workflows are
  *injected automatically* (the agent doesn't have to ask). Surfacing something
  *reinforces* it.
- **act** — the agent uses tools, each gated by the permission policy and journaled
  to an event log.
- **reflect** — the agent may explicitly save memories/skills/workflows; and even
  if it doesn't, a deterministic *episode* of the run is captured automatically.
- **consolidate** — after the run, a cheap "sleep" pass fades unused memories,
  merges duplicates, and mines the event log for recurring procedures.

The genius (and the thing to internalize) is that **the involuntary parts of this
loop are infrastructure, not agent behavior.** Recall injection, event journaling,
and episode capture happen *around* the agent via lifecycle hooks — so learning
isn't contingent on the model "remembering to learn." And the loop is the same
whether the turn was typed at a TTY, typed into the web console, or fired by a
schedule at 09:00 with nobody watching.

### How the code is laid out (the macro map)

```
relife/
  cli.py            entry point: do / chat / work / build / serve / doctor / consolidate / dream /
                    memory * / agent * / mcp / crew / crews  (+ __main__.py: `python -m relife`)
  agent.py          build_options() + the streaming SDK loop + to_event() + ask_model_oneshot
  permissions.py    classify() pure policy + grants + TTY callback + UI approval callback
  hooks.py          UserPromptSubmit/PostToolUse/Stop lifecycle hooks (per-client factory)
  agents.py         the agent registry: identity → memory scope; inherit/fork/promote; MCP tokens
  workitems.py      `relife work`: GitHub issue plumbing around the agent (deterministic)
  doctor.py         `relife doctor`: pure run_checks() over injected Probes
  config.py         every path, model id, env knob and tunable in one place
  prompts/          system.md (persona), rem.md
  web/index.html    the self-contained browser console (vanilla JS + EventSource)
  memory/
    cognitive.py    pure ACT-R math: activation, fused_score, forgetting
    store.py        SQLite store (schema v3: spaces), two-stage recall, reinforce-on-write
    vector_index.py pluggable semantic candidate search (brute force / ANN)
    embeddings.py   soft-optional local ONNX embeddings
    spaces.py       space names + MemoryScope(read, write, source)
    service.py      MemoryService facade (memory + skills + workflows + events + spaces/packs)
    client.py       MemoryClient protocol, Local/ScopedMemoryClient, default_client(), off_loop
    remote/         wire.py (dict ⇄ dataclass), daemon.py (FastAPI, also /mcp), http_client.py
    skills.py       single reusable procedures (Markdown files, per space)
    workflows.py    multi-step procedures (Markdown files, per space)
    events.py       tool-event log (own table in the same SQLite db)
    consolidate.py  deterministic "sleep" pass (LLM-free, per space)
    rem.py          opt-in LLM "dream" pass (adversarial critic)
    tools.py        the memory tools, defined once (ToolSpec) for every transport
    context.py      the recalled-context block (recall hook + memory_context tool)
    server.py       in-process SDK MCP server for Claude (built from tools.py)
    mcp_server.py   standalone MCP server for any agent: stdio + streamable HTTP
    _text.py        shared stopword tokenizer
  crew/             `relife crew` ([crewai] extra; Python ≤ 3.13)
    spec.py         pure: CrewSpec + normalize_spec (the plan's validator)
    planner.py      one tool-less Claude call → JSON plan (retry, then fallback)
    agent.py        ReLifeAgent(BaseAgentAdapter): a crew member that is a ReLife turn
    turns.py        run one ReLife turn headlessly; run_sync bridge
    llm.py          ClaudeMaxLLM(BaseLLM): Claude via the CLI for CrewAI agents
    native.py       CrewAI agents on other models, with ReLife memory attached
    memory_tools.py ReLife memory tools as CrewAI BaseTools (same specs)
    build.py        profiles (handoff) + Crew assembly
    runner.py       run_crew(): plan → record → confirm → kickoff → outcomes
    record.py       CrewRunRecord + CrewRunStore (data/crews/<id>/)
    prompts/        planner.md
  build/
    ledger.py       durable plan + progress (resume source of truth)
    server.py       MCP server exposing the ledger to the orchestrator
    agents.py       the `builder` subagent definition
    orchestrator.py run_build(): decompose → delegate → resume
    prompts/        orchestrator.md
  server/
    app.py          create_app() routes + serve(); auth, SSE, sessions, schedules
    session.py      AgentSession (one ClaudeSDKClient), ApprovalBroker, SessionManager
    security.py     pure policy: tokens, CSRF, bind guard, workspace confinement
    schedules.py    Schedule record + cadence math + JSON ScheduleStore
    scheduler.py    the tick loop: fire due schedules, record outcomes
    runs.py         RunRecord + summarize_events() + RunStore
```

Notice the **rhyme**: every layer has a *pure policy/math module* (`cognitive.py`,
`permissions.classify` + `grant_allows`, `security.py`, `schedules.py`,
`runs.summarize_events`, `doctor.run_checks`, `MemoryScope`, `crew/spec.normalize_spec`)
and a *thin imperative shell* around it. That rhyme is the house style — M16 names it
explicitly.

---

<a name="m2"></a>
## M2. The agent runner & the SDK seam

### The control flow

Every invocation flows the same way:

```
cli.py (parse command, resolve workspace)
   → wire 3 pieces: permission callback, MCP servers, memory hooks
      → agent.build_options(...)  assembles ClaudeAgentOptions
         → ClaudeSDKClient(options)  (streaming transport)
            → client.query(prompt); async for msg in client.receive_response(): render
               → _maybe_consolidate()  (the "sleep" beat, after the run)
```

Look at `cli.py:do` (`cli.py:65`): it resolves the workspace, builds the
`can_use_tool` callback bound to that workspace, gets the default MCP servers, gets
the memory hooks, and hands all three to `run_task`. `chat` and `build` do the same
with their own variations — and so does the server's `AgentSession.start()`
(`server/session.py:135`), which is the same three pieces with a *different
permission callback*. **The CLI's whole job is to assemble three pluggable pieces
and start the loop.**

### `build_options()` — the assembly point

`agent.build_options()` (`agent.py:67`) is the single funnel where a run's
configuration is assembled into `ClaudeAgentOptions`. It takes:

- `cwd` — the workspace,
- `permission_mode` + `can_use_tool` — the permission policy (M3),
- `mcp_servers` — browser + memory (+ build ledger for builds),
- `hooks` — the learning-loop hooks (M6),
- `system_prompt` — persona (default ReLife, or the orchestrator's),
- `agents` — subagent definitions (M7),
- `resume` — a session id to continue,
- `max_budget_usd` — an optional spend cap.

`do`/`chat --agent NAME` changes *one* thing: the memory client. `cli._agent_memory`
resolves the registered agent's `MemoryScope` (M13), wraps the client in a
`ScopedMemoryClient`, and passes it to both `default_mcp_servers(memory_client=…)` and
`memory_hooks(client)` — so the agent's tools *and* its involuntary recall/journal see
only that agent's spaces. Permissions, persona and model are unchanged. A crew member
(M15) is assembled the same way, per task.

Two non-obvious but important choices live here:

- `setting_sources=None` (`agent.py:102`) — **deliberately do not inherit the
  surrounding repo's Claude Code settings.** ReLife is self-contained; it defines
  its own behavior and must not be silently reconfigured by whatever `.claude/`
  happens to be in the cwd. (One thing this does *not* switch off: the claude.ai
  connectors, which the CLI attaches from the account, not from settings — M11.)
- `env=config.agent_env()` (`agent.py:99`) — prepend the GitHub CLI dir to PATH for
  the agent subprocess, but *only* when `gh` isn't already resolvable (see
  `config.agent_env`, `config.py:281`). This exists because `gh` was winget-installed
  mid-session and wasn't on PATH for the already-running shell.

### Why this is a *seam* (the injection design)

`build_options` doesn't *construct* permissions, servers, or hooks — it *receives*
them. The same is true of `run_task`/`run_chat`. This is dependency injection, and
it buys three concrete things:

1. **Testability.** `classify()` and the hook callbacks are plain functions tested
   directly, with no live agent. The expensive, non-deterministic SDK loop is the
   thin shell; all the logic lives in injectable, deterministic pieces.
2. **Persona/behavior swapping.** The build orchestrator passes a *different* system
   prompt (`preset_system_prompt(ORCHESTRATOR_PROMPT_FILE)`) and a *different* set
   of subagents into the *same* `build_options`. The server passes a *different
   permission callback* (M9). One assembly funnel, many behaviors.
3. **An out-of-process split.** Because consumers depend on injected interfaces,
   swapping the implementation behind one (memory becoming a daemon — M8) changes
   only the injected object, not the consumers. This was "future" in the first
   edition of this guide; it has since happened, exactly as planned.

### The system prompt: preset + append

`preset_system_prompt()` (`agent.py:51`) returns
`{"type": "preset", "preset": "claude_code", "append": <persona>}`. ReLife keeps the
**`claude_code` preset** (so it inherits Claude Code's strong coding behavior) and
*appends* `prompts/system.md` (persona + safety + memory/skill/workflow
instructions). The orchestrator swaps the append file to change persona without
losing the coding baseline.

### The hard constraint: `can_use_tool` requires streaming

This is the most important non-obvious fact in M2.

The one-shot helper `query(prompt=str)` **cannot** be used when you attach a
permission callback. A permission decision is an *interactive round-trip* mid-run
("can I use this tool?" → "yes/no"), which only the **streaming transport**
(`ClaudeSDKClient` + `receive_response()`) supports.

So `run_task` (`agent.py:335`) uses `ClaudeSDKClient` explicitly, and its docstring
says exactly why. This even applies to the deny-everything case:
`ask_model_oneshot` (`agent.py:111`) — used by the REM pass for a pure text-in/
text-out judgment — attaches `_deny_all_tools` (`agent.py:106`) as its
`can_use_tool`, and *because* it attaches a callback at all, it **must** run in
streaming mode too. "No tools" is still enforced *through* the permission seam,
which forces streaming.

### `to_event()` — one taxonomy, two renderers

`to_event(msg)` (`agent.py:143`) is a **pure** function mapping one streamed SDK
message to zero-or-more JSON-serializable events: `text`, `thinking`, `tool_use`,
`tool_result`, `result`. It is the *single source* of the streaming taxonomy. The
terminal `_render` (`agent.py:183`) consumes it, and so does the server's SSE stream
(`server/session.py:264`) — so what the browser sees and what the terminal prints
can't drift.

The trap it encodes (`agent.py:151`): **tool results arrive on a `UserMessage`, not
the assistant one.** The CLI reports each result as a `user`-type frame carrying
`ToolResultBlock`s. Walking only `AssistantMessage` — the obvious thing — made the
UI stream every tool *call* and never a single *result*. Text on a `UserMessage` is
the caller's own turn echoed back and is dropped (the server publishes that itself
on submit — which M10's run recorder relies on).

### `maybe_consolidate` — one decision, two callers

`maybe_consolidate()` (`agent.py:210`) is the "sleep beat": it asks the memory
client to *decide and run* the throttle in one call (`client.maybe_consolidate()`).
The CLI flavour `_maybe_consolidate()` (`agent.py:259`) runs it inline and prints a
note; the server flavour `maybe_consolidate_off_loop()` (`agent.py:245`) runs it on
a worker thread because inline it froze every session's stream (M9). A
process-wide **non-blocking** `threading.Lock` (`agent.py:207`) means two sessions
finishing together don't sweep the same store twice — the loser just gets `None`
and the throttle fires again next turn. Why the decision lives on the memory side
rather than here is an M8 lesson.

### `ask_model_oneshot` — the SDK-touching escape hatch

`ask_model_oneshot(system_prompt, prompt)` (`agent.py:111`) runs the model with **no
tools, no MCP, no hooks** and returns `(text, cost_usd)`. It exists for one caller:
the REM "dream" pass, which needs the model as a *pure advisor* with zero ability to
take action. Tools are hard-denied, so the call can never touch the filesystem or do
anything but emit text. The cost is read off the `ResultMessage` so REM can report
spend.

> M2 mastery check: cli wires 3 pieces → `build_options` assembles them → streaming
> client runs the loop; the injection seam buys testability + persona swap + the
> service split; `can_use_tool` forces streaming even in the deny-all one-shot;
> `to_event` is the one taxonomy both renderers share, and tool results ride the
> `UserMessage`.

---

<a name="m3"></a>
## M3. The permission model

### What it's for

ReLife runs **autonomously** for a large class of actions but must **never** take an
irreversible outward action without approval. `permissions.py` is the gate that
decides, per tool call, **allow** (run it now) or **ask** (get human approval).

### The autonomy model (v1)

- **Auto-allow:** reading, browsing, editing files *inside the workspace*, building
  and testing code, and **git including `git push`** (the user explicitly authorized
  git), ReLife's own MCP tools, and **connector *reads*** (search mail, list events).
- **Always-ask:** anything *outward-facing* — sending email, posting data off the
  machine, publishing packages, remote shells, any `gh` call that isn't a known read
  (verb-based — below), global package installs, writing *outside* the workspace (by
  any door — see below), **connector *writes*** (send, create, delete…), and **any
  unrecognized tool** (fail-closed).
- **Pre-approved, narrowly:** a schedule's *grants* turn a few specific ask-cases
  into allows for one unattended turn (below, and M10/M12). They are the only
  sanctioned widening of the policy.

### `classify()` is a pure function

`classify(tool_name, tool_input, workspace)` (`permissions.py:584`) returns
`("allow" | "ask", reason)`. It has **no I/O and no side effects** — given the same
inputs it always returns the same decision. That's what makes the security policy
unit-testable in isolation, without ever spinning up an agent. Two wrappers turn the
pure decision into the SDK's async `can_use_tool`:

- `make_permission_callback()` (`permissions.py:896`) — the **TTY** path: prompts
  y/n for ask cases.
- `make_approval_callback()` (`permissions.py:945`) — the **UI** path: pushes ask
  cases to a broker that shows a card in the browser and awaits the click (M9).

Both call `classify()` verbatim. **There is exactly one policy source**; the
wrappers differ only in *who gets asked*.

The classification order (`permissions.py:589`):

1. `_ALWAYS_ALLOW_TOOLS` (read-only / planning / shell control) → **allow**.
2. `_FILE_WRITE_TOOLS` → allow **only if** the target path is inside the workspace,
   else **ask**.
3. `_SHELL_TOOLS` (Bash/PowerShell) → first, any **nested shell** (`bash -c "…"`,
   `cmd /c`, `pwsh -Command`, `Start-Process -ArgumentList`) is re-classified on its
   inner command line (`_inner_commands`, `permissions.py:472`); then **ask** if the
   command matches the outward/destructive regex, installs packages globally
   (`_global_install`, `permissions.py:177`), runs a `gh` verb that isn't a known read
   (`_gh_outward`, `permissions.py:260`), *or* writes/deletes outside the workspace;
   else **allow**.
4. Tools starting with a trusted MCP prefix (`mcp__relife`, `mcp__browser`) → allow.
5. Connector tools (`mcp__claude_ai_*`) → verb-based: write-verb → **ask**, read-verb
   → allow, neither → **ask**.
6. **Everything else → ask** (the fail-closed default, `permissions.py:640`). This
   branch does real work: the CrewAI tools handed to a ReLife crew member are served
   as an MCP server named `crew_tools` precisely so they land *here* (M15).

### The deepest idea: denylist for shells, allowlist for tools

This is the architectural insight worth grilling yourself on.

For **shell commands**, the policy uses a **denylist** (`_OUTWARD_SHELL`,
`permissions.py:84`): allow by default, ask only for a *finite, enumerable set of
dangerous patterns* — email senders, uploading
curl/wget and non-GET `Invoke-RestMethod`, a download piped into an interpreter,
scp/sftp/rsync/ssh and `Enter-PSSession`/`Invoke-Command -ComputerName`, package
publish, `sudo`/`-Verb RunAs`/`Set-ExecutionPolicy`, and unrecoverable device ops
(`rm -rf /`, `mkfs`, `dd of=/dev/…`, `Format-Volume`) — **in both shells**, POSIX
and PowerShell alike. A Bash-only pattern set was a hole, not a simplification: on
Windows, PowerShell is the shell the agent actually reaches for, and
`Send-MailMessage` was auto-allowed until the MVP pass-1 fix.

Why denylist here and not an allowlist? Because **the set of safe shell commands is
effectively infinite** (every build invocation, every test runner, every git
subcommand, every file utility...). You cannot enumerate "all safe commands." But
the set of genuinely *dangerous, outward* shell patterns is **small and
enumerable.** When one side of a partition is infinite and the other is finite, you
must define the policy in terms of the finite side. So: enumerate the dangerous,
allow the rest.

For **tools in general** (the unknown-tool case), the policy is the opposite — an
**allowlist** with a fail-closed default: an unrecognized tool is *asked*, not
allowed. Why is fail-closed cheap *here* but would be intolerable for shells?
Because **unrecognized tools are rare** — they appear only when a new tool is wired
in, which is a development-time event. Prompting on something that almost never
happens costs almost nothing. Shell commands, by contrast, are *constant* —
fail-closed there would mean a prompt on every build and test, which would make
autonomy worthless. **The policy shape on each axis is chosen by which side is
finite and how often the "ask" path fires.**

A denylist is only as good as its adversary, so it has one: `tests/test_permissions_corpus.py`
holds a `SHOULD_ASK` corpus of bypass attempts (the release-hardening pass found and
closed 56 — nested shells, PowerShell aliases, env-var paths, persistence keys) and a
`SHOULD_ALLOW` corpus of everyday build/test/git commands, so tightening the gate can't
quietly make ordinary work prompt. Any change to `permissions.py` extends both.

### The workspace boundary binds the shell too (the redirect hole)

The first edition of the policy contained the workspace only on `Write`/`Edit`. That
left a one-character escape: `echo x > ~/.bashrc` is a *shell* command, matches no
outward pattern, and was auto-allowed. So `classify()` now asks the shell three
questions, not one (`permissions.py:606–622`):

1. Does it match `_OUTWARD_SHELL`? → ask.
2. `_write_targets(command)` (`permissions.py:490`) — what does it **write to**?
   Redirects (`> f`, `>> f`, `2> f` — fd duplication like `2>&1` can't match by
   construction, `permissions.py:319`), writer verbs (`tee`, `Set-Content`,
   `Out-File`, `New-Item`…), copy/move destinations (last positional of `cp`/`mv`/
   `Copy-Item`), and named destination params (`-OutFile`, `-Destination`, `-Path`).
   `/dev/null`, `nul`, `$null` are recognized sinks, not files.
3. `_delete_targets(command)` (`permissions.py:519`) — what does it **delete**?
   `rm`/`del`/`Remove-Item`/`Clear-Content`… positionals and `-Path`.

`_escapes()` (`permissions.py:542`) asks if any target isn't provably inside the
workspace. The **asymmetry** is deliberate: for *deletes* `strict=True`, so a target
we can't evaluate (`rm -rf "$DIR"`) counts as outside — an unevaluated recursive
delete is exactly the case worth a prompt. For *writes* `strict=False`, so an
ordinary `… > $LOG` inside the workspace doesn't start prompting; only an
unresolvable target that also *names a path* (`$HOME/.ssh/x`) is rejected. The
extraction is explicitly **heuristic** — shell grammar is not a regex — and it is
tuned to err toward asking rather than to be complete.

### Path containment: defeating traversal, symlink and `~` escapes

File writes are allowed only `_under()` the workspace (`permissions.py:561`). The
check is not a string-prefix test (which `../` and symlinks would defeat). It:

1. **expands `~` first** (`permissions.py:574`) — otherwise `rm -rf ~/Documents`
   looks like a *relative* path inside the workspace,
2. resolves the path against the workspace if relative,
3. calls `.resolve()` on both the target and the workspace — which **collapses `..`
   segments and follows symlinks to their real location**,
4. and only then checks `target == workspace or workspace in target.parents`.

So `workspace/../../etc/passwd` resolves to `/etc/passwd`, which is not under the
resolved workspace → **ask**. A symlink inside the workspace pointing to `/etc`
resolves to `/etc` → **ask**. Resolving *before* comparing is what makes the
containment real instead of cosmetic. (M9's `resolve_workspace` applies the identical
discipline to the *server's* workspace root — same idea, one level up.)

**Two Windows lessons from the live smokes.** In Git Bash, `/d` is *drive D:* — but
`_FLAG` used to read any single-letter `/x` token as a cmd switch (`del /s /q`) for
every verb, so `rm -rf /d` was **auto-allowed**. Slash switches are now recognised
only for cmd built-ins (`_SLASH_SWITCH_VERBS`, `permissions.py:423`). The mirror-image
false ask: Git Bash spells `D:\relife\workspace` as `/d/relife/workspace`, so writes
inside the workspace in that spelling asked; `_msys_to_windows` (`permissions.py:534`)
now translates them, for the `Bash` tool on Windows only (PowerShell never uses that
form).

### GitHub CLI: verb-based and fail-closed (MVP pass 10)

`gh` used to be gated by *group* (`gh pr|issue|release|api|gist` → ask), which was
wrong in both directions: reading the user's own work items (`gh issue list
--assignee @me`, `gh pr view`) prompted — so a scheduled "triage my issues" run was
denied before it could look — while `gh repo delete`, `gh repo edit --visibility
public`, `gh secret set`, `gh workflow run` and `gh auth login` ran **unasked**.

`_gh_outward(command)` (`permissions.py:260`) now finds **every** `gh` occurrence in
the command (`_GH_CALL`, `:223` — inside `bash -c "…"`, after `time`/`xargs`, a full
path to `gh.exe`) and requires each to be a known read (`issue|pr list/view/status`,
`pr diff|checks|checkout`, `search`, `status`, `run list/view`, …) or `gh repo
create|clone` — the v1 build → create → push flow, kept autonomous like `git push`.
Anything else asks. `gh api` is allowed only as a REST **GET** (`_gh_api_reads`,
`:228`: no method and no `-f/-F/--input` field flags, or an explicit `-X GET`), and
`gh api graphql` always asks, since a query document can mutate. This is the
connector argument again: enumerate the *reads*, which are stable, and let every verb
nobody thought of fall to "ask."

### Grants: the one sanctioned widening (MVP passes 9 & 14)

An unattended run has nobody at the approval card, so "summarize my inbox and email
me the digest" could never finish. A schedule may therefore carry **grants** —
narrow, user-set pre-approvals — and they are the only place the policy is widened
on purpose, so they are pure and fail-closed (`permissions.py:667–895`):

- **Kinds:** `email` (Gmail send/reply/forward/draft) and `calendar` (create-event),
  each bound to ≤ `GRANT_MAX_ADDRESSES` (5) listed addresses; and `pull_request`
  (M12). `normalize_grants` (`:711`) validates a grant in full or drops it — a
  hand-edited grant is never *widened* by a lenient parse.
- **`grant_allows(grants, tool, input)`** (`:858`) is consulted **only after
  `classify()` said ask**, and covers only connector calls (plus the one bound
  `gh pr create` shape). Destructive operations (`_GRANT_NEVER`: delete, modify,
  share…) are excluded even when the operation also matches the grant's shape.
- **Every address must be listed.** `_addresses_in` walks the call's fields, skipping
  free-text content (`body`, `subject`, `htmlBody`, … — `_is_content_key` normalizes
  camelCase via `_snake`), and a recipient field holding something that isn't an
  address (a group alias, `ops-team`) fails the grant.
- **Email needs a recipient we can see.** A reply that infers its recipient from the
  thread, raw MIME, `send_message(draftId=…)` (sends the stored draft and ignores
  `to`) and `reply(replyAll=true)` (keeps the thread's CC) all fall back to asking —
  the last two were real bypasses found by reading the **live** connector schemas
  after 1.0, when `to=[me]` passed the address check.

The session side (one turn only, a per-run use cap, every use journaled) is M10; the
`pull_request` binding is M12.

### Connectors: a verb-based policy for tools whose names you don't control

Gmail / Google Calendar / Google Drive arrive as `mcp__claude_ai_<Service>__<tool>`
(M11 explains how). Their exact tool names are Google's, they can change, and they
are the **first genuinely outward capability** the agent has. So the policy
(`permissions.py:58–71`, applied at `:628`) keys on the **verb in the tool name**,
not on an enumerated list:

- `_CONNECTOR_WRITE` (`send|create|delete|modify|reply|forward|draft|label|…`) → ask.
- `_CONNECTOR_READ` (`search|list|get|read|fetch|find|…` plus `authenticate`) → allow.
- Neither → ask. **Both → the write check wins** (it's tested first).

The `authenticate`/`complete_authentication` allowance matters: linking a Google
account is a read-only handshake that opens a browser, and asking for it would just
add friction to something that can't send anything.

### The approval prompt shows *what*, not just *that*

An "approve?" line with a blank tool name is worthless for an email. So both
wrappers render `agent._tool_brief(tool_input, limit=400)` (`agent.py:266`): the
obvious key for built-ins (`command`, `file_path`, `url`…), and for a connector call
— which has none of those — the **leading scalar fields** (`to=… subject=…`). The user
approves a concrete recipient and subject, not a mystery.

### Non-interactive runs deny, never block

`make_permission_callback` defaults `interactive` to whether stdin is a TTY. In a
non-interactive run, ask-cases are **denied**, not blocked (`permissions.py:927`).
This means an unattended run *never hangs waiting for input* and *never takes an
unapproved outward action* — it just declines and moves on. (A ReLife crew member,
M15, runs with this TTY callback too: approvals surface in the terminal that started
the crew.) There's even a defense
for a pseudo-TTY where `isatty()` lies: if reading the prompt raises, it **fails
closed → deny** (`permissions.py:933`). The UI path has the same shape with a timer:
no click within `AGENT_APPROVAL_TIMEOUT` → deny (M9). The scheduler inherits it for
free: nobody watching ⇒ timeout ⇒ deny (M10). **One rule, three surfaces.**

### Two shell tools, gated identically

On Windows the agent has both `Bash` and `PowerShell`. `_SHELL_TOOLS` contains both
(`permissions.py:40`), so the same outward/destructive gating applies to whichever
shell the model picks. *Any new shell tool must be added there* or it would fall
through to the fail-closed unknown-tool branch.

> M3 mastery check: pure `classify()` with two thin wrappers (TTY, UI); shells use
> a denylist because the safe set is infinite and the dangerous set finite; the
> unknown-tool fail-closed default is cheap because it fires rarely; the workspace
> boundary binds the shell's redirects/copies/deletes too, strict for deletes;
> `_under()` expands `~` and resolves before comparing; connectors *and* `gh` are
> gated by verb, reads enumerated, everything else asks; grants are the one widening
> — consulted only after "ask", connector-only (plus one bound PR shape), every
> address listed, destructive ops never; "no answer ⇒ deny" is the same rule at TTY,
> UI and schedule.

---

<a name="m4"></a>
## M4. The cognitive core (`cognitive.py`)

### What it is

`cognitive.py` is the **deterministic, pure-math heart** of ReLife's brain-like
memory. **No I/O, no LLM, no DB** — just functions over a memory's stats. The same
math is reused by **recall** (ranking) and by **consolidation** (forgetting), which
is why it's isolated: one source of truth for "how strong is this memory?"

### Activation (ACT-R inspired)

A memory's base-level **activation** rises with how *often* and how *recently* it's
used, and decays as it sits idle (`activation()`, `cognitive.py:37`):

```
activation = ln(1 + use_count)
             − DECAY · ln(1 + age_days(last_used))
             + IMPORTANCE_BOOST · importance
```

Three forces, each chosen deliberately:

- **Frequency: `ln(1 + use_count)`.** The logarithm is the key. It means the *first*
  few uses matter a lot and additional uses matter progressively less. Why? To stop a
  memory that's been used 500 times from *steamrolling* a freshly-relevant exact
  match. Linear frequency would let a popular-but-off-topic memory dominate forever;
  log frequency keeps the popular ones strong without making them unbeatable.
- **Recency decay: `− DECAY · ln(1 + age_days)`.** Also logarithmic, so something
  goes "stale" quickly at first then plateaus — recent things drop off fast, old
  things age slowly. `DECAY` (default 0.35) tunes the forgetting rate.
- **Importance lift: `+ IMPORTANCE_BOOST · importance`.** A steady additive lift so
  explicitly-salient memories resist decay.

### The four-signal fused score

Recall doesn't rank on activation alone. `fused_score()` (`cognitive.py:59`) combines
**four** signals into one number:

```
score = W_SEM · semantic        (0.45)  — local embedding cosine, [0,1]
      + W_KW  · keyword          (0.30)  — token overlap fraction, [0,1]
      + W_ACT · sigmoid(act)     (0.15)  — cognitive activation, squashed
      + W_IMP · importance       (0.10)  — explicit salience, [0,1]
      + KIND_RECALL_BOOST[kind]          — a small per-kind prior
```

Two subtle decisions:

- **`sigmoid(act)` (`cognitive.py:22`).** Activation is *unbounded* (it's a sum of
  logs and a lift; it can be any real number). The other three signals live in
  `[0,1]`. To fuse them fairly with fixed weights, activation must be squashed onto
  `(0,1)` first — that's what the sigmoid does (with overflow guards at ±60). Without
  it, a single huge activation could swamp the weighted sum and make the weights
  meaningless.
- **Importance double-counts on purpose.** Importance influences recall through *two*
  paths: it lifts `activation()` (slowing forgetting) **and** it's a standalone term
  via `W_IMP`. This is intentional and documented at `config.py:86`: importance
  should *both* slow forgetting *and* act as a direct relevance signal.

The per-kind prior (`KIND_RECALL_BOOST`, `config.py:98`) gives durable kinds
(`preference` +0.05, `pattern` +0.02) a gentle edge at equal evidence — mirroring how
stable knowledge stays more accessible than one-off episodes.

### Two-tier forgetting

Forgetting happens in **two stages**, both applied only by the consolidation sweep
(never by recall):

1. **Soft archive — `should_archive()` (`cognitive.py:85`).** A memory is archived
   only when **all** hold: it's idle past `MIN_FORGET_AGE_DAYS` (14), its activation
   has fallen below `FORGET_THRESHOLD` (0.20), and it is **not pinned**. `preference`
   memories and anything with `importance >= PIN_THRESHOLD` (0.80) are *never*
   archived — like core facts a person keeps regardless of use.
2. **Hard delete — `should_hard_delete()` (`cognitive.py:114`).** A second, *slower*
   tier: a memory that was already archived and then left untouched past
   `HARD_DELETE_AGE_DAYS` (90) is permanently removed, so the store doesn't grow
   without bound. Preferences and pinned items are exempt (defense in depth — they're
   never archived in the first place).

### Why two tiers instead of one delete

This is the idea to internalize. A single "delete when faded" rule would be
**irreversible** and would destroy **cyclical / seasonal** memories — something you
use heavily every December, ignore for 11 months, and would delete in March under a
one-shot rule. The two-tier design makes the first forgetting step **reversible**
(archived rows still exist and can be reactivated by being saved/recalled again) and
only deletes after a *much* longer idle period that even seasonal memories wouldn't
cross. Forgetting becomes recoverable; deletion is the rare last resort. The same
tier is what `relife memory forget` and REM's `prune` use (M6, M11) — **every
"forget" in the system is the reversible one**.

> M4 mastery check: log-frequency stops an over-used memory steamrolling exact
> matches; the sigmoid normalizes unbounded activation onto [0,1] for fair fusion;
> two-tier archive→delete preserves cyclical memories reversibly; importance
> double-counts on purpose.

---

<a name="m5"></a>
## M5. The memory store (`store.py`) + the seams around it

### What it is

`store.py` is the persistence + retrieval layer: SQLite-backed facts, preferences,
episodes, and patterns whose relevance behaves like M4 describes. The store is an
**injectable class** `MemoryStore(db_path)` (`store.py:119`); module-level
`save`/`recall`/… are back-compat shims over a lazily-built default instance bound to
`_DB_PATH` (`store.py:40`). `_DB_PATH` stays a reassignable global so tests can point
the default at an isolated database — and the default is rebuilt whenever it changes
(`_store()`, `store.py:696`). That reassignable global is also, as M8 shows, exactly
how the daemon binds everything to one DB.

### Schema and migrations

One table, `memories` (`_create_schema`, `store.py:174`), with the cognitive columns:
`importance`, `last_used_at`, `use_count`, `status`, `embedding` (a BLOB of packed
floats) — and, since schema **v3**, `space` (which agent's memory this is; default
`'default'`) and `source` (provenance: who wrote it, or `import:<space>`), indexed
`memories_space(space, status)`. Schema version is tracked via **`PRAGMA
user_version`** (`store.py:159`) with ordered migrations:

- `version == 0` covers three cases at once (fresh DB, a v1 store, or a v2 store that
  never had the pragma set): create the table if absent, idempotently add the v2
  columns (`_migrate_to_v2`, `store.py:193`), backfill `last_used_at` from
  `created_at`, and stamp **v2** — not the current version.
- `_apply_migrations()` (`store.py:215`) then runs every later step in order (v3 adds
  `space`/`source` and the index), each idempotent behind a column-exists check.

That "stamp v2, then migrate" detail is a bug that was caught, not a style choice:
the `version == 0` branch used to stamp `SCHEMA_VERSION` directly. That was harmless
while there was nothing after v2; the moment v3 existed, every **fresh** database would
have been stamped v3 *without* the v3 columns, so any query touching `space` would fail
(caught while building v3, before it shipped). Every
pre-spaces store upgrades in place and reads as space `default`.

### Two-stage recall (the core algorithm)

This is the single most important mechanism in the memory layer. `recall()`
(`store.py:495`) does **not** score the whole table. It works in two stages:

**Stage 1 — candidate generation (`_candidates`, `store.py:399`).** Pull a *bounded*
set of candidates (capped at `CANDIDATE_TOPN` = 50) using indexes:

- **Keyword candidates** via the **FTS5** full-text index, ordered by `bm25`
  relevance (`store.py:421`). Scales to large stores because it's an index lookup,
  not a scan.
- **Semantic candidates** via the pluggable **vector index**, but only rows whose
  cosine clears `SEM_CANDIDATE_THRESHOLD` (0.65) and aren't already in the keyword
  set (`store.py:436–445`).
- If FTS5 is unavailable, a **fallback keyword scan** over the whole table —
  correct, just slower; fine for small stores.
- Every path takes `spaces`: the FTS query, the fallback scan, the document-frequency
  count and the vector search are all filtered to the caller's spaces in SQL
  (`_space_filter`). At the store `spaces=None` means *unfiltered* — mechanism for
  consolidation and admin listings; the *policy* (an agent-facing read with no spaces
  means `default`) lives one layer up, in the service (M13).

**Stage 2 — fuse-rank only the candidates.** For each candidate compute the full
`fused_score`, drop anything under `RECALL_FLOOR` (0.12, `store.py:557`), sort, and
return the top `k`.

### The property that makes recall safe: Stage 1 *is* the relevance gate

Internalize this, because it's the linchpin connecting M5 to M6.

A candidate enters Stage 1 **only** if it has keyword overlap **or** strong semantic
similarity to the query (the explicit gate at `store.py:535`). Activation and
importance are **not consulted in Stage 1 at all.** They only act in Stage 2, where
they tie-break *among rows that already passed the relevance gate.*

The consequence: **a strong-but-irrelevant memory can never surface.** No matter how
high its activation or importance, if it doesn't match the query topically it never
enters the candidate pool, so it's never ranked, so it's never returned. This is the
"unrelated query → nothing" guarantee.

**"Matches topically" has to mean more than one shared word.** The release-hardening
pass found the gate leaking: a memory sharing a single *generic* word with the query
("write", "test" — which also name tools inside episode text) cleared Stage 1, and
activation + importance alone cleared `RECALL_FLOOR`, so noise surfaced, got
reinforced, and fed itself. Now a memory overlapping the query on **one** term
surfaces only if that term is *distinctive* — present in at most
`RECALL_COMMON_TERM_FRACTION` (20%) of active memories (with a `RECALL_COMMON_MIN_DOCS`
floor so tiny stores aren't starved). The document frequency is counted within the
caller's spaces. And — crucial for M6 — it's *why
reinforcement-on-recall is safe*: reinforcement can only ever strengthen memories
that were *relevant enough to surface*, so the rich-get-richer loop is structurally
starved of irrelevant fuel.

### Reinforce-on-write (and on recall — but not on *looking*)

`save()` (`store.py:276`) does **not** blindly insert. If the exact text already
exists *in the same space*, it **reinforces** the existing row (refreshes recency, bumps `use_count`,
keeps the higher importance, reactivates it) and returns its id. With embeddings on,
it also catches **near-duplicate paraphrases** via `_semantic_duplicate` above
`SAVE_DEDUP_SIM` (0.93) and reinforces those instead of cloning (`store.py:328`).
Saving the same knowledge twice makes it *stronger*, not *duplicated* — exactly like
recalling it. `recall(..., reinforce=True)` applies the same logic: surfacing is a
use.

The flag matters. The *hook* recalls with `reinforce=True` (a use by the agent). The
**inspection CLI** (`relife memory search`, M11) recalls with `reinforce=False`
(`cli.py:389`) — so a human *looking* at what the agent would be shown does not
distort what the agent will be shown. **Observation must not be a use.**

### Soft-optional everything: FTS5, embeddings, ANN

Three capabilities are **soft-optional** — present them if available, degrade
gracefully if not, never a hard dependency:

- **FTS5** (`_init_fts`, `store.py:237`) is *feature-detected* by trying to create
  the virtual table; if the SQLite build lacks it, `_fts_ok` stays False and recall
  falls back to a keyword scan. *Gotcha:* for an external-content FTS5 table,
  `COUNT(*)` reads the **content table**, not the index — so it can't tell you the
  index is empty. The rebuild decision keys off whether the table is being created
  for the first time (a `sqlite_master` check, `store.py:241`), not a count.
- **Embeddings** (`embeddings.py`) use **`fastembed`** (ONNX, CPU, offline, *no API
  key* — because Max, not API). `available()`/`embed()`/`cosine()` all degrade to
  `None`/keyword+activation if the package is absent or disabled. **Tests force
  embeddings OFF** via an autouse fixture so the suite stays deterministic, except
  tests marked `@pytest.mark.semantic`. The lazy model build is double-checked-locked
  so it's constructed at most once.
- **ANN index** (`vector_index.py`) — see next.

### The vector index seam (`vector_index.py`)

Semantic candidate search hides behind a `VectorIndex` protocol (`vector_index.py:39`)
with two backends:

- **`BruteForceIndex`** — an exhaustive cosine scan over the `embedding` column
  (`vector_index.py:60`). Always correct, no extra storage, the default and the
  source of truth.
- **`SqliteVecIndex`** — an ANN `vec0` virtual table via the optional `sqlite-vec`
  extension (`vector_index.py:89`). For large stores.

The defining safety mechanism is `get_index()` (`vector_index.py:201`): it returns
the ANN backend **only after a runtime self-test (`_self_test`, `vector_index.py:173`)
proves a correct round-trip on *this machine*.** This catches the case that "import
succeeded" doesn't — an extension that **loads but misbehaves** (wrong build, broken
distance metric). If the self-test fails, it silently falls back to brute force. So a
broken extension can *never* corrupt recall; the worst case is "slower, still
correct." The `embedding` BLOB column stays the source of truth either way — the ANN
table is a synced accelerator used only to *prune* the candidate set, never to *rank*
(ranking re-computes exact cosine from the column), which makes it robust to
distance-metric quirks.

### The service / client seams (`service.py`, `client.py`)

Two layers of indirection sit between consumers and the store:

- **`MemoryService`** (`service.py:78`) — the in-process *facade* for long-term
  memory (`save`/`recall`/`forget`/`archive(id)`/`get(id)`/`consolidate`/
  `maybe_consolidate`/`dream`/stats), **procedural memory** (`skill_*`,
  `workflow_*`), **and the tool-event log** (`log_event`/`events_for_task`/
  `event_count`). It resolves the default store on each call so it honors
  `_DB_PATH` reassignment; the skill/workflow/event methods delegate to the module
  functions *at call time* so they honor `_SKILLS_DIR`/`_WORKFLOWS_DIR`/
  `events._DB_PATH` reassignment and the daemon's binding. The whole consumer-facing
  surface routes through this seam; only `consolidate`/`dream` stay module-level
  (they *mine* the defaults, so the daemon binds them — M8).
- **`MemoryClient`** (`client.py:56`) — the *consumer-facing* protocol. The MCP tools,
  all three hooks, the CLI and the server all go through `default_client()`
  (`client.py:310`), which picks the transport **by environment**: `RELIFE_MEMORY_URL`
  set → `HttpMemoryClient`; unset → `LocalMemoryClient` (`client.py:88`).
- Every method takes keyword-only `space`/`spaces`/`source`, and the service added
  space administration (`spaces`, `copy_space`, `archive_space`, `export_space`,
  `import_pack`). Agents never see those parameters: they get a
  **`ScopedMemoryClient`** (`client.py:175`) wrapped around whichever transport is in
  play — the subject of M13.

In the first edition these were "no-op indirections paid now so the migration is a
no-op later." The migration has now happened (M8), and the claim held: **no consumer
was touched.** The seam is the reason.

### Procedural memory: skills and workflows

Beyond facts, ReLife stores **procedures** as human-readable Markdown files (so
they're diffable and the consolidation pass can write them mechanically):

- **Skills** (`skills.py`) — a *single* reusable procedure ("how I scaffold a Python
  CLI"). Frontmatter `name` + `when_to_use`, then steps. Recall is keyword overlap
  over name+when+body with the **name weighted 2×** (`skills.py:115`).
- **Workflows** (`workflows.py`) — a *multi-step* ordered chain ("scaffold → test →
  repo → push"), same file format plus a `trigger` field. Same weighted keyword
  recall.
- **Per space.** The `default` space keeps the historical `data/skills/` and
  `data/workflows/` (so no file moved when spaces arrived); any other space lives
  under `data/spaces/<space>/{skills,workflows}` (`spaces.space_dir`). A find over
  several spaces lets an earlier space shadow a same-named procedure in a later one,
  so an agent's own version of a skill wins over the one it inherited.

### The MCP surface (`server.py`)

`memory_server(client=None)` (`memory/server.py:67`) exposes the tools the agent calls
directly: `memory_save` (takes `importance`), `memory_recall`, `memory_forget`,
`skill_write`/`skill_find`, `workflow_save`/`workflow_find`, `memory_consolidate`, and
`memory_dream`. Surfaced under the `relife_memory` server → `mcp__relife_memory__*`,
which the trusted `mcp__relife` prefix auto-allows (no permission change). Memory is
shipped *as an MCP server even though it's in-process* so the agent-facing contract is
identical whether the store is local or a daemon.

The tools themselves are no longer written here: each is a transport-neutral
`ToolSpec` in `memory/tools.py` (name, description, JSON schema, `handler(client,
args)`), and `server.py` only wraps `INTERNAL_TOOLS` as SDK tools — names and schemas
unchanged for Claude. The same specs are served to *other* agents by
`memory/mcp_server.py` and to CrewAI agents as `BaseTool`s (M14, M15). With no
client, the tools resolve the module's `default_client` at call time; with one, they
are bound to that (scoped) client.

> M5 mastery check: two-stage recall = indexed candidates → fuse-rank only those;
> Stage 1 IS the relevance gate, so importance/activation only tie-break among
> already-relevant rows and can never float an irrelevant-but-strong memory into the
> prompt; one shared generic word isn't relevance; recall is a use but *inspection*
> is not; the vector-index self-test catches "loads but misbehaves"; the service/client
> seams are why the daemon split touched no consumer; schema v3 adds `space`/`source`,
> and a fresh DB stamps v2 then migrates.

---

<a name="m6"></a>
## M6. Hooks & the learning loop

This is where "learns over time" stops being a slogan and becomes wiring. M4/M5 gave
the *memory organ*; M6 is the **nervous system that feeds it automatically**, plus the
two offline maintenance passes.

### 6a — The three hooks (`hooks.py`)

**The core problem.** If recall and reflection depend on the *agent deciding* to call
`memory_recall`/`memory_save`, they won't happen reliably — an LLM under task pressure
forgets to check and to take notes. ReLife's move: **make the loop structural, not
behavioral.** The SDK fires lifecycle hooks at fixed moments and ReLife hangs the
loop's involuntary parts on them. The agent benefits from memory *whether or not it
ever thinks about memory.*

Three hooks, registered in `memory_hooks()` (`hooks.py:163`):

| Event | Function | Loop beat | Job |
|---|---|---|---|
| `UserPromptSubmit` | `_recall_hook` | **recall** | inject relevant memory+skills+workflows before the prompt |
| `PostToolUse` | `_event_hook` | (feeds consolidate) | journal every tool call |
| `Stop` | `_episode_hook` | **reflect** (involuntary) | capture a deterministic episode of the run |

The symmetry: `_recall_hook` injects at the **start**, `_episode_hook` captures at the
**end**, `_event_hook` records the **middle**. The hooks bracket the whole turn.

All three are built by one factory, `_make_hooks(get_client)` (`hooks.py:93`). The
module-level trio is bound to `lambda: default_client()` — looked up *at call time*,
so under a daemon (M8) every one of them reaches the daemon, and tests can monkeypatch
`hooks.default_client`. `memory_hooks(client)` builds a fresh trio over one agent's
`ScopedMemoryClient` instead (M13), which is how an agent's journal and episodes land
in its own space.

**And none of them touch the client on the event loop** (MVP pass 11). The hooks are
`async` callers of a *sync* client; under `relife serve` they share one loop with
every session, so each inline recall (FTS + ONNX with embeddings), each journaled tool
call (a loopback POST in daemon mode) froze every other session's stream and pending
approval. Every such call now goes through `memory.client.off_loop(fn, …)`
(`client.py:40`, `anyio.to_thread`). Thread safety is the pass-4 argument: a fresh
SQLite connection per call, a locked model load, a thread-safe `httpx.Client`. This
delivered the practical goal of the once-planned async `MemoryClient` variants without
a second client surface.

#### Hook 1 — `_recall_hook` (UserPromptSubmit)

Before the prompt reaches the model (`hooks.py:96`), it:

1. **Stashes the prompt *and a turn watermark*** in `_last_prompt[session_id]`
   (`hooks.py:103`) — the prompt so the *Stop* hook can later pair intent with
   approach, and the session's latest event id (`_turn_start_event_id`,
   `hooks.py:43`) so that pairing covers **this turn only**. The event log is keyed
   by session, and a chat / server session runs many turns; without the watermark
   every later episode replayed the whole session's tools.
2. **Gathers candidates** from three sources (`memory/context.py:52`): top-5 memories
   (`recall(..., reinforce=True)`), top-2 skills, top-1 workflow. Each carries
   `(section_label, key_text, rendered)`. This block is built by
   `memory/context.build_context(client, query)` — shared with the `memory_context`
   MCP tool, so an agent *without* a hook (M14) asks for exactly what a ReLife agent is
   handed. Anything from a space other than the caller's own is labelled with its
   provenance — `[fact · via veteran]` — so inherited memory reads as background from
   another agent, not as the caller's own conclusions.
3. **De-duplicates across sources and caps size**: `_is_dup` (`memory/context.py:28`) rejects a
   block whose token Jaccard against anything already kept is ≥ `RECALL_DEDUP_JACCARD`
   (0.8), and a running byte count is held under `RECALL_INJECT_BUDGET` (2400). It's a
   **cross-section greedy knapsack** with priority order memory → skill → workflow.
4. **Returns `additionalContext`** (`hooks.py:64`) — the SDK splices it into the
   model's context for *this* prompt only. Nothing survived → return `{}`.

The whole recall body runs under `try/except → {}` (`hooks.py:108`): recall is a
convenience, not a dependency, so a memory daemon that is down degrades to "no
recalled context" instead of raising into the SDK on every prompt.

**The subtle part: recall is a use.** `reinforce=True` means *surfacing* strengthens
(M4 activation rises). The memories ReLife keeps leaning on stay strong; the ignored
ones fade. The injection mechanism *is* the reinforcement mechanism — the feedback
loop that makes the cognitive model self-tune under real usage.

**Why this loop is safe (the runaway you must understand).** Reinforcement-on-recall
is positive feedback: surfaced → reinforced → higher activation → ranks higher → more
likely to surface again. If recall were naive, that would degrade into a *popularity
contest* — the same high-activation memories crowd out genuinely relevant ones, and
recall becomes self-reinforcing noise. It doesn't happen here because of M5's **Stage
1 relevance gate**: a memory must keyword/semantic-match the prompt to even enter the
candidate pool, and activation is *not* consulted there. So the loop can only ever
strengthen *relevant* memories; an irrelevant one never gets the "use" that would
strengthen it. **The relevance gate sits upstream of the feedback loop and starves it
of fuel.**

**Why the byte budget.** Injected recall competes for the *same* context window as the
actual task and conversation, and every token costs Max budget. Unbounded injection
would let the *recalled past* crowd out the *present task*, on every prompt. The cap
protects working room and budget. The fixed memory→skill→workflow priority is safe
because the candidate set is small (k=5/2/1) and memories are one-liners, so the cap
rarely binds; when it does, losing the rarely-relevant single workflow is the cheapest
sacrifice (and de-dup ran first, so a workflow overlapping surfaced memories was
redundant anyway).

#### Hook 2 — `_event_hook` (PostToolUse)

Deliberately dead simple (`hooks.py:114`): after *every* tool call, write one row to
the event log — tool name, a 120-char `_brief` of what it did, tagged with
`session_id` as `task_id`. Two tells:

- Wrapped in `try/except … pass` (`hooks.py:127`) — **journaling must never break a
  run.** Observability is strictly subordinate to the task.
- It stores a **brief, not just the tool name.** Three `Bash` calls are useless as
  "Bash, Bash, Bash"; the *command* is what lets consolidation's `_action_label` later
  distinguish `git-clone` from `test` from `git-push`. The journal captures *actions*,
  not just *tool types*.

#### Hook 3 — `_episode_hook` (Stop)

On run end (`hooks.py:131`): pop the stashed `(prompt, start_id)` for the session; if
none, bail. Pull this session's events **after `start_id`** — this turn's, not the
whole session's; if fewer than `EPISODE_MIN_EVENTS` (3) happened (`hooks.py:142`),
bail (too little to be worth remembering). Collapse the tool
sequence (consecutive dups removed). Save a deterministic one-liner
`"Task: <intent> | Approach: <tool → tool>"` as `kind="episode"`.

**Why stash the prompt (and the watermark) at the *start*?** Each hook callback only
receives *its own event's* payload. At `Stop`, the user prompt is **not in scope** — it belonged to the
`UserPromptSubmit` event, which already fired and is gone. The only way the episode
hook can know the intent is to have captured it earlier. And `_last_prompt`
(`hooks.py:83`) is a **dict keyed by session_id**, not a single string, so
**concurrent sessions don't clobber each other's prompts** — which stopped being a
theoretical nicety the moment the server hosted eight sessions in one process (M9).

**Why the episode floor?** `consolidate._cluster_episodes` groups episodes by token
overlap and mints a `pattern` memory (importance 0.7) from any cluster hitting
`RECUR_THRESHOLD`. Trivial one-tool runs all share the same boilerplate
(`"Task:"`, `"Approach:"`, a single common tool), so they'd **cluster on noise** and
mint a *false pattern* that then pollutes recall. The floor ensures the clusterer only
sees runs with enough real content that overlap means *genuine* procedural similarity.

**The unifying idea of 6a.** All three hooks share one philosophy: **the learning loop
is involuntary infrastructure, not agent behavior.** Recall injected, events
journaled, episodes captured — all *around* the agent, by the harness,
deterministically, with failures swallowed so they never sabotage the task. And the
fail-soft discipline is principled: you swallow errors where the failure mode is *lost
data* (journaling, episodes — future learning degrades, recoverable) but **never**
where it's *wrong action* (`classify()`, the recall gate — present-tense safety). The
asymmetry is the rule: present-tense harm gets fail-closed; future-tense learning gets
fail-soft.

### 6b — The two offline passes: "sleep" vs "dream"

6a was the *online* loop (around each prompt). 6b is the *offline* loop — two passes
that run *between* tasks and reshape the store itself. ReLife deliberately has **two**,
and the whole design rests on why they're separate.

**The mental model.**
- `consolidate.py` = **slow-wave sleep**: cheap, automatic, constant, **no
  consciousness** (LLM-free). Mechanical housekeeping.
- `rem.py` = **REM/dreaming**: expensive, **opt-in**, only when you choose, the **only
  pass where the LLM reflects** qualitatively on memory.

#### `consolidate.run_consolidation()` (the sleep) — `consolidate.py:362`

Runs automatically when enough events accrued (`should_auto_run`:
`events.count() − last ≥ CONSOLIDATE_EVERY`, gated by `AUTO_CONSOLIDATE`,
`consolidate.py:74`), invoked via `agent.maybe_consolidate()` after each run. Four
deterministic steps:

1. **Decay & forget (`_decay_and_archive`, `consolidate.py:83`).** Tier 1: archive
   active memories that `should_archive`. Tier 2: hard-delete rows that are *already
   archived* and `should_hard_delete`. Nothing goes active→deleted in one pass — only
   already-archived rows are deletion candidates.
2. **Dedupe (`_dedupe`, `consolidate.py:113`).** Merge near-duplicates: keyword Jaccard
   ≥ 0.9, **or** (embeddings on) semantic cosine ≥ `DEDUP_SIM` (0.90), which catches
   paraphrases sharing few tokens. The survivor is *reinforced*; the duplicate
   *deleted* — merging concentrates strength rather than losing it.
3. **Detect patterns (`_detect_patterns`, `consolidate.py:299`).** Two detectors:
   recurring **episodes** → `pattern` memory; recurring **action n-grams** from the
   event log → `pattern` memory **+ a synthesized workflow**.
4. **Synthesize workflows** — turning the journal into replayable plans.

**The n-gram cleverness (the part to really understand).**
- N-grams run over **action labels, not raw tool names** (`_tool_ngrams`,
  `consolidate.py:276` + `_action_label`, `consolidate.py:215`). For shell tools the
  *action* is derived from the command, so three different `Bash` calls become
  `git-clone`, `test`, `git-push` — distinct nodes — instead of collapsing into one
  meaningless `Bash → Bash → Bash` that would hide the real workflow and leave only
  trivial editor motions visible.
- A **meaningfulness gate** (`_is_meaningful_seq`, `consolidate.py:262`) only promotes
  a sequence that contains at least one *distinctive* action (git/test/build/docker, an
  MCP tool, browsing…). Pure `Write → Edit` editor motions are a real regularity but
  *not a workflow*, so they're skipped, not turned into noise.
- **Maximal sequences win** (`_contains`, `consolidate.py:269`): process longest-first
  and drop any sequence already contained in an accepted longer one. So you get
  `clone→test→push` once, not also its sub-sequences as separate workflows.

So a recurring real procedure becomes an *automatically* synthesized workflow, which
`_recall_hook` then surfaces next time a matching task appears. **The journal from 6a
feeds workflow synthesis in 6b, which feeds injection in 6a. The loop closes.**

**Consolidation never crosses a space.** Decay is per row, so it needed nothing. But
`_dedupe` groups by space before merging, episodes are clustered per space
(`_cluster_episodes(space=)`), tool n-grams are mined per space (events carry a
`space` column too), and `_detect_patterns` writes each pattern memory and
synthesized workflow **into the space it came from** (report lines are prefixed
`[space]`). Without that, one agent's sweep could merge away another agent's memory,
or teach the user's default memory a habit that only a crew reviewer has.

**Where the throttle lives — and why it must not move.** `should_auto_run()` reads
the event count against a watermark. The count and the pass are decided **together
on the memory side** (`MemoryService.maybe_consolidate`, `service.py:331`), and
`agent.maybe_consolidate()` just calls that. The tempting refactor — gate
client-side, then call `consolidate()` — silently breaks under a daemon: the gate
would read the *caller's* (empty) local event log while the work runs server-side,
so auto-consolidation would **never fire** (commit `85cd67c` was exactly this
lesson). Any "has enough accrued?" decision about server-side work belongs
server-side.

#### `rem.py` / `run_rem()` (the dream) — `rem.py:283`

The qualitative judgement consolidation *structurally cannot do*: deterministic dedup
can't tell a memory is **wrong, unsafe, hallucinated, or contradictory** — that needs a
model to read and reason. REM asks the LLM to be an **adversarial critic** over recent
memories. But the model is an **advisor only**; every safeguard exists to make "let an
LLM judge our memory" safe:

1. **Bounded input — the replay buffer (`_select_buffer`, `rem.py:96`).** Reviews only
   memories *new since the last pass* (watermark `last_reviewed_id` in
   `data/rem_state.json`), **most-salient-first** (importance as the surprise/dopamine
   proxy). Falls back to most-recent if nothing's new. Capped at `REM_BATCH_MAX` (40).
2. **A reference frame (`_reference_set`, `rem.py:111`).** Established knowledge
   (preferences, patterns, pinned high-importance memories, capped `REM_REFERENCE_MAX`
   = 30) for the critic to check contradictions *against* — it may act on the buffer,
   not the reference set (except naming a reference as the "keep" side of a
   contradiction).
3. **Advisor, not actor (`_apply`, `rem.py:179`).** The model returns *verdicts as
   JSON*; the application is **deterministic and runs here, not in the model.** Allowed
   actions: `keep` / `prune` / `reweight` only — it **cannot edit memory text.**
4. **Reversible.** The only destructive action is `archive` (recoverable), never
   `delete`.
5. **Confidence-gated.** Verdicts below `REM_MIN_CONFIDENCE` (0.7) are ignored.
6. **Capped.** `REM_MAX_PRUNE_FRACTION` (0.25) bounds how much one pass can archive;
   intents are applied most-confident-first until the cap, then skipped.
7. **Journaled.** Every applied action is appended to `data/rem_journal.jsonl` for
   audit/recovery (`_journal`, `rem.py:86`).
8. **SDK-free via injection.** The model call is injected as `ask_model` (default
   `agent.ask_model_oneshot`, `rem.py:271`), so the module is unit-tested with a
   canned-JSON stub and stays SDK-free.

And `_parse` (`rem.py:160`) is defensive: garbage/prose/fence-wrapped output → empty
result → `_apply` treats it as "keep everything" (a no-op). **A misbehaving critic
degrades to doing nothing, never to corruption.**

#### Why two passes (the architecture, not a slogan)

There are **two kinds of memory maintenance with opposite cost/risk profiles**:

- **Mechanical housekeeping** (forget/merge/habit-form) is cheap, safe, and *should
  happen constantly* → deterministic, LLM-free, auto-runs. **Putting an LLM here is not
  "a bit pricey" — it's structurally dangerous.** Consolidation auto-runs on event
  volume, so the *harder the user works, the more often it fires*, and every fire would
  draw from the **same finite Max session budget the user's task is spending** (M1).
  The perverse result: productivity accelerates budget drain, and a long productive
  session hits the Max limit *early*, stalling mid-work on bookkeeping nobody asked
  for. That breaks ReLife's central promise — runs on the *subscription*, no metered
  cost. Hence "consolidation is LLM-free" is a **guard-tested invariant.**
- **Qualitative judgement** (is this wrong/unsafe/contradictory?) *requires* a model,
  which is expensive and fallible → opt-in, budget-gated, never auto-run, and wrapped
  in advisor-only + reversible + capped + confidence-gated + journaled guardrails so
  the fallible judge can't corrupt the store.

Mixing them would either make sleep too expensive or make dream too dangerous. **The
separation *is* the architecture.** Guard tests assert `consolidate.py` never
references the SDK/`ask_model` and `rem.py` keeps its SDK import lazy.

> M6 mastery check: three hooks (one factory, per-client) make recall/journal/reflect
> involuntary infrastructure, every memory call off the loop;
> reinforcement is safe because the relevance gate precedes the loop; the prompt is
> stashed because it's out of scope at Stop; the episode floor prevents boilerplate
> false patterns; fail-soft is correct only where failure costs data, not safety;
> consolidate must be LLM-free because it auto-runs on the shared Max budget; the
> throttle decides where the data lives; consolidation stays inside each space; REM is
> opt-in + advisor-only + reversible because it's the fallible LLM path.

---

<a name="m7"></a>
## M7. Build orchestration (`relife/build/`)

### The problem it solves

A single `do`/`chat` run lives in **one context window**. A large project — many
files, many subsystems — won't fit; the context fills, the agent loses the thread, and
quality collapses. `relife build` scales past that by **decompose → delegate →
resume**.

### The three moving parts

1. **`BuildLedger` (`ledger.py`)** — the durable plan + progress, at
   `data/builds/<id>/ledger.json` with a human-readable `plan.md` mirror. Pure and
   deterministic (no agent calls), so it's fully unit-testable. It is the **source of
   truth for resume.** Key fields: `build_id`, `spec`, `workspace`, `session_id`, and a
   list of `Milestone`s each with a `status` (pending/in_progress/done/failed) and a
   `summary`. Writes are **atomic-ish** (temp file then `replace`, `ledger.py:115`) so a
   crash mid-write can't corrupt the ledger. `latest_for(workspace)` (`ledger.py:88`)
   finds the most-recently-updated ledger for a workspace (for `--resume` with no id).
   (The same tmp+`replace` idiom reappears in `ScheduleStore` and `RunStore` — M10.)
2. **`build_server` (`server.py`)** — an in-process MCP server `relife_build` exposing
   three tools to the orchestrator, **bound to one ledger via closure** so they mutate
   the right ledger on disk: `build_plan_set` (record the decomposition),
   `build_milestone_update` (set status + summary), `build_status` (read the ledger).
   Surfaces as `mcp__relife_build__*` → already auto-allowed by the trusted
   `mcp__relife` prefix (no permission change).
3. **The `builder` subagent (`agents.py`)** — an `AgentDefinition` the orchestrator
   delegates each milestone to **via the Task tool**. Its prompt (`build/agents.py:53`)
   constrains it hard: implement *exactly one* milestone, match existing conventions,
   **verify it** (build + tests), stay scoped, defer outward actions to the
   orchestrator, and **report back a concise summary only — not a transcript.**

### Why subagents: fresh context per milestone

This is the core idea. Each milestone runs in a **fresh `builder` context window** via
the Task tool. The orchestrator stays *small* — it holds the plan and the concise
summaries, not the implementation detail of every milestone. The builder absorbs the
detail and throws its context away when done, returning only the outcome. So the
orchestrator's context grows with the *number of milestones* (cheap), not with the
*total implementation work* (expensive). That's what lets a build scale past a single
context window.

The builder's tool list (`build/agents.py:54`) is read/search/edit/shell + browser + memory,
but **not** the build-ledger tools — those belong to the orchestrator alone.

### `run_build()` — wiring and resume (`orchestrator.py:43`)

A fresh build: create a ledger, give the orchestrator the initial prompt ("decompose
into milestones with `build_plan_set`, then delegate each to a `builder`",
`orchestrator.py:24`). A resume: load the ledger (by id, or the latest for the
workspace), and give the resume prompt ("here's the ledger; call `build_status`,
spot-check that 'done' milestones really exist, continue from the first unfinished
one", `orchestrator.py:33`).

Three robustness decisions worth understanding:

- **Resume runs in the ledger's own workspace, not the CLI's cwd.** The build lives
  where it was created; resuming from any directory must `cwd` into the *ledger's*
  workspace, or the agent couldn't see its prior work and would build in the wrong
  place.
- **The `session_id` is persisted to continue the same conversation**
  (`orchestrator.py:116–118`): each `ResultMessage` with a session id is written to
  the ledger, so the next `--resume` continues the same CLI session.
- **Expired-session fallback** (`orchestrator.py:99–109`). A persisted `session_id`
  can be *gone* — it expires across a Max session-limit reset, and the `claude`
  subprocess then fails to start. So resume tries the saved session, and on
  `ClaudeSDKError` it **drops the stale handle and starts a FRESH session**,
  re-injecting the full ledger in the resume prompt — so no milestone progress is
  lost even though the conversation handle died. This directly reflects the Max
  constraint from M1.

### The CLI subtlety (`cli.py:171`)

`--resume` is a **boolean flag** (`cli.py:181`), and the positional `spec` argument
(`cli.py:173`) *doubles* as the optional build id on resume. This is deliberate: if
`--resume` took a value, it would swallow the following option (e.g. `--workspace`).
As a boolean, you write `relife build --resume <id> -w <path>` unambiguously, and
`--resume` alone resumes the most recent build for the workspace.

> M7 mastery check: decompose → delegate (each milestone to a fresh `builder` via Task)
> → resume (BuildLedger + persisted session_id); fresh contexts keep the orchestrator
> small so builds scale past one window; resume is robust to a dead session because the
> ledger re-injects state and falls back to a fresh session.

---

<a name="m8"></a>
## M8. Memory as a process — the daemon split (`memory/remote/`)

### What it is, and what it is *not*

`relife memory serve` runs the memory layer as a **standalone FastAPI daemon** on
`127.0.0.1:8787`; setting `RELIFE_MEMORY_URL` makes every consumer in another process
talk to it over HTTP instead of opening `data/relife.db` themselves. It is **opt-in**:
unset the variable and everything is in-process, exactly as before. Nothing in the
default path requires the `[daemon]` extra to be installed.

Why would you want it? Two reasons, both about *multiple writers*: (a) several ReLife
processes (a CLI run, the agent server, a scheduled run) sharing one brain without
racing on SQLite, and (b) the always-on server (M9) not having to own the store's
lifetime. The split is the payoff of M5's seams — and the interesting part is what
*didn't* change.

### The three files, and the dependency rule between them

- **`wire.py`** — dict ⇄ dataclass for `Memory`, `Skill`, `Workflow`, `Event`,
  `ConsolidationReport`, `RemReport` (`wire.py:45–177` in that file's own numbering).
  **Dependency-free** — no fastapi, no httpx — and imported by *both* sides, so the
  daemon and the client can literally never disagree on a field list. Every field is a
  string or number, so JSON round-trips losslessly (unicode included); `activation()`
  is *derived*, not serialized. This is the single source of truth for the wire
  format; a contract test asserts the round-trip.
- **`daemon.py`** — `create_app(db_path, token, skills_dir, workflows_dir,
  spaces_dir, agents_path, mcp_hosts)` returns the FastAPI app; `serve()` runs
  uvicorn. Routes mirror the `MemoryClient` protocol one-to-one
  (`daemon.py:158–321`): `/save`, `/recall`, `/forget`, `/archive`, `/memories`,
  `/memories/{id}`, `/count`, `/spaces`, `/spaces/copy`, `/spaces/archive`,
  `/spaces/{space}/export`, `/spaces/import`, `/consolidate`, `/consolidate/maybe`,
  `/dream`, `/skills/*`, `/workflows/*`, `/events/*` — plus `/mcp`, the same memory
  as an MCP server for other agents (M14).
- **`http_client.py`** — `HttpMemoryClient`, which implements the same protocol over a
  **pooled keep-alive `httpx.Client`** (a fresh TCP handshake per recall would dwarf
  the ~1–3 ms the call itself takes) and **rebuilds real `Memory`/`Skill`/… objects**
  from the wire dicts, so consumers get identical types either way.

### The binding trick (the thing to really understand)

The obvious design is "inject a `MemoryStore` into the service." The daemon
deliberately does **not** do that. `_bind_db()` (`daemon.py:51`) instead **reassigns
the module-level globals** `store._DB_PATH` and `events._DB_PATH` to the daemon's DB,
and `_bind_dirs()` (`daemon.py:68`) does the same for `skills._SKILLS_DIR`,
`workflows._WORKFLOWS_DIR` and `spaces._SPACES_DIR` (where non-default spaces keep
their procedures). Then it constructs a plain default `MemoryService()`.

Why? Because `consolidate()` and `dream()` **mine the module-level defaults** — they
read the event log, sweep the store, and *write synthesized workflows* via the module
functions, never via a client. (Via the client would be worse than wrong: the daemon
calling itself over HTTP from inside a request handler would **deadlock its own event
loop**.) An injected store would cover `save`/`recall` but leave upkeep pointed at the
wrong DB. Reassigning the globals means *every* operation — reads, writes, and
upkeep — hits one database and one set of dirs. The test-isolation mechanism
(reassignable `_DB_PATH`) turned out to be the deployment mechanism too.

`_bind_dirs` leaves a `None` dir **untouched** on purpose: a test that builds an app
against a tmp DB without dirs must not clobber the real skills dir for the rest of
the process.

### Concurrency: serialize on the loop, on purpose

Every handler is `async def` calling the **sync** service directly (`daemon.py:12`).
That means all DB + embedding access is serialized on the single event loop — no
threadpool writers, no SQLite lock contention. For a single-user daemon that's the
right trade: correctness for free, and per-request offloading is a later refinement
if it's ever needed. The exception is `/dream`: REM can run for minutes, so on the
*client* side it's the one `async` protocol method and is dispatched via
`anyio.to_thread` with `timeout=None` (`http_client.py:182`) so it doesn't block the
caller's loop; the `ask_model` callable **can't cross the wire**, so the daemon uses
its own default (`agent.ask_model_oneshot`) — meaning the daemon's machine needs the
logged-in `claude` CLI too.

### Failing identically on both transports

`_post_write()` (`http_client.py:87`) and the shared `_check` map the daemon's HTTP
400 back to the `ValueError` the in-process path raises for invalid input (a bad
space name included). That's not cosmetics: the
**conformance suite is parametrized over both transports**, so the same test asserts
the same exception whether the client is local or HTTP. The seam is only real if
failure modes match, not just success modes.

### The sidecar: advisory, never a lock

`serve()` writes `data/relife.db.daemon` (pid + url) next to the DB and removes it on
shutdown (`daemon.py:326, 268`). A process that then goes in-process **without**
`RELIFE_MEMORY_URL` sees the sidecar and **warns** (`client._warn_if_daemon_running`,
`client.py:296`) — it does not refuse. A crash or SIGKILL can orphan the sidecar, and a
stale marker must never brick the default path. `relife doctor` surfaces the same
condition as a `warn` with the fix (`doctor.py:359`).

### What it cost, and what it proved

- Zero consumer changes: hooks, MCP tools, CLI, server all still call
  `default_client()`.
- The `MemoryClient` protocol grew to cover skills, workflows *and* the event log
  (commits `cb6b651`, `a484471`) — because the split is only clean if the **whole**
  consumer-facing surface crosses the seam. Half a seam is a bug generator.
- One real lesson (M6): the auto-consolidate throttle had to move server-side.
- When spaces arrived the seam paid again: the conformance suite gained space cases,
  and the daemon gained routes — consumers still only call `default_client()`.
- The daemon's **only lifespan** is the MCP session manager for `/mcp`. The REST routes
  still need none (the schema is initialized eagerly), so a plain `TestClient` keeps
  working for them; only the `/mcp` tests enter the lifespan.

> M8 mastery check: opt-in by env var; `wire.py` is the shared, dependency-free
> contract; the daemon binds module globals instead of injecting a store because
> upkeep mines the defaults (and calling itself over HTTP would deadlock); handlers
> serialize on the loop; 400⇄`ValueError` keeps both transports failing identically;
> the sidecar warns, never refuses.

---

<a name="m9"></a>
## M9. The always-on agent server + web console (`relife/server/`)

### The problem it solves

`do`/`chat` are **cold**: a process per task, a `ClaudeSDKClient` subprocess spun up
and torn down, a TTY for approvals. That's fine at a keyboard and useless for an agent
that should *be there* — reachable from a browser, holding a conversation across
hours, and (M10) running things on its own. `relife serve` turns the one-shot loop
into a **long-lived process** hosting persistent sessions, streaming their work to a
self-contained web UI, and routing approvals to the browser.

It mirrors the memory daemon's shape on purpose: a side-effect-free `create_app()`
factory, a `serve()` with lazy uvicorn, token auth, env knobs in `config.py`. But it
is a far more dangerous process than the daemon — **it runs shell commands and edits
files** — and almost every design decision below follows from that.

### `AgentSession` — one conversation, one subprocess, one loop (`session.py:112`)

A session owns exactly one `ClaudeSDKClient`, **kept open across turns** — the same
thing `run_chat` proves works. `start()` (`session.py:135`) is M2's three-piece
assembly with one substitution: `make_approval_callback` instead of the TTY callback.
A worker task (`_run`, `session.py:254`) drains an inbound queue → `client.query` →
`receive_response()` → `to_event` → publish. Each event gets a monotonically
increasing sequence id and lands in a **ring buffer** (`_RING_MAX` = 500) *and* in
every subscriber's queue.

Three details that carry weight:

- **One turn can't kill the session** (`session.py:268`): any exception in a turn is
  published as an `error` event and the worker continues to the next turn.
- **Publishing is activity** (`session.py:219`): every emitted event `touch()`es the
  session, so a turn streaming longer than the idle timeout with no browser attached
  is never reaped out from under itself.
- **The user's turn is echoed** as a `user` event *when the worker dequeues it*
  (`session.py:260`) — not when HTTP accepted it. M10's run recorder keys on exactly
  that echo.

### `ApprovalBroker` — a permission prompt as a future (`session.py:53`)

When `classify()` says "ask", the callback calls `broker.request(...)`. The broker
publishes an `approval_request` event (tool, reason, and a 400-char `_tool_brief` so
the card shows *what* is about to leave the machine), then **awaits an
`asyncio.Future`** with `asyncio.wait_for(fut, timeout)`. A *different* request
handler — `POST /sessions/{id}/approvals/{approval_id}` — calls `broker.resolve()`,
which sets the future. Timeout → `False` → deny (`session.py:89–91`), the same
default as the non-interactive TTY.

This works cleanly *because* everything is on one event loop: the suspended worker
and the resolving request interleave with no locks. It's the same concurrency model
as the daemon; the difference is that here a suspended coroutine is a *blocked agent
run*, which is precisely what you want — the agent physically cannot proceed past
an outward action until a human clicks.

### The SSE stream and reconnect (`app.py:314`)

`GET /sessions/{id}/events` is a `StreamingResponse` (no `sse-starlette`
dependency). It:

1. reads `Last-Event-ID` (header, preferred) or `?last_id` (query),
2. calls `session.subscribe(last_id)` (`session.py:200`), which snapshots the ring
   *and* registers the queue **with no `await` between them** — so on a single loop
   no event can slip between backlog and live stream,
3. yields the backlog, then live events, sending `: ping` every heartbeat and
   **touching the session on each ping** so a watching browser is never reaped.

A subscriber that stops reading (a wedged tab, a paused debugger) hits
`_SUB_QUEUE_MAX`; the publisher **drops that subscriber's oldest event** rather than
growing memory (`session.py:230`). The ring + `Last-Event-ID` replay is how the
client catches back up. Nothing here is unbounded.

### The UI's reattach (why `GET /sessions/{id}` exists)

The obvious UI — `POST /sessions` on page load — was wrong in a way that only shows
up in use: every reload spawned a **second `ClaudeSDKClient` subprocess**, orphaned
the first until the idle reaper (an hour), lost the transcript, and after
`AGENT_MAX_SESSIONS` reloads returned 429 with no way out. So the page remembers its
session id in `localStorage` (`web/index.html:325`), **probes** `GET /sessions/{id}`
(`app.py:270`) on load, and if alive reattaches by streaming `?last_id=0` — replay the
whole ring (`web/index.html:455`). A fresh page has no `Last-Event-ID` (that's
per-`EventSource`), which is why the query form exists; the route prefers the header
so an *automatic* reconnect still resumes rather than replays. A "new session" button
`DELETE`s the current one so a slot can be freed from the browser.

### `security.py` — the policy as pure functions (`security.py`)

Every decision about *who may talk to the server* and *where a session may write* is
a plain function with no I/O — the `cognitive.py` discipline applied to security:

- **`token_matches`** (`:21`) — constant-time (`secrets.compare_digest`); `expected is
  None` = auth disabled.
- **`presented_token`** (`:30`) — from **either** `Authorization: Bearer` or the cookie.
- **`same_origin`** (`:46`) — CSRF: a mutating request's `Origin` must match `Host`. A
  *missing* Origin is allowed — that's curl or an SDK client, which carries a bearer
  header a third-party page can't ride on.
- **`host_allowed`** (`:63`) — the DNS-rebinding guard (release hardening): a
  *tokenless* server answers only to loopback `Host` names (`is_loopback`, `:84`) plus
  any listed in `RELIFE_AGENT_ALLOWED_HOSTS`. Without it, a web page on an
  attacker-controlled name that resolves to `127.0.0.1` could drive a tokenless server
  from the user's own browser — same-origin checks pass, because to the browser it
  *is* the same origin.
- **`guard_bind`** (`:94`) — **refuses** a non-loopback bind without a token.
- **`resolve_workspace`** (`:111`) — confinement to `AGENT_WORKSPACE_ROOT`, resolving
  *before* checking so `..` and symlinks are caught (M3's `_under`, one level up).
- **`AttemptLimiter`** (`:132`) — fixed-window per-address counter for token guessing.

`app.py` only wires these to routes (`app.py:181–199`).

### Why auth accepts a cookie (and why you must not "simplify" it away)

A browser **`EventSource` cannot set request headers.** With header-only bearer auth,
the SSE stream — the one route that carries the *entire transcript and every approval
prompt* — was the one route the UI could never authenticate. So `POST /auth`
(`app.py:227`) exchanges the token for an **HttpOnly, `SameSite=Strict`** cookie, and
every route accepts *either* carrier. Bearer stays for programmatic clients.

Cookie auth is *ambient* — the browser attaches it to any request to that origin —
which is why mutating routes additionally require same-origin (`mutate =
[same_origin, token]`, `app.py:199`). Drop the cookie and the UI can't see its own
agent; drop the Origin check and a cross-site page could drive it. Both halves are
load-bearing. Failed `/auth` attempts are rate-limited per client address.

### The workspace root is a privilege boundary, not a convenience

`classify()` **auto-allows writes inside a session's workspace** (M3). So if
`POST /sessions` accepted a raw path, the *HTTP caller* would be choosing the
auto-allow blast radius — `{"workspace": "C:\\"}` and the whole disk is writable
without a prompt. `resolve_workspace` confines every server-created session to
`AGENT_WORKSPACE_ROOT` (default `./workspace`) and rejects escapes with a 400
(`app.py:257`). The CLI's `--workspace` is deliberately *not* confined: that's the
local user speaking directly, not a request body.

### Everything is bounded, because the process never exits

Each session is a **subprocess**. A long-lived process with unbounded anything is a
resource-exhaustion lever even on loopback. So: `AGENT_MAX_SESSIONS` (429), an idle
reaper under the app lifespan (`app.py:139`) that skips sessions with a live SSE
subscriber, `AGENT_MAX_QUEUED_TURNS` (429 — `submit` uses `put_nowait` rather than
blocking a request handler per pending turn, `session.py:165`), `AGENT_MAX_MESSAGE_CHARS`
(413), `AGENT_MAX_SUBSCRIBERS` (429), and drop-oldest on lagging queues. Shutdown
`aclose()`s every session so **no subprocess is orphaned**.

### Consolidation off the loop (MVP pass 4)

After each turn the worker runs the sleep beat. Inline it was a disaster only a
long-lived process reveals: the pass does SQLite scans, dedupe and (with embeddings)
ONNX inference — seconds on a big store — and on the *one* loop that hosts every
session's SSE stream and every pending approval, **everything froze for the
duration.** `maybe_consolidate_off_loop()` (`agent.py:245`) runs it on a worker
thread. It's safe there because the store opens a fresh SQLite connection per call
and `httpx.Client` is thread-safe; the non-blocking process lock (M2) stops two
sessions sweeping at once. A pass that changed something is published as a `note`
event so the console shows the brain ticking.

### Testing with zero model calls

`create_app(session_factory=...)` (`app.py:105`) accepts an injected factory. Tests
pass a fake `AgentSession` that emits scripted events, so the **whole HTTP + SSE +
approval flow** — auth, cookie, CSRF, 429s, reattach, approval resolve/timeout — runs
without a model, the same discipline as `rem.ask_model`. Nothing in `create_app`
touches the network or spawns a task; the reaper and scheduler start under the
lifespan.

### Still local-first, fail-closed

There is **no TLS and no multi-user model**. `serve()` refuses a non-loopback bind
without a token; with a token it still prints that a reverse proxy belongs in front.
Exposing it beyond `127.0.0.1` is a deliberate future step, not a supported
configuration — the `doctor` check for it (`doctor.py:455`) says so in as many words.

> M9 mastery check: one `ClaudeSDKClient` per session, kept open; approvals are
> futures resolved by another request on the same loop, timing out to deny; SSE +
> ring + `Last-Event-ID` (and `?last_id=0` for reattach) — because every reload used
> to orphan a subprocess; cookie auth exists because `EventSource` can't set headers,
> and the Origin check exists because cookies are ambient; the workspace root is a
> privilege boundary; everything is bounded; consolidation runs off the loop.

---

<a name="m10"></a>
## M10. Autonomy that delivers — scheduler & run outcomes

### The problem, in two halves

M9 gave the agent persistence. Autonomy needs one more thing: **triggers** — "check my
inbox at 09:00 on weekdays," "every 30 minutes, pull and run the tests." And once a
run fires with nobody watching, it must **deliver**: you need to know afterwards what
it did, what it cost, and — most of all — **what it could not do because you weren't
there to approve it.** The scheduler is the first half; run outcomes are the second.
Both were designed around one constraint: a scheduled run is a *turn*, not a new kind
of thing.

### Schedules are a pure model (`schedules.py`)

A `Schedule` (`schedules.py:191`) is a durable record: name, task text, a `spec`, a
confined workspace, `enabled`, `next_run_at`, and a bounded inline run history.
Two spec forms, kept in the user-facing shape:

- `{"every": "30m"}` — interval. `parse_every` (`:45`) accepts `s|m|h|d`;
  `normalize_spec` (`:82`) **floors** it at `AGENT_SCHEDULE_MIN_INTERVAL` (300s)
  because every firing spends Max budget, so a one-minute schedule is a mistake to
  refuse, not a wish to honour.
- `{"at": "09:00", "days": ["mon","fri"]}` — daily at a **local** wall-clock time.
  `next_run` (`:124`) works in naive local time and converts with `timestamp()`, so
  09:00 stays 09:00 across a DST change instead of drifting by the offset.

Everything time-related takes an **explicit `now`**, so the cadence math is testable
at any instant. `ScheduleStore` (`:325`) is a JSON file at `data/schedules.json`
with atomic tmp+`os.replace` writes (the ledger idiom) — and it writes **nothing**
until the first change, so `create_app()` stays side-effect-free.

### The scheduler fires *turns* (`scheduler.py`)

`Scheduler.run()` is one background task under the app lifespan (like the reaper),
ticking every `AGENT_SCHEDULER_TICK` (30s). `tick(now)` (`:162`) fires each due
schedule (`enabled and next_run_at <= now`). A firing is `_submit` (`:214`):

1. Each schedule owns a **session** in its confined workspace — created via the same
   `resolve_workspace` as `POST /sessions`, reused while alive, recreated after the
   reaper closes it. The UI's **watch** button attaches the console to that session
   through M9's reattach path.
2. The turn text is `scheduled_prompt()` (`:47`): a preamble saying the run is
   automatic and possibly unattended, that a needed approval may be denied by
   timeout, and to **report that rather than retry**, then the task. The UI renders
   it compactly as `⏱ scheduled · name`.
3. `session.submit(prompt)` — and from here it is *exactly* a typed turn: same hooks,
   same journaling, same learning, same `classify()`, same approval broker.

That last point is the design. There is no "scheduled-run mode" in the agent. Because
the approval path is M9's future-with-timeout, **nobody watching ⇒ the ask-case times
out ⇒ deny** — the CLI's non-interactive rule, inherited with zero new code. One
safety rule, three surfaces (M3).

### Grants ride one turn (MVP pass 9)

The one exception to "nobody watching ⇒ deny" is a schedule's **grants** (policy in
M3). They are attached to the *turn*, not the session: `AgentSession.submit(text,
grants=)` (`session.py:165`) queues the grants with that turn's text, so typing into
the same session via *watch* gets none. `make_approval_callback(..., preauthorize=)`
consults `grant_allows` after `classify()` says ask and before the broker. Uses are
capped per turn by `AGENT_GRANT_MAX_USES` (3) — a loop, or a prompt-injected "mail me
50 times", stops at three and falls back to asking — and every use is published as an
`approval_auto` event, which the run record lists under `acted` ("done for you —
pre-approved" in the panel). The scheduled preamble tells the agent exactly what it
may do and in what shape. Schedules accept `grants` on POST/PATCH; an invalid grant
on disk is dropped on load, never widened.

**Cadence policy, all deterministic:**
- After *any* attempt the schedule advances from **now** (`fire`, `:171`) — a slot
  missed while the server was down fires **once** on the next tick, never N×.
- A schedule whose session is still `busy` (`AgentSession.busy`, `session.py:248`: a
  turn in flight *or queued*) is **skipped** for that slot rather than stacking turns —
  an hourly task that takes ninety minutes must not pile up.
- A run that can't start (session ceiling, full queue, bad workspace) is recorded as
  `skipped: …` / `error: …` and the schedule *still advances* — visible in its history,
  no tight retry against the same wall.

Tests drive `Scheduler.tick(now)` against a fake manager — never the real loop, never
a model.

### Run outcomes: the ring buffer must not be the only copy (`runs.py`)

`submitted` only means the turn was queued. The session it ran in is reaped an hour
later, and with it the ring buffer. So for each firing the scheduler **subscribes to
the session *before* submitting** (`scheduler.py:265` — so the turn's first event can't
be missed) and spawns a recorder task (`_record`, `:291`) that collects **exactly that
turn's** events:

- it starts at the `user` echo **whose text equals the scheduled prompt** — a turn the
  user typed into the same session (via *watch*) is not ours;
- it ends at `result` → `done`, or `error` → `error`, or `AGENT_SCHEDULE_RUN_TIMEOUT`
  (2h) → `timeout` (the recorder stops waiting; the agent is not killed), or server
  shutdown → `interrupted` (`aclose`, `:150`).

Then `_finish` writes a `RunRecord` to `data/runs/<schedule>/<run_id>.json`
(bounded per schedule by `AGENT_RUN_HISTORY`, removed with the schedule) and
**upgrades the schedule's inline history entry** from `submitted` to the real
status plus tool count, cost, denied count and a summary snippet.

A run id is a URL path parameter that names a file, so `RunStore.get` refuses
anything that isn't the exact `YYYYMMDD-HHMMSS-mmm` shape `new_id` produces
(`runs.py:172`) before it touches the filesystem.

`summarize_events()` (`runs.py:31`) is pure over the `to_event` taxonomy:

- `summary` = the agent's **closing text** — everything after its last tool call (the
  preamble asks it to end with a summary); if it never used a tool, all of its text.
- `denied` = every `approval_request` whose `approval_resolved` came back `False`,
  each with the **tool and the brief the user would have seen on the card** — so
  "needed you — denied unattended" in the panel tells you exactly what to do by hand.

The recorder is a live subscriber, so the idle reaper never closes a session
mid-recording; it unsubscribes when the turn ends. If every stream slot is already
held by watchers (`TooManySubscribers`), the turn still goes out — someone is
clearly looking — but the history entry says `submitted (unrecorded: …)` rather than
a `submitted` nothing would ever upgrade (`scheduler.py:268`).

### Why this shape and not a job queue

A "proper" job system would run scheduled tasks in their own runner with their own
permission mode. That would have created a **second agent path** — a second place
where policy, journaling and learning could drift from the interactive one. By making
a firing *a turn in a session*, every property M3–M9 established applies unchanged,
and the only new code is cadence math and a recorder. The cost is that a scheduled
run shares the server's session ceiling and can be skipped when busy; both are
visible in the history rather than hidden.

> M10 mastery check: a schedule is a pure record with explicit-`now` cadence math
> and a floored interval; a firing is a *turn* in a per-schedule session, so it
> inherits hooks, journaling, learning and the timeout-to-deny approval rule with no
> new code; advance-from-now, skip-when-busy, record-and-advance on failure; the
> recorder subscribes before submitting, keys on the prompt echo, and persists
> summary + denied list because the ring buffer dies with the session; grants ride one
> turn, capped and journaled.

---

<a name="m11"></a>
## M11. The operator surface — connectors, `doctor`, memory inspection

Three smaller pieces that turn ReLife from "runs" into "runs for a person who has to
trust and maintain it." They share one motive: **the things that go wrong here go
wrong silently or late**, and each piece moves the failure to where you can see it.

### Connectors (Gmail / Google Calendar / Google Drive) — the first real outward reach

The claude.ai connectors ride the **logged-in subscription**: no keys, no local
server, nothing in `mcp_servers`. The CLI attaches them to every session from the
account — `setting_sources=None` notwithstanding — and they surface to the agent as
`mcp__claude_ai_<Service>__<tool>`. Linking a Google account happens **in-session**
on first use (say "connect my gmail" → approve the browser sign-in).

Two consequences shaped the code:

- **The permission policy had to become verb-based** (M3), because the tool names
  are Google's and the read/write split is the only stable thing about them. Reads
  run autonomously; sends/creates/deletes ask; the approval card shows recipient and
  subject via `_tool_brief`.
- **What the connectors can actually do was checked against the live schemas** after
  1.0 (linking Gmail + Calendar in a Claude Code session exposes the same account
  connectors ReLife gets, without spending a ReLife turn). Gmail **can send**
  (`send_message`, `reply`, `forward` — an earlier note said drafts only); recipients
  are lists of plain addresses, content fields are camelCase (`htmlBody`,
  `forwardText`), attendees are `{email, …}`. That check found the two grant bypasses
  in M3 (`draftId`, `replyAll`); `tests/test_grants.py` pins the real names and shapes.
- **`doctor` has to check them** (`_check_connectors`, `doctor.py:394`): it runs
  `claude mcp list`, parses the `claude.ai <Service>` lines (`parse_mcp_list`,
  `:384`), and distinguishes *not enabled on the account* (fix: claude.ai → Settings
  → Connectors) from *enabled but Google not linked* (fix: link it from a session).
  Without this, the first symptom is a mid-run tool error behind the SDK.

### `relife doctor` — a pure check matrix over injected probes (`doctor.py`)

ReLife leans on a lot outside the package: the `claude` CLI and its login, Node for
the browser MCP, `gh`, SQLite with FTS5, optional extras, the connectors, the memory
daemon, and now the agent server and its schedules. Each fails **late and
cryptically** — mid-run, inside a subprocess — or, for the always-on side, **silently**:
schedules only fire while `relife serve` is running; a run that needed an approval
just records a denial; a non-loopback bind without a token is refused at start.

The structure is the repo's policy-module discipline once more: `run_checks(p)`
(`doctor.py:538`) is **pure over an injected `Probes` bundle** (`:44` — `which`, `run`,
`env`, `import_ok`, `fts5_ok`, `http_get`, `schedules`, …; every field a value or a
callable), so the whole matrix is unit-tested with scripted environments;
`default_probes()` (`:78`) is the one place that touches the real machine. Each
`Check` carries a `status` (`ok|warn|fail|skip`), a `detail`, and a **`fix`** — the
command or setting that resolves it — and the process exits 1 on any `fail`.

Two checks worth knowing exist:

- **`api key`** (`:233`) warns if `ANTHROPIC_API_KEY` is set — the M1 constraint made
  executable.
- **`agent server` + `schedules`** (`:455`, `:502`): bind/token sanity first (`serve()`
  would refuse), then reachability; *not running* is normal for CLI use but a `warn`
  when enabled schedules exist ("they will not fire"); and the schedules check reads
  each enabled schedule's last run and reports **"last run failed: …"** and **"needed
  you: … (N denied)"** in one line — doctor is where someone looks when "nothing seems
  to happen."
- **`crews` + `agents`** (platform pass): on Python 3.14 the crews check *skips* with
  the exact `py -3.12 -m venv .venv` recipe (CrewAI requires < 3.14); otherwise it
  reports whether the `[crewai]` extra imports. The agents check counts registered
  agents by runtime (and how many hold an MCP token), and warns when
  `data/agents.json` had unreadable records (the original file is kept aside).

### Memory inspection & correction (`relife memory *`, MVP pass 3)

A learning system you can't inspect is one you can't trust, and one you can't
correct is one that reinforces its own mistakes (M16 names this as the biggest
risk). The `memory` sub-app (`cli.py:345`) closes that gap:

- **`search "<q>"`** — what the agent *would be shown*, ranked the same way —
  **without reinforcing** (`reinforce=False`, `cli.py:389`). Looking never distorts.
- **`list [--kind] [--archived] [--sort recent|strong|oldest]`**, **`show <id>`** —
  full record: text, tags, importance, activation, use history.
- **`forget <id>… | --query "<q>" [--yes]`** (`cli.py:464`) — **archive** (the
  reversible M4 tier, never delete), confirming unless `--yes`. Id-addressed
  `archive(id)`/`get(id)` were added to the service and the daemon routes
  (`/archive`, `/memories/{id}`) for exactly this: query-based `forget` archives
  *whatever matches best*, the wrong tool when a human points at one row.
- **`stats`** — counts, strongest memories, what has faded.
- **`--space S`** on `search`/`list`/`forget` looks at one agent's memory instead of
  the default; **`memory spaces`** lists every space with its counts; **`memory export
  SPACE -o pack.json`** / **`memory import pack.json --space S`** move a space between
  machines or homes (M13). `list`/`show` print each row's space and source.

All of it goes through `default_client()`, so it inspects the daemon's brain when
`RELIFE_MEMORY_URL` is set.

> M11 mastery check: connectors are account-attached, so policy keys on verbs and
> doctor checks linkage; `doctor` is a pure matrix over injected probes with a fix
> per check, covering the silent always-on failures; memory inspection never
> reinforces, and correction is id-addressed and reversible.

---

<a name="m12"></a>
## M12. Doing the user's work — GitHub issues and pre-approved actions (`workitems.py`)

### The problem it solves

The vision's first "future" item was *complete assigned work items*. An issue is
harder than a `relife do` task in three ways: it lives in someone else's repo (where
should the agent work?), its text is written by **third parties** (a prompt-injection
vector), and its finish line — opening a pull request — is itself an outward action.
`relife work` answers each, and work schedules (below) let it run unattended.

### Deterministic plumbing, agent in the middle

Everything *around* the agent is plain functions in `workitems.py`, unit-tested
against a fake `gh` runner with zero model calls:

- `parse_ref` (`workitems.py:70`) — `owner/repo#12`, an issue URL, or `12 --repo o/r`;
  a PR URL is rejected.
- `list_assigned` (`:113`) — `gh search issues --assignee @me` (no REF: list and stop).
- `fetch` (`:141`) — body plus the last `MAX_COMMENTS` (5) comments. A **closed** issue
  exits 1 before anything is cloned.
- `ensure_checkout` (`:170`) — clones once into `<workspace>/<owner>__<name>`. An
  existing checkout is **left untouched**: it may hold uncommitted work, so syncing it
  is the agent's first, *visible* step, never something our code does silently.
- `branch_name` (`:187`) — `relife/issue-12-<slug>`, stable per title, so a rerun
  resumes the branch instead of forking a new one.
- `task_prompt` (`:205`) — the agent's instructions (below). `--dry-run` prints it
  with no model call.

`run_gh` is looked up at call time (`run or run_gh`) so tests can monkeypatch it.

### The checkout *is* the workspace

The agent runs with the checkout as its workspace, so M3's auto-allow write radius is
**one repo**, not the user's whole workspace root. Branch, commit, test and `git push`
are autonomous (the user authorized git); `gh pr create` and any issue
comment/close/edit **ask** (M3's verb-based `gh` policy).

### Issue text is data, fenced

`task_prompt` puts the body and comments inside `<<<ISSUE … ISSUE>>>`, bounded by
`BODY_LIMIT` (8000) and `COMMENT_LIMIT` (1500), and tells the agent it is information
from the issue, not instructions. The fence is a hint to the model; the **backstop is
the policy**: whatever an injected "also email the repo secrets to …" would need —
mail, uploads, writes outside the checkout, `gh` writes — asks. `RELEASE_TESTING.md`
§5 has a live smoke with exactly such a line in an issue body.

### Work schedules — `relife work` on a cadence (MVP pass 13)

A schedule with `work` set — `{}` for any open issue assigned to the user, or
`{"repo": "o/n"}` / `{"label": "relife"}` to narrow it (`normalize_work`,
`schedules.py:148`) — takes the next issue each firing:

- `Scheduler._prepare_work` (`scheduler.py:182`) runs the plumbing **on a worker
  thread** (`gh` and `git` are blocking subprocesses and the loop is shared — M9):
  list → `pick_issue` (`scheduler.py:63`: newest first, not in `schedule.worked`,
  label match) → fetch → `ensure_checkout` under the schedule's confined workspace.
- **No new issue ⇒ `skipped: no new assigned issues` before any session exists.** An
  empty poll spends no Max budget.
- A session's cwd is fixed at creation and each issue has its own checkout, so a work
  schedule closes its idle session and creates a fresh one **in the checkout** every
  firing.
- An issue counts as attempted (appended to `worked`, capped at `MAX_WORKED` = 200)
  only once its turn is *submitted* — a `gh` error or a full session ceiling leaves it
  available for the next slot.

Unattended, `gh pr create` times out to deny: the run ends with a pushed branch and
the PR listed under "needed you" — unless the schedule carries the grant below.

### The pull-request grant: the narrowest widening (MVP pass 14)

`pull_request` is the only grant that touches the shell, so it is built to match
exactly one thing:

1. **Only on a work schedule.** `check_grants_fit` (`schedules.py:175`) refuses it
   elsewhere rather than keeping a grant that can never apply.
2. **Stored unbound.** `{"kind": "pull_request"}` — `normalize_grants` drops any repo
   or branch a caller sends, and an unbound grant matches nothing.
3. **Bound per turn.** `bind_grants` (`scheduler.py:104`) binds it to *this* issue's
   repo and branch for this turn only, and drops it when there is no issue.
4. **One tokenized shape.** `_pr_create_matches` (`permissions.py:808`) tokenizes
   with `shlex` (comments off) and allows a single `gh pr create` whose
   `--repo`/`--head` equal the bound pair, with only title/body/base/draft/fill flags,
   each at most once, and no positional; the raw command may contain no `$` or
   backtick (bash *and* PowerShell expansion) and no operator token (`; | & > < ( )`).
   So no chain, redirect, `--body-file`, reviewer ping or `--web` rides along.

The scheduled preamble spells that shape out (`_grants_note`, `scheduler.py:76`) —
including "plain-text title and body, no backticks" — because a well-meant Markdown
code span in the PR body would otherwise fail the match and fall back to asking. The
pass added 36 tests, mostly bypass attempts: chains, newlines, `$(…)`, `$env:`,
backticks, `--body-file`, the wrong repo or head, `bash -c` wrapping.

> M12 mastery check: everything around the agent is deterministic and tested against
> a fake `gh`; the checkout is the workspace, so the auto-allow radius is one repo;
> issue text is fenced and bounded, and the policy is the real backstop; an existing
> checkout is never reset by our code; an empty poll spends nothing; an issue counts
> as attempted only once submitted; the PR grant is stored unbound, bound per turn,
> and matches one tokenized shape.

---

<a name="m13"></a>
## M13. Many agents, one store — memory spaces & handoff (`agents.py`, `memory/spaces.py`)

### The problem it solves

Until the platform pass, ReLife was **one agent with one memory**: every memory,
skill, workflow and event in one pool, reachable only by the in-process Claude agent.
A platform needs three things that pool couldn't give: **identity** (which agent is
this?), **isolation** (a crew reviewer's habits shouldn't leak into the user's memory,
and an agent on someone else's model shouldn't be able to rewrite it), and
**handoff** (a new agent should start from what an experienced one learned, not from
nothing). Spaces, a registry and one enforcing client answer all three.

### Spaces: a column, not a database

A **space** is a name (`validate_space`, `spaces.py:35`:
`[a-z0-9][a-z0-9_-]{0,47}`) carried on every memory and event (schema v3, M5) and
mapped to a directory for skills and workflows (`space_dir`, `:50`). `default` is the
user's main agent — everything from before spaces is there, and its procedures stayed
in `data/skills|workflows`. One SQLite file, one FTS index, one consolidation pass —
filtered by space at every layer.

Why a column and not a database per agent? Because the point is *controlled
sharing*: inheriting is a read across spaces, a fork is an `INSERT … SELECT` between
spaces (`copy_space`), promote is the same copy into `default`, and dedupe/recall stay
one implementation. Separate files would have turned each of those into a
cross-database merge.

**Semantics differ by layer, deliberately** — the part to internalize:

- **Store:** `spaces=None` means *unfiltered*. That's mechanism — consolidation and
  admin listings need every space.
- **Service/client:** an agent-facing read with no spaces means the **`default`**
  space (exactly the pre-spaces behaviour), while `all_memories`/`count` default to
  every space (they serve humans and admin).
- **An empty list matches nothing**, never everything. A bug that empties a scope must
  fail *closed*.

### `MemoryScope` — what an agent may read and write

`MemoryScope(read, write, source)` (`spaces.py:58`) is a frozen value: `read` is the
ordered tuple of spaces recall searches, `write` is the one space saves land in,
`source` is the provenance stamp. `__post_init__` validates every name and puts `write`
first in `read` (deduplicated), so an agent always reads its own memory, and a second
writable space can't be expressed.

### The registry — identity → scope (`agents.py`)

An `AgentProfile` (`agents.py:63`) has a `name` (its id), a `runtime` (`relife` = a
ReLife/Claude agent; `llm` = a CrewAI agent on a model string; `external` = anything
that connects over MCP), `model`, `description`, its own `space` (defaults to its
name), `inherits`, `isolated`, `parent`, `created_at` and a `token_hash`.
`profile.scope()` (`:84`) is pure:

- **write** = its own space — never `default` (the name `default` is rejected for an
  agent, and a record whose space is `default` doesn't load);
- **read** = own + `inherits` + `default`, unless `isolated` (own + inherits only —
  for an agent on a third-party provider that shouldn't be shown the user's personal
  memory).

`AgentStore` (`:155`) is `data/agents.json` with the `ScheduleStore` discipline (M10):
lazy load, atomic tmp + `os.replace` writes, every record validated through
`from_dict` on load *and* on `put`, unreadable records dropped with the original file
kept aside — never "repaired" into something wider. `scope_for(name)` (`:246`) raises
`LookupError` for an unknown name: **an unknown agent never falls back to the default
scope.**

### Handoff: inherit, fork, promote, packs

All four are explicit user acts (`relife agent create|attach|detach|promote`,
`relife memory export|import`), each at a different point on the live-vs-snapshot and
read-vs-write axes:

| Operation | What moves | Live? | Lands in |
|---|---|---|---|
| **inherit** (`create_agent(inherit=…)`, `attach`) | read access to the parent's space | live | nothing — read-only |
| **fork** (`create_agent(fork=…)`) | a copy of the parent's memories, skills, workflows | snapshot | the new agent's own space |
| **promote** (`promote`, `:325`) | a copy of the agent's memories (+ skills/workflows, unless `--id` picks rows) | snapshot | `default` (or `--to`) |
| **pack** (`export_space` / `import_pack`) | a `relife-memory-pack` v1 JSON file | snapshot | the space it's imported into |

The details that matter:

- **Inheritance is transitive by flattening** (`create_agent`, `:257`): inheriting from
  `veteran` adds `veteran`'s own space *and* everything `veteran` inherits, so a lineage
  reads down its whole ancestry. A fork keeps its parent's inherits too.
- **`copy_space` keeps strength and provenance** — importance, use count, embedding,
  `source` — and reinforces a text the destination already has instead of
  duplicating it. Skills and workflows are copied unless the destination already has
  that name (never overwritten).
- **Promote is the only way into `default`** besides the main agent itself and an
  explicit import. An agent can't write there; the user decides what it learned is
  worth trusting, and a promoted memory still says which agent wrote it.
- **Packs are validated in full before anything is written** (`validate_pack`), and
  imported rows are stamped `import:<origin>`; existing skills/workflows are skipped.
- **Attachments are read-only by design.** The plan had a `--write` attachment mode; it
  was dropped because a save can only land in *one* place, so a second "writable"
  space would be a write that silently goes somewhere else. Agents that should share
  writes are given the **same `--space`**.
- **`delete_agent`** (`:341`) archives the agent's space (reversibly), unless another
  agent still writes to it or `--keep-memory` is given.

### `ScopedMemoryClient` — the one enforcement point (`client.py:175`)

Everything above is policy; this class is where it's enforced, and it's enforced by
*shape*:

- Its agent-facing methods **take no space argument at all.** `save(text, kind, tags,
  importance)` writes to `scope.write` with `scope.source`; `recall`, `skill_find` and
  `workflow_find` read `scope.read`. Passing `space=` is a `TypeError`, not an
  override.
- `forget(query)` searches only `scope.write`; `archive(id)` refuses a row outside it,
  so an agent can't archive memory it inherited. `get(id)` hides rows it can't read,
  `events_for_task` filters to its space, `spaces()` lists only what it can see.
- Admin operations (`copy_space`, `export_space`, `import_pack`, `archive_space`) raise
  `PermissionError`; so does `dream` unless the scope writes `default` (REM spends the
  user's budget on the user's memory).
- It wraps *any* `MemoryClient` — Local or Http — so the same rules hold in-process,
  through the daemon, and over MCP (M14).

That is the identity rule the rest of the platform leans on: **an agent's scope comes
from who it is, never from what it — or its model — asks for.** No MCP tool schema
has a `space` property, and a test enforces it.

### Plumbing: one agent's memory through the existing seams

`memory_hooks(client)`, `memory_server(client)` and
`config.default_mcp_servers(memory_client)` each accept a scoped client; with none
they resolve the module's `default_client` at call time (unchanged for the main agent
and for the tests that monkeypatch it). That is all `--agent` (M2), a crew member
(M15) and the MCP servers (M14) need.

> M13 mastery check: a space is a column plus a directory, so sharing is a query, not a
> merge; store `None` = unfiltered, service `None` = default, `[]` = nothing; a scope
> writes one space and reads own + inherited + default (unless isolated); inherit is
> live, read-only and transitive, while fork/promote/packs are snapshots that keep
> provenance; promote is the only way into the user's memory; attachments are
> read-only because a save lands in one place; `ScopedMemoryClient` enforces by having
> no space parameter; an unknown agent never gets the default scope.

---

<a name="m14"></a>
## M14. Memory for any LLM — the MCP surface (`memory/tools.py`, `memory/mcp_server.py`)

### The problem it solves

ReLife memory had exactly one consumer: the in-process Claude agent, through an SDK
MCP server living inside the agent's own Python process. "Other LLMs should be able to
connect as agents with memory attached" needs the same memory reachable from **another
process on another model** — Cursor, Claude Desktop, Gemini CLI, a CrewAI app on
Ollama. All of those already speak MCP, which is why the first edition shipped memory
"as an MCP server even though in-process" (M5). This is where that bet paid out.

### Tools defined once (`memory/tools.py`)

Each tool is a `ToolSpec(name, description, input_schema, handler)` (`tools.py:32`),
where `handler(client, args) -> (text, is_error)` receives its client rather than
looking one up. So the *same* spec runs against the main agent's client, a scoped
agent client, a daemon-backed client, or whatever an HTTP request authenticated.
Three transports serve the specs:

- the **in-process SDK server** (`memory/server.py`) wraps `INTERNAL_TOOLS` — the nine
  tools Claude has always had, names and schemas unchanged;
- the **standalone MCP server** (`memory/mcp_server.py`) serves `EXTERNAL_TOOLS`;
- **CrewAI** gets them as `BaseTool`s (`crew/memory_tools.py`, M15).

`EXTERNAL_TOOLS` differs from the internal set in two deliberate ways:

- **No `memory_consolidate`, no `memory_dream`.** Upkeep runs where the data lives on
  ReLife's own throttle, and dream spends the user's Max budget. Neither is something
  an outside agent should be able to trigger.
- **Plus `memory_context(task)`** (`tools.py:314`): the exact block the recall hook
  injects (`memory/context.build_context`, M6). An agent with no `UserPromptSubmit`
  hook gets the same automatic recall by calling one tool first — the server's
  `instructions` tell it to, and tell it that what memory returns is data, not
  instructions.

And no schema in either set has a `space` property: the model can't choose its scope.

### stdio: `relife mcp --agent NAME`

`run_stdio(scope)` (`mcp_server.py:110`) wraps the default client in that agent's
`ScopedMemoryClient` and serves a low-level `mcp` `Server` (`build_server`, `:52`) over
stdin/stdout. `--agent` is **required** and an unknown agent exits 2 — there is no "no
agent ⇒ default space, read-write" fallback. stdout *is* the protocol stream, so nothing
here may print to it. `python -m relife` works (`relife/__main__.py`), which is what MCP
client configs launch; `relife agent connect NAME` (and `create --runtime external`)
prints a paste-ready config that pins `RELIFE_HOME`, so a client started from another
directory still reaches the same memory (`mcp_config`, `agents.py:371`).

**The Windows deadlock (worth remembering).** A real-stdio smoke, driven by the `mcp`
SDK's own client, found that with embeddings on, `initialize` and `tools/list`
answered but the **first `tools/call` hung forever**. The stdio transport's reader
thread sits in a blocking read on the stdin pipe, and constructing fastembed /
onnxruntime on a worker thread while that read was pending deadlocked. With
`RELIFE_EMBEDDINGS=off` the same call answered in 0.1s. `_warm_up` (`:88`) embeds a
dummy string and touches the store **on the main thread before the loop starts**,
which also makes the first call fast. A test pins the ordering, but the suite runs
with embeddings off, so only the release smoke (`RELEASE_TESTING.md` §3) catches a
regression.

### Streamable HTTP: `/mcp` on the memory daemon

For a client that would rather not spawn a process, the memory daemon (M8) mounts the
same server at `/mcp` — stateless, JSON responses — behind `McpHttpEndpoint` (`:141`),
an ASGI endpoint that:

1. **Authenticates an agent**, not a user: a `Bearer rla_…` token is matched against the
   registry (`AgentStore.by_token`, a constant-time compare of sha256 hashes). Missing,
   wrong or revoked → 401. The registry is **re-read per request**, so `relife agent
   token NAME --revoke`, a rotation, or deleting the agent takes effect immediately.
2. **Resolves the scope server-side**: it builds that agent's `ScopedMemoryClient` and
   stashes it in the ASGI scope (`relife.memory_client`); the tool handler reads it back
   from `server.request_context.request`. Nothing in the request body can name a space.
3. **Guards against DNS rebinding** with the MCP SDK's transport security, allowed
   hosts and origins derived from the bind address (`allowed_hosts_for`, `:131`).

Tokens are `rla_` + 32 random bytes, shown once and stored only as a hash
(`issue_token`, `agents.py:355`); a lost token is rotated, not recovered.

### Why the low-level server and not FastMCP

FastMCP derives schemas from Python signatures. Here the schemas already exist (the
`ToolSpec`s, which must stay identical to what Claude's SDK server exposes), and the
HTTP path needs the client resolved per request. The low-level `Server` takes schemas
as data and lets the handler pick the client from the request context — less magic,
one source of truth.

> M14 mastery check: tools are `ToolSpec`s with the client passed in, served three
> ways; external agents get no upkeep or dream and gain `memory_context`; no schema has
> a space; stdio requires `--agent` and never prints to stdout; on Windows the
> embedding model is warmed before serving; `/mcp` authenticates an agent token per
> request (revocation is immediate), resolves the scope server-side, and guards
> against DNS rebinding.

---

<a name="m15"></a>
## M15. Crews — CrewAI plans, ReLife staffs (`relife/crew/`)

### The problem it solves

Some tasks want a *team*: a builder and a reviewer, a researcher and a writer. CrewAI
is a mature framework for exactly that — agents with roles, tasks with dependencies,
sequential or manager-led processes — but its agents are thin LLM loops with whatever
tools they're handed. ReLife's agents are the opposite: full Claude Code agents with a
permission policy, memory and learning. `relife crew` uses each for what it's good
at: **CrewAI orchestrates; ReLife provides the members, and the memory for every
member.**

### The pipeline (`runner.run_crew`, `runner.py:71`)

```
task ─► plan (Claude, one tool-less call — or the user's --spec file) ─► validated CrewSpec
     ─► record (data/crews/<id>/record.json) ─► show plan ─► confirm
     ─► ensure_profiles (create/reuse agents — memory handed down here)
     ─► build_crew ─► Crew.kickoff() ─► per-task outcomes ─► record ─► consolidate
```

Everything that spends budget sits behind the confirmation (unless `--yes`): planning
is one call and `--plan-only` stops after it. The record is saved *before* the prompt,
so a declined plan is still on disk (`cancelled`), and the `finally` block always
stamps and saves it (`done` / `error` / `interrupted`) and runs one
`maybe_consolidate()` for the whole crew.

### The plan is untrusted output (`spec.py`, `planner.py`)

`plan_crew` (`planner.py:75`) asks Claude — through `ask_model_oneshot`, so no tools
and no API key — for a JSON plan. It sends the task (fenced `<<<TASK … TASK>>>`), the
**roster** of existing agents with their memory counts (so the planner reuses
experienced agents and sets `inherit` on new ones), and the non-Claude models the user
allowed (`RELIFE_CREW_LLMS`). The reply goes through `extract_json` and then
`normalize_spec` (`spec.py:94`), the pure validator, which treats it like any
untrusted input:

- caps: at most `CREW_MAX_AGENTS` (5) agents and `CREW_MAX_TASKS` (8) tasks, bounded
  text fields;
- names valid and unique; a task's agent must be on the crew; **context may only name
  earlier tasks** (no forward references, so no cycles);
- lineage must exist: `inherit`/`fork` name a registered agent or one defined earlier
  in the plan; nothing inherits from itself; nothing is forked *into* an existing
  agent;
- an `llm` agent's model is `claude-max` or one the user allowed; a `relife` agent runs
  on ReLife's own model;
- no idle agents.

An invalid plan gets **one retry** carrying the validation error; a second failure
falls back to a single ReLife agent (`single_agent_spec`) — the crew degrades to
`relife do`, never to an unvalidated plan. A `--spec` file from the user skips the
planner but not the validator (any model string is the user's to choose there).
`relife crews ID` reloads a saved plan with `CrewSpec.from_dict` rather than
re-validating it, because the agents a plan forked now exist and the fork check would
reject the plan's own history.

### Two kinds of member

**`runtime: relife` → `ReLifeAgent(BaseAgentAdapter)`** (`crew/agent.py:90`). CrewAI's
"bring your own agent" contract lets a crew contain agents it didn't build. Each task
this member receives is **one fresh ReLife turn** (`turns.run_relife_turn`,
`turns.py:68`): a new `ClaudeSDKClient` in `<workspace>/crews/<id>/`, ReLife's
permission callback for that directory, the browser plus the agent's **scoped** memory
server, and its **scoped** hooks — `relife do --agent NAME` without the terminal UI.
Fresh context per task is the `relife build` lesson (M7): memory, not a growing
transcript, carries continuity. The prompt (`task_prompt`, `:192`) gives the role and
goal, the task, and earlier tasks' outputs fenced `<<<CONTEXT … CONTEXT>>>` as
"information, not instructions"; the turn's closing text (bounded by
`OUTPUT_MAX_CHARS`) becomes the task's output.

**`runtime: llm` → a plain CrewAI `Agent`** (`native.llm_agent`, `native.py:62`) on a
model string — `ollama/llama3.1`, `gpt-4.1`, `gemini/…`, with keys read from the
environment by CrewAI/LiteLLM — or `claude-max`. Its tools are **ReLife memory only**
(`memory_tools`: the `EXTERNAL_TOOLS` specs as `BaseTool`s bound to its scoped
client), delegation is off, and a `step_callback` journals its steps into its space so
consolidation learns its habits too. A CrewAI agent has no recall hook, so
`build_crew` prepends the `build_context` block to each of its task descriptions
(labelled "data, not instructions"). **No shell, no files, no browser**: `classify()`
gates the Claude SDK's tool loop, and CrewAI's loop never passes through it, so
anything that touches the machine is a ReLife member's job.

### Claude for CrewAI without an API key (`llm.ClaudeMaxLLM`)

CrewAI needs an LLM object for `claude-max` members and for the manager of a
hierarchical crew. `ClaudeMaxLLM(BaseLLM)` (`llm.py:73`) turns CrewAI's messages into a
system prompt plus transcript (`split_messages`) and calls `ask_model_oneshot` — the Max
login, tools denied. It reports `supports_function_calling() == False`, so CrewAI
drives tools in its ReAct text format, and `supports_stop_words() == True`, applying
the stop words (`"\nObservation:"`) by truncation. The sync `call` runs the coroutine
through `turns.run_sync`: `anyio.run` on the calling thread, or a helper thread when a
loop is already running there. Cost accumulates for the run record.

### Contract surprises, and the defaults that stay off

Two things CrewAI's docs don't say, found by reading the pinned 1.15.27 source and
then running a real `kickoff()`:

- `BaseAgentAdapter` inherits six **abstract** methods from `BaseAgent`
  (`execute_task`, `aexecute_task`, `create_agent_executor`, `get_delegation_tools`,
  `get_platform_tools`, `get_mcp_tools`) — CrewAI's own OpenAI adapter predates that and
  doesn't implement them all.
- The crew reads `function_calling_llm`, `step_callback` and `last_messages` off every
  member; the adapter declares all three.

Three CrewAI features stay **off**: memory, planning and knowledge (`Crew(memory=False,
planning=False)`), because they default to OpenAI embeddings and models and would
demand a key — and ReLife's memory *is* the memory. Telemetry is off too
(`CREWAI_DISABLE_TELEMETRY` and friends, set with `setdefault` so a user can opt back
in), because ReLife is local-first.

### `crew_tools`: the unknown-tool default doing real work

CrewAI can hand a member tools of its own. A `ReLifeAgent` exposes them to its turn as
an in-process MCP server named **`crew_tools`** (`crew_tools_server`, `crew/agent.py:52`) —
deliberately *not* `relife_*` — so each call surfaces as `mcp__crew_tools__…`, misses
the trusted prefix, and lands in `classify()`'s fail-closed branch: **ask**. Tools
ReLife hasn't vetted don't become autonomous by arriving with a crew.

### Where memory flows

`ensure_profiles` (`build.py:32`) is the handoff point: a planned agent that doesn't
exist is created with the plan's `inherit`/`fork` (M13); one that exists is reused and
any planned inherits are attached. Every member gets `ScopedMemoryClient(client,
profile.scope())`, so a crew of three writes into three spaces, reads what each
inherited plus the user's default, and never writes `default`. Afterwards `relife
memory search --space NAME` shows what each learned, and `relife agent promote NAME`
is the user's call.

### Python 3.14

CrewAI 1.15 requires Python < 3.14, and ReLife on this machine runs on 3.14. The
`[crewai]` extra carries the marker `python_version < '3.14'`, so it installs nothing
there; the crew modules that import CrewAI (`llm`, `agent`, `native`, `memory_tools`,
`build`) are imported only after `_crewai_or_exit()` gives a one-line fix, while
`spec`, `planner`, `record`, `turns` and `runner` are CrewAI-free and tested
everywhere. Crews run from a 3.12 `.venv`, where `tests/test_crew.py`
(`importorskip("crewai")`) runs a real `Crew.kickoff()` — a ReLife member on a fake
turn runner and a CrewAI member on `ClaudeMaxLLM` with a stub model — with zero model
calls.

> M15 mastery check: CrewAI orchestrates, ReLife provides the members and the memory;
> the plan is untrusted and validated (caps, backward-only context, known lineage), with
> one retry and then a single-agent fallback; a ReLife member is one fresh, scoped
> ReLife turn per task with context fenced; a CrewAI member gets memory but no
> machine-touching tools, because `classify()` can't gate CrewAI's loop; Claude reaches
> CrewAI only through the CLI; CrewAI memory/planning/telemetry stay off; `crew_tools`
> falls to ask on purpose; handoff happens in `ensure_profiles`.

---

<a name="m16"></a>
## M16. Trade-offs, failure modes & "why not X"

This module is the synthesis: the recurring design *principles*, the deliberate
*sacrifices*, and the *weak points*.

### The recurring principles (the "house style")

1. **Pure core, thin imperative shell.** The hard logic is pure and deterministic —
   `cognitive.py`, `classify()` + `grant_allows`, `ledger.py`, the hook callbacks,
   REM's `_apply`, `security.py`, `schedules.py`, `runs.summarize_events`,
   `doctor.run_checks`, the `workitems` plumbing, `MemoryScope`/`AgentProfile.scope`,
   `crew/spec.normalize_spec` — and tested without a live agent, a socket or a model
   (a real CrewAI `kickoff()` included). The SDK loop, FastAPI routes and
   uvicorn are thin wrappers around injected pieces. *Trade-off:* more indirection
   and more seams to understand, in exchange for a system you can test cheaply and
   reason about — essential when the expensive path (live runs) burns Max budget.

2. **Soft-optional dependencies, never hard.** FTS5, embeddings (`fastembed`), the
   ANN extension (`sqlite-vec`), the `[daemon]`, `[server]` and `[crewai]` extras all
   degrade or opt in gracefully (`[crewai]` is marker-gated off Python 3.14, and the
   modules that import it load only behind `_crewai_or_exit()`). The pattern is always: feature-detect or self-test → use if it
   works → silently fall back to the always-correct path otherwise. *Trade-off:* the
   fast path isn't guaranteed, and there are more code paths; in exchange the system
   always runs correctly on a bare install and can never be *broken* by an optional
   component.

3. **Reversibility over destruction.** Forgetting is two-tier (archive, then much-later
   delete). REM's only destructive action is archive, journaled. `relife memory
   forget` archives. Deleting an agent archives its space; fork and promote *copy*,
   never move. *Trade-off:* the store carries archived dead weight longer;
   in exchange a bad decision (a faded-but-seasonal memory, a wrong critic verdict, a
   mistaken human) is recoverable.

4. **Budget-awareness as a first principle (the Max constraint).** Anything automatic
   and frequent is LLM-free; the one LLM memory path is opt-in, capped, and gated. No
   hosted embeddings. Schedules are floored at five minutes. An empty work poll is
   skipped before a session exists. A crew's plan is shown and confirmed before it
   runs, and Claude inside CrewAI goes through the CLI. *Trade-off:* the cheap
   deterministic passes can't do qualitative judgement; that capability is quarantined
   into the opt-in REM pass.

5. **Fail-closed for safety, fail-soft for data.** Permissions, the recall gate,
   `guard_bind` and `resolve_workspace` fail *closed* (deny / surface nothing / refuse)
   because their failure mode is *wrong action*. Journaling, episode capture, the
   reaper, the scheduler tick and the run recorder fail *soft* (`except: pass`, or
   record-and-advance) because their failure mode is *lost data* or *a missed slot*,
   which is recoverable and must never break a task or the server.

6. **One rule, many surfaces.** "No answer ⇒ deny" is the same rule at the TTY, in
   the browser card, and in an unattended schedule. `classify()` is the single policy
   source behind both callbacks. `to_event` is the single taxonomy behind both
   renderers. A scheduled run is *a turn*, not a mode. One `ToolSpec` per memory tool
   serves Claude, external MCP clients and CrewAI; one `build_context` serves the
   recall hook and the `memory_context` tool; a crew member is `relife do --agent` in
   a different harness. Every time the system grew a surface, the design reused the
   rule instead of cloning it.

7. **Seams that were paid for, then cashed in.** Memory was an MCP server even
   in-process; `MemoryService`/`MemoryClient` were no-op indirections. When the
   daemon split came, no consumer changed. It paid again for the platform: an agent's
   scope is just a *wrapper* around whichever client is in play, and other LLMs attach
   to the same tools Claude uses without Claude's contract changing. The extra layers
   were the price; the painless migrations were the return.

8. **Decide where the data lives.** The auto-consolidate throttle runs on the memory
   side; the daemon binds module globals so upkeep hits the same DB; the scheduler
   advances from *now* in the process that fires. Whenever a decision was split
   across a boundary it silently broke (`85cd67c`). Keep the decision next to the
   data it reads.

9. **Identity decides scope, never the model.** No memory tool schema has a space;
   `ScopedMemoryClient` has no space parameter; `/mcp` resolves the scope from the
   bearer token server-side; an unknown agent exits rather than falling back to the
   default; grants are bound to a turn by the scheduler, not named by the agent. A
   model can be prompt-injected — so it never holds the key to its own permissions.

10. **Third-party text is fenced data.** Issue bodies (`<<<ISSUE`), crew tasks
   (`<<<TASK`), earlier crew output (`<<<CONTEXT`), inherited memory (`via <space>`),
   the MCP server's own instructions — each says "information, not instructions".
   The fence is a hint; the permission policy is the backstop, and is written as if
   the fence will sometimes fail.

### The sharpest trade-offs (state these crisply)

- **LLM-free consolidation buys affordability at the cost of intelligence.** The
  constant pass can only do *mechanical* maintenance. It cannot tell that a memory is
  wrong or contradictory. That intelligence exists only in the opt-in REM pass — so
  between dreams, the store can hold contradictions and garbage that only a
  human-triggered pass (or `relife memory forget`) will catch.
- **Reinforcement-on-recall makes useful memories self-strengthening — and depends
  entirely on the relevance gate to not become a popularity contest.** The safety of the
  whole feedback loop rests on M5's Stage-1 gate being correct. Weaken that gate and the
  reinforcement loop turns pathological.
- **Subagent delegation keeps the orchestrator small at the cost of cross-milestone
  context.** A builder sees only its one milestone + what already exists in the
  workspace. Parallel milestones are deliberately deferred for the same reason.
- **One event loop buys a simple concurrency story at the cost of one bad citizen
  freezing everyone.** The daemon and the server both serialize on a single loop —
  no locks, clean approval futures — but anything slow *on* the loop stalls every
  session (the consolidation freeze of MVP pass 4). The rule that fell out: CPU/IO
  work goes to a thread; only coordination stays on the loop.
- **A scheduled run being "just a turn" reuses everything — and inherits the
  session ceiling and skip-when-busy.** A long task can starve its own next slot.
  That's recorded, not hidden, and preferred over a second agent path that could
  drift from the interactive one.
- **Cookie auth makes the UI possible and makes CSRF a real concern.** Two guards
  (`SameSite=Strict` + the Origin check) instead of one, and a rule that neither can
  be "simplified" away.
- **Grants let unattended runs finish — and are a real widening of the policy.** They
  are kept narrow by construction (connector-only plus one bound PR shape, listed
  addresses, a per-run use cap, every use journaled), but each new kind is a new
  attack surface and has to be tested like one (the PR grant shipped with 36 bypass
  tests; two email bypasses were still found against the live schemas).
- **Read-only attachments make "where did that save go?" trivially answerable — at
  the cost that agents sharing writes must share a space.** Two agents on one space
  can't be told apart in it except by `source`.
- **Non-Claude agents get memory but no hands.** A crew member on Ollama can reason,
  recall and remember, but every file edit, test run and push goes to a ReLife member,
  because ReLife's permission policy can't see inside CrewAI's tool loop. Less capable
  crews, in exchange for one policy covering everything that touches the machine.
- **A fresh SDK client per crew task buys clean context at the cost of startup time
  and no shared transcript.** Continuity between a member's tasks is whatever it saved
  to memory and what CrewAI passes as context — the `relife build` trade, again.

### The biggest risks / weak points

1. **Deterministic pattern mining can mint junk into long-term memory.** The episode/
   n-gram detectors are heuristic. The `EPISODE_MIN_EVENTS` floor and
   `_is_meaningful_seq` gate mitigate but don't eliminate a spurious `pattern`/workflow
   that then gets surfaced by recall. Mitigation now exists downstream too: `relife
   memory list --kind pattern` and `forget`.
2. **A wrong memory that keeps matching queries gets *reinforced*, not corrected.**
   Between dreams the only corrective is a human with `relife memory search`/`forget`
   — which is why the inspection CLI never reinforces.
3. **Shell containment is heuristic.** `_write_targets`/`_delete_targets` close the
   common escapes and err toward asking, but shell grammar is not a regex. A
   sufficiently creative command (`eval`, a heredoc, `python -c "open(...)"`) is
   not caught by path extraction — it's caught only if it matches an outward pattern.
   The defense in depth is the workspace itself being a throwaway directory.
4. **The agent server has no TLS and no multi-user model.** Loopback + optional token
   is the supported shape; anything further needs a reverse proxy and is explicitly
   unsupported today.
5. **Live verification is expensive and rationed.** Because live runs consume Max
   budget, the team relies on the ~800 deterministic tests and avoids hammering live
   runs. Some end-to-end behaviors are validated less often than unit-level logic; at
   the time of writing, the live crew smokes (`relife crew --plan-only`, a two-agent
   crew) have **not** been run.
6. **Windows-specific sharp edges.** Console encoding must be forced to UTF-8
   (`agent.py:40`) or Rich crashes on glyphs under cp1252; two shell tools must be
   gated identically; `gh` PATH injection is a workaround for a mid-session install;
   Git Bash path spellings (`/d/…`) needed their own translation; the stdio MCP server
   deadlocked until the embedding model was warmed first. Handled, but the kind of
   environment coupling that can resurface.
7. **CrewAI is a fast-moving dependency.** Pinned `>=1.15,<2` and unavailable on Python
   3.14; its adapter contract already moved under its own adapters once (M15). The real
   `kickoff()` test is the tripwire, and it only runs where CrewAI installs.
8. **Memory written by other models enters the store.** It's confined to their own
   spaces, labelled `via <space>` when shown to anyone else, and never reaches `default`
   without `promote`. But an agent that inherits from an external agent does read what
   that agent wrote — inheritance is explicit, and it is a trust decision.

### "Why not X" — alternatives consciously rejected

- **Why not a metered API key?** It would untie ReLife from the Max subscription and
  add per-token cost, defeating the central bet and removing the constraint that makes
  the design coherent. `doctor` warns if one is set.
- **Why not LLM-driven consolidation?** It auto-runs on the shared Max budget;
  productivity would accelerate budget drain and stall long sessions (M6). Quarantined
  into opt-in REM instead.
- **Why not hosted embeddings?** No API key available; semantic recall uses a local
  offline ONNX model that degrades to keyword+activation if even *that* is absent.
- **Why not delete faded memories directly?** Irreversible; destroys cyclical/seasonal
  memories. Two-tier archive→delete makes the first step recoverable.
- **Why not let the REM model archive/edit memory directly?** It's fallible; a bad or
  prompt-injected verdict could corrupt the store. So it's advisor-only — verdicts
  applied deterministically, reversibly, capped, gated, journaled; text edits forbidden.
- **Why not an allowlist for shell commands?** The safe set is infinite and the
  dangerous set finite — you can only define the policy over the finite side.
- **Why not an enumerated list of connector tool names?** The names are Google's and
  can change; the read/write verb is the stable signal, and "neither verb ⇒ ask" keeps
  it fail-closed.
- **Why not inject a store into the daemon's service?** `consolidate()`/`dream()` mine
  the module defaults and write via module functions; an injected store would leave
  upkeep on the wrong DB, and routing upkeep through the client would deadlock the
  daemon's own loop (M8).
- **Why not header-only auth on the server?** `EventSource` can't set headers, so the
  transcript + approval stream would be unauthenticatable from the UI (M9).
- **Why not accept any workspace path in `POST /sessions`?** The caller would be
  choosing the auto-allow blast radius (M9).
- **Why not a separate job runner for schedules?** It would be a second agent path
  that could drift from the interactive one in policy, journaling and learning; a
  firing is a turn instead (M10).
- **Why not parallel milestones in build?** Coordinating writes/decisions across fresh,
  independent contexts is hard and error-prone; deferred deliberately.
- **Why not let an agent pass a `space`?** Then the scope is whatever the model —
  possibly prompt-injected — asks for. Scope comes from identity (M13).
- **Why not a database per agent?** Inherit, fork, promote and dedupe would all become
  cross-database merges; a `space` column keeps them single queries (M13).
- **Why not writable attachments?** A save can only land in one place; a second
  writable space is a write that goes somewhere unexpected. Share a `--space` instead.
- **Why not give CrewAI agents the shell?** `classify()` gates the Claude SDK's loop,
  not CrewAI's. Machine-touching work goes to ReLife members (M15).
- **Why not CrewAI's own memory and planning?** They default to OpenAI embeddings and
  models and would demand a key, and they'd be a second memory beside ReLife's.
- **Why not `anthropic/claude-…` through LiteLLM for Claude members?** That needs
  `ANTHROPIC_API_KEY` and metered billing — exactly what M1 rules out. `ClaudeMaxLLM`
  goes through the logged-in CLI.
- **Why not spawn `relife mcp` for each CrewAI agent inside a ReLife crew?** The same
  specs as in-process `BaseTool`s give the same scoping with no subprocess per agent;
  stdio and `/mcp` are for agents that live outside ReLife.
- **Why not reset an existing checkout in `relife work`?** It may hold the user's
  uncommitted work. Syncing is the agent's first, visible step (M12).

---

<a name="appendix-a"></a>
## Appendix A. The complete request lifecycle (three traces)

### Trace 1 — `relife do "add a /health endpoint and push"` (interactive)

1. **CLI** (`cli.py:do`) resolves the workspace, builds `can_use_tool` bound to it,
   gets `default_mcp_servers()` (browser + memory) and `memory_hooks()`, and calls
   `run_task`.
2. **`build_options`** (`agent.py:67`) assembles `ClaudeAgentOptions`: model, effort,
   preset+persona system prompt, the permission callback, MCP servers, hooks,
   `setting_sources=None`, `env` with `gh` on PATH.
3. **Streaming client** starts (`agent.py:356`). `client.query(prompt)` sends the task.
4. **`UserPromptSubmit` → `_recall_hook`** fires first: stashes the prompt by session,
   recalls top memories/skills/workflows through `default_client()` (reinforcing
   them), de-dups + budget-caps, and **injects** the surviving context.
5. **The model acts.** Each tool call hits **`classify()`**: `Read`/`Grep`/`Edit`
   (inside workspace) → allow; `Bash "pytest"` → allow; `Bash "git push"` → allow
   (authorized). `Bash "echo x > ~/.bashrc"` → **ask** (write target outside the
   workspace). `gh pr create` → **ask** (or deny if non-interactive).
6. **`PostToolUse` → `_event_hook`** journals each call (tool + brief + session id)
   via the client. The `git push` is recorded as action `git-push`.
7. The model may call **`memory_save`/`skill_write`** explicitly (reflect).
8. **`Stop` → `_episode_hook`** pops the prompt, sees ≥3 events, collapses the tool
   sequence, and saves `"Task: add a /health endpoint and push | Approach: Read → Edit
   → test → git-push"` as an episode.
9. **`_maybe_consolidate`** (`agent.py:259`) asks the memory side to decide-and-run:
   if enough events accrued it fades unused memories (two-tier), merges duplicates,
   and mines n-grams — if `Read → Edit → test → git-push` has now recurred ≥3 times,
   it synthesizes a workflow that **future `_recall_hook` calls will surface**. The
   loop has closed.
10. **Later, manually:** `relife dream` runs REM — the model reviews recent memories as
    an adversarial critic and reversibly prunes/reweights, journaling every action.
    Or: `relife memory search "health endpoint"` shows what would be recalled,
    without reinforcing it.

### Trace 2 — a schedule "inbox triage, weekdays 09:00" fires with nobody watching

1. **`relife serve`** is running: `create_app()` built the app, the lifespan started the
   reaper and the `Scheduler` tick loop.
2. At 09:00:07 a tick finds the schedule due (`enabled and next_run_at <= now`). Its
   session was reaped overnight, so `_submit` **creates one** in the schedule's
   confined workspace (`resolve_workspace`) — M2's three pieces with the UI approval
   callback — and stores the new `session_id` on the schedule.
3. The scheduler **subscribes** to the session, then submits `scheduled_prompt()`
   (preamble + task). The schedule's `next_run_at` advances to tomorrow 09:00
   (local, DST-safe); history gets a `submitted` entry.
4. The worker dequeues the turn and **echoes it as a `user` event** — the recorder sees
   the echo whose text equals its prompt and starts collecting.
5. `_recall_hook` injects what ReLife knows about this inbox; the agent calls
   `mcp__claude_ai_Gmail__search_messages` → `classify()` sees a **read verb → allow**.
6. It decides to reply to one thread: `mcp__claude_ai_Gmail__send_message` → **write
   verb → ask**. `make_approval_callback` → `broker.request()` publishes an
   `approval_request` (tool + `to=… subject=…` brief) and awaits the future. Nobody is
   watching. After `AGENT_APPROVAL_TIMEOUT` (300s) → `False` → **deny**;
   `approval_resolved{approved: false}` is published.
7. The preamble told the agent not to fight a denial; it finishes with a closing
   summary ("Triaged 14 messages; one reply needed your approval and was denied —
   see below"). `to_event` yields `result`; the recorder stops.
8. `_finish` writes `data/runs/<schedule>/<run_id>.json` with `status=done`, the
   summary, tool count, cost, and `denied=[{tool: …send_message, brief: "to=… subject=…"}]`,
   and upgrades the history entry from `submitted`. Meanwhile the `Stop` hook saved
   the episode and `maybe_consolidate_off_loop()` ran the sleep beat on a thread.
9. You open the console: the card shows the last outcome; **runs** shows the summary
   and the "needed you — denied unattended" block with the exact email. Or you run
   `relife doctor` and its `schedules` line says `needed you: inbox triage (1 denied)`.

### Trace 3 — `relife crew "add CSV export to the todo CLI and review it"`

1. **CLI** (`cli.py`, `crew`) calls `_crewai_or_exit()` (on Python 3.14 it prints the
   venv recipe and exits), then `run_crew`.
2. **Plan.** `roster()` lists registered agents with memory counts — say `builder`
   with 340 memories. `plan_crew` makes one tool-less `ask_model_oneshot` call; the
   model proposes `builder` (reuse, `relife`) for an `implement` task and a new
   `reviewer` (`llm: claude-max`, `inherit: [builder]`) for a `review` task with
   `context: [implement]`. `normalize_spec` accepts it.
3. **Record and confirm.** `data/crews/<id>/record.json` is written as `planned`; the
   plan is printed (agents, engines, lineage, tasks) and the user says yes.
4. **Profiles.** `ensure_profiles` reuses `builder` and creates `reviewer` with
   `inherits = [builder, …whatever builder inherits]` — the memory handoff.
5. **Assembly.** `build_crew` gives each member `ScopedMemoryClient(client, scope)`.
   `builder` becomes a `ReLifeAgent` working in `workspace/crews/<id>/`; `reviewer` a
   CrewAI `Agent` on `ClaudeMaxLLM` with ReLife memory tools. The review task's
   description is prefixed with `build_context(reviewer, …)`, which can surface
   `builder`'s memories labelled `via builder`.
6. **Task 1.** CrewAI calls `ReLifeAgent.execute_task` → `run_relife_turn`: a fresh
   `ClaudeSDKClient` with the TTY permission callback, the scoped memory server and the
   scoped hooks. Recall injects `builder`'s memory; edits and tests inside the crew
   workspace run on their own; anything outward asks in the terminal. Events and the
   episode land in space `builder`. The closing text is the task's output.
7. **Task 2.** CrewAI hands the reviewer the implement output as context;
   `ClaudeMaxLLM.call` → `ask_model_oneshot` (no tools on the Claude side). CrewAI's
   ReAct loop may call `memory_recall`/`memory_save` — the save lands in space
   `reviewer`, source `reviewer`.
8. **Outcome.** The record becomes `done` with per-task outputs, tool counts, cost and
   any denials; `maybe_consolidate()` runs once, per space, so `builder`'s sweep never
   touches `reviewer`'s memory. `relife crews <id>` shows it all, and `relife agent
   promote reviewer` would copy the reviewer's lessons into `default` if the user
   trusts them.

---

<a name="appendix-b"></a>
## Appendix B. File-by-file index

| File | Responsibility |
|---|---|
| `cli.py` | Typer CLI: `do`/`chat` (`--agent`)/`work`/`build`/`serve`/`doctor`/`consolidate`/`dream` + `memory {search,list,show,forget,stats,serve,ping,spaces,export,import}` + `agent {list,create,show,attach,detach,promote,token,connect,delete}` + `mcp` + `crew`/`crews`; wires permissions + MCP + hooks. |
| `__main__.py` | `python -m relife` (what MCP client configs launch). |
| `agents.py` | `AgentProfile` (identity → `MemoryScope`), `AgentStore` (`data/agents.json`), `create_agent` (inherit/fork), `attach`/`detach`/`promote`/`delete_agent`, hashed `rla_` tokens, `mcp_config`. |
| `workitems.py` | `relife work` plumbing: `parse_ref`, `list_assigned`, `fetch`, `ensure_checkout`, `branch_name`, `task_prompt` (issue text fenced). |
| `agent.py` | `build_options`, `run_task`/`run_chat`, `to_event` (shared taxonomy), `_render`, `ask_model_oneshot`, `maybe_consolidate` (+ `_off_loop`), `_tool_brief`, Windows UTF-8 fix. |
| `permissions.py` | `classify()` pure policy; nested-shell re-classification; `_OUTWARD_SHELL` denylist (both shells); `_global_install`; `_gh_outward` (verb-based `gh`); `_write_targets`/`_delete_targets`/`_escapes` shell containment (+ `_msys_to_windows`); `_under()`; connector verb policy; grants (`normalize_grants`, `grant_allows`, `_pr_create_matches`); `make_permission_callback` (TTY) + `make_approval_callback` (UI, `preauthorize`). |
| `hooks.py` | `_make_hooks(get_client)` → recall (inject+reinforce via `build_context`), event (journal), episode (capture); module trio over `default_client()`; `memory_hooks(client)` for a scoped agent; every call `off_loop`. |
| `doctor.py` | `run_checks(Probes)` pure check matrix (CLI/login/api-key/model/node/gh/FTS5/dirs/extras/daemon/agent server/schedules/connectors/crews/agents), `default_probes()`, `to_json`. |
| `config.py` | All paths (`RELIFE_HOME`), model id, every cognitive/recall/REM/daemon/server/scheduler/grant/crew tunable, `default_mcp_servers(memory_client)`, `agent_env`. |
| `web/index.html` | Self-contained console: auth prompt, transcript via `EventSource`, approval cards, reattach via `localStorage` + `?last_id=0`, schedules panel with watch/run/runs, work toggle and pre-approvals. |
| `memory/cognitive.py` | Pure ACT-R math: `activation`, `sigmoid`, `fused_score`, `should_archive`, `should_hard_delete`. |
| `memory/store.py` | `MemoryStore`: schema v3 (spaces)/migrations, two-stage space-filtered `recall`, reinforce-on-`save` (per space), candidate + common-term gate, `archive`/`get`, `copy_space`/`archive_space`/`space_counts`; module shims over `_DB_PATH`. |
| `memory/vector_index.py` | `VectorIndex` protocol, `BruteForceIndex`, `SqliteVecIndex`, `get_index` self-test. |
| `memory/embeddings.py` | Soft-optional local ONNX embeddings; `available`/`embed`/`cosine`. |
| `memory/service.py` | `MemoryService` facade: memory + `archive(id)`/`get(id)` + skills + workflows + events + spaces (`spaces`/`copy_space`/`archive_space`/`export_space`/`import_pack`) + `maybe_consolidate` (throttle decided here) + `dream`. |
| `memory/client.py` | `MemoryClient` protocol, `LocalMemoryClient`, `ScopedMemoryClient` (the scope enforcement point), `default_client()` (env-selected transport), `off_loop`, daemon-sidecar warning. |
| `memory/spaces.py` | `DEFAULT_SPACE`, `validate_space`, `space_dir`, `MemoryScope(read, write, source)`. |
| `memory/remote/wire.py` | Dependency-free dict ⇄ dataclass for every type that crosses the HTTP seam. |
| `memory/remote/daemon.py` | FastAPI daemon: `_bind_db`/`_bind_dirs` (module-global binding, spaces dir included), routes mirroring the protocol (incl. `/spaces/*`), `/mcp` + its lifespan, sidecar file, `serve()`. |
| `memory/remote/http_client.py` | `HttpMemoryClient`: pooled httpx, rebuilds real objects, 400→`ValueError`, `dream` off-thread with no timeout. |
| `memory/skills.py` | Single procedures as Markdown, per space; weighted keyword `find_skills` (earlier space shadows). |
| `memory/workflows.py` | Multi-step procedures as Markdown (+`trigger`), per space; weighted keyword `find_workflows`. |
| `memory/events.py` | `EventLog`: append-only tool journal (with `space`), `events_by_task`, `count`. |
| `memory/consolidate.py` | Deterministic "sleep", per space: decay/archive/delete, dedupe, n-gram mining, workflow synthesis, `should_auto_run`. |
| `memory/rem.py` | Opt-in LLM "dream": replay buffer, critic prompt, deterministic reversible `_apply`, journal. |
| `memory/tools.py` | `ToolSpec` + the memory tools defined once; `INTERNAL_TOOLS` (Claude's nine) and `EXTERNAL_TOOLS` (no consolidate/dream, + `memory_context`). |
| `memory/context.py` | `build_context(client, query)`: the deduped, budgeted recall block with `via <space>` provenance. |
| `memory/server.py` | MCP `relife_memory` (in-process SDK server): `INTERNAL_TOOLS` bound to a client (`memory_server(client)`). |
| `memory/mcp_server.py` | Standalone MCP server: `build_server`, `run_stdio` (+ `_warm_up`), `McpHttpEndpoint` (agent bearer token → scoped client), `http_session_manager`, `allowed_hosts_for`. |
| `memory/_text.py` | Shared stopword tokenizer used by every keyword path. |
| `build/ledger.py` | `BuildLedger`: durable plan/progress, atomic writes, `latest_for`, `status_brief`. |
| `build/server.py` | MCP `relife_build`: `build_plan_set`/`build_milestone_update`/`build_status`, ledger-bound. |
| `build/agents.py` | `builder` `AgentDefinition` + tool list + orchestrator prompt path. |
| `build/orchestrator.py` | `run_build`: fresh vs resume, ledger wiring, session persistence + expired-session fallback. |
| `server/app.py` | `create_app()` (auth/cookie/CSRF wiring, sessions, SSE, approvals, schedules, runs routes, lifespan), `serve()` with `guard_bind`. |
| `server/session.py` | `ApprovalBroker` (future + timeout), `AgentSession` (one `ClaudeSDKClient`, ring buffer, subscribers, `busy`, per-turn grants + use cap), `SessionManager` (ceiling, idle reaper). |
| `server/security.py` | Pure: `token_matches`, `presented_token`, `same_origin`, `is_loopback`, `guard_bind`, `resolve_workspace`, `AttemptLimiter`. |
| `server/schedules.py` | `Schedule` record (grants, `work`, `worked`), `parse_every`/`parse_at`/`parse_days`/`normalize_spec`/`next_run` (explicit `now`), `normalize_work`, `check_grants_fit`, `ScheduleStore` (atomic JSON, corrupt file preserved). |
| `server/scheduler.py` | `Scheduler`: tick loop, `fire`/`_submit` (per-schedule session, skip-when-busy, advance-from-now), `_prepare_work` + `pick_issue` (work schedules), `bind_grants`, `_record` recorder, `scheduled_prompt`. |
| `server/runs.py` | `summarize_events` (closing summary + denied + acted lists), `RunRecord`, `RunStore` (bounded per-schedule files). |
| `crew/spec.py` | `AgentSpec`/`TaskSpec`/`CrewSpec`; `normalize_spec` (the plan's validator); `load_spec_file`; `single_agent_spec`. |
| `crew/planner.py` | `plan_crew` (one tool-less call, one retry, fallback), `extract_json`, `planner_prompt` (task fenced). |
| `crew/turns.py` | `run_relife_turn` (one fresh scoped ReLife turn), `TurnResult`, `result_from_events`, `run_sync`. |
| `crew/agent.py` | `ReLifeAgent(BaseAgentAdapter)`, `crew_tools_server` (outside the trusted prefix), `task_prompt` (context fenced). |
| `crew/llm.py` | `ClaudeMaxLLM(BaseLLM)`: Claude via `ask_model_oneshot`, ReAct text, stop words by truncation. |
| `crew/native.py` | `default_llm`, `llm_agent` (CrewAI `Agent` with ReLife memory tools, no delegation), `journal_step`. |
| `crew/memory_tools.py` | `RelifeMemoryTool(BaseTool)` over a `ToolSpec`; `memory_tools(client)`. |
| `crew/build.py` | `ensure_profiles` (the handoff point), `build_crew` (scoped members, recall block for CrewAI tasks, CrewAI memory/planning off). |
| `crew/runner.py` | `run_crew` (plan → record → confirm → kickoff → outcomes → consolidate), `roster`, `describe_plan`, telemetry off. |
| `crew/record.py` | `TaskOutcome`, `CrewRunRecord`, `CrewRunStore` (`data/crews/<id>/record.json`, id-shape checked). |

---

<a name="appendix-c"></a>
## Appendix C. Every tunable in `config.py`

**Model:** `MODEL = claude-opus-4-8` (`RELIFE_MODEL`), `EFFORT = high` (`RELIFE_EFFORT`).

**Paths:** `RELIFE_HOME` decides where `data/` and `workspace/` live (default: beside the
code in a source checkout, `~/.relife` for an installed wheel — never `site-packages`).
Under `data/`: `relife.db`, `skills/`, `workflows/`, `spaces/<space>/`, `agents.json`
(`AGENTS_PATH`), `crews/` (`CREWS_DIR`), `builds/`, `schedules.json`, `runs/`.

**Activation / forgetting:**
- `DECAY = 0.35` — recency forgetting rate.
- `IMPORTANCE_BOOST = 1.5` — how strongly importance lifts activation.
- `FORGET_THRESHOLD = 0.20` — archive below this activation…
- `MIN_FORGET_AGE_DAYS = 14` — …and idle at least this long…
- `PIN_THRESHOLD = 0.80` — …and importance under this (≥ is pinned, never archived).
- `HARD_DELETE_AGE_DAYS = 90` — archived + idle this long → permanent delete.
- `DEFAULT_IMPORTANCE` — preference 0.7, pattern 0.65, fact 0.5, episode 0.45.

**Fused recall:**
- `W_SEM 0.45 / W_KW 0.30 / W_ACT 0.15 / W_IMP 0.10` — the four-signal weights.
- `KIND_RECALL_BOOST` — preference +0.05, pattern +0.02.
- `RECALL_FLOOR = 0.12` — absolute relevance floor in Stage 2.
- `CANDIDATE_TOPN = 50` — max Stage-1 candidates.
- `SEM_CANDIDATE_THRESHOLD = 0.65` — min cosine for a zero-keyword row to be a candidate.
- `RECALL_COMMON_TERM_FRACTION = 0.2`, `RECALL_COMMON_MIN_DOCS = 3` — a one-word match
  counts only if the word is in at most this share of active memories.

**De-dup:** `DEDUP_SIM = 0.90` (consolidation merge), `SAVE_DEDUP_SIM = 0.93` (save-time
paraphrase reinforcement).

**Injection (hooks):** `RECALL_INJECT_BUDGET = 2400` chars, `RECALL_DEDUP_JACCARD = 0.8`.

**Episodes / consolidation:** `EPISODE_MIN_EVENTS = 3`, `RECUR_THRESHOLD = 3`,
`AUTO_CONSOLIDATE = on`, `CONSOLIDATE_EVERY = 5`.

**REM:** `REM_BATCH_MAX = 40`, `REM_REFERENCE_MAX = 30`, `REM_MIN_CONFIDENCE = 0.7`,
`REM_MAX_PRUNE_FRACTION = 0.25`. (No AUTO flag — never auto-runs by design.)

**Embeddings:** `EMBED_MODEL = BAAI/bge-small-en-v1.5`, `EMBEDDINGS_ENABLED = auto`
(auto|on|off).

**Memory daemon (opt-in):** `MEMORY_URL` (`RELIFE_MEMORY_URL`, unset = in-process),
`MEMORY_TOKEN`, `MEMORY_HOST = 127.0.0.1`, `MEMORY_PORT = 8787`,
`MEMORY_DB_PATH = data/relife.db`, `MEMORY_SIDECAR_PATH = data/relife.db.daemon`.

**Agent server:** `AGENT_HOST = 127.0.0.1`, `AGENT_PORT = 8600`, `AGENT_TOKEN`
(required for any non-loopback bind), `AGENT_APPROVAL_TIMEOUT = 300s` (no click →
deny), `AGENT_COOKIE` / `AGENT_COOKIE_MAX_AGE = 30d`, `AGENT_AUTH_MAX_ATTEMPTS = 10`
per `AGENT_AUTH_WINDOW = 60s`; `AGENT_ALLOWED_HOSTS` (`RELIFE_AGENT_ALLOWED_HOSTS`) —
extra Host names a tokenless server answers to (DNS-rebinding guard).
- Ceilings: `AGENT_MAX_SESSIONS = 8`, `AGENT_SESSION_IDLE_TIMEOUT = 3600s`,
  `AGENT_REAP_INTERVAL = 60s`, `AGENT_MAX_MESSAGE_CHARS = 16000`,
  `AGENT_MAX_QUEUED_TURNS = 8`, `AGENT_MAX_SUBSCRIBERS = 8`.
- `AGENT_WORKSPACE_ROOT = ./workspace` — containment root for server-created sessions.

**Scheduler / runs:** `AGENT_SCHEDULER = on` (`RELIFE_AGENT_SCHEDULER=0` disables),
`AGENT_SCHEDULER_TICK = 30s`, `AGENT_MAX_SCHEDULES = 32`,
`AGENT_SCHEDULE_MIN_INTERVAL = 300s` (floor on the `every` form),
`AGENT_SCHEDULE_HISTORY = 10` (inline entries), `AGENT_SCHEDULES_PATH =
data/schedules.json`, `AGENT_RUNS_DIR = data/runs`, `AGENT_RUN_HISTORY = 50` records
per schedule, `AGENT_SCHEDULE_RUN_TIMEOUT = 7200s` (recorder gives up; agent not
killed), `AGENT_GRANT_MAX_USES = 3` (pre-approved actions per run before falling back
to asking).

**Crews:** `CREW_MAX_AGENTS = 5`, `CREW_MAX_TASKS = 8`, `CREW_LLMS` (`RELIFE_CREW_LLMS`,
comma-separated non-Claude models the planner may staff; keys stay in the env for
CrewAI/LiteLLM), `CREW_CLAUDE_LLM = claude-max` (always available).

Most are overridable via `RELIFE_*` environment variables (see `config.py`).

---

*End of module deep dive. For the interactive Socratic grilling, start at M3 (the
policy shape argument), then M5→M6 (the relevance gate and why reinforcement is
safe), then M8→M10 (why the daemon binds globals, why auth needs a cookie, why a
scheduled run is a turn), then M13→M15 (why scope comes from identity, why tools are
defined once, why a CrewAI agent gets no shell).*
