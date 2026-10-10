# How ReLife Works — A Reader's Guide

> This file is **for you, the human**, not for the agent. It explains ReLife from
> the top (what it is and why) down to the bottom (what each file does and how a
> single request flows through the code). Read it start to finish once; after
> that, use the section headers to jump back to whatever you forgot.
>
> Nothing here is needed to *run* the project — it's purely to *understand* it.
> The terse, authoritative notes live in `PROJECT_CONTEXT.md` and `CLAUDE.md`;
> this is the friendly walkthrough.

---

## 1. The one-paragraph version

ReLife is a **personal AI agent that does real work on your computer and gets
better over time.** You give it a task in plain English ("scaffold a weather
CLI", "build me a todo app"); it plans, writes code, runs tests, uses git, and
can drive a web browser — all on its own, asking permission only for things that
reach *outside* your machine (sending email, publishing packages). After each
job it **writes down what it learned** (facts, reusable "skills", and multi-step
"workflows"), and its memory works **like a brain** — what it keeps using stays
sharp, what it ignores quietly fades, and it even **invents its own workflows**
from things it finds itself repeating. It runs on your **Claude Code Max
subscription**, not a paid-per-call API key.

It can also stay running (a web console, tasks on a schedule, working the GitHub
issues assigned to you), and it has grown into a small **platform**: you can register
several agents, each with its own memory; hand one agent's memory to a new one; let
*other* AI apps (Cursor, Gemini CLI, a model running on your own machine) use
ReLife's memory; and have CrewAI plan a team of agents for a job that ReLife then
staffs and runs.

That's the whole product. Everything below is *how* those paragraphs are true.

---

## 2. The mental model (read this part slowly)

ReLife is **not** an AI model. It's a thin, opinionated **harness wrapped around
Claude**. Picture three layers:

```
   ┌─────────────────────────────────────────────────────────┐
   │  YOU                                                     │
   │   relife do "build a todo app"                          │
   └───────────────────────────┬─────────────────────────────┘
                               │
   ┌───────────────────────────▼─────────────────────────────┐
   │  ReLife (this Python project — a few dozen small files) │
   │   • decides what Claude is allowed to do (permissions)  │
   │   • gives Claude extra abilities (MCP servers)          │
   │   • feeds Claude its past memories (hooks)              │
   │   • for big jobs, splits work into milestones (build)   │
   │   • lets other agents & models share its memory         │
   └───────────────────────────┬─────────────────────────────┘
                               │  (Claude Agent SDK)
   ┌───────────────────────────▼─────────────────────────────┐
   │  The `claude` CLI  →  Claude (the model)                │
   │   Reads/writes files, runs shell commands, thinks,      │
   │   calls tools. This is the "brain + hands".             │
   └─────────────────────────────────────────────────────────┘
```

The key insight: **ReLife doesn't contain intelligence. It contains *policy and
plumbing*.** Claude is the intelligence; ReLife decides what Claude can touch,
what context it gets, and — for large jobs — how the work is broken up so it
fits in Claude's limited working memory.

Three things ReLife *adds* on top of raw Claude:

| What | Why it matters |
|------|----------------|
| **Permissions** | Lets Claude act autonomously on safe stuff (code, git) while still stopping at anything that could affect the outside world. |
| **Memory** | Raw Claude forgets everything between runs. ReLife gives it a notebook (facts + skills) it can re-read. |
| **Build orchestration** | A single Claude conversation can only hold so much. Big projects are split into milestones, each done in a *fresh* conversation. |
| **Agents & crews** | Several agents, each with its own memory, that can inherit from each other — and teams of them, planned by CrewAI. |

### What is the "Claude Agent SDK"?

A Python library from Anthropic that lets your code *drive* Claude programmatically.
You hand it options (which model, what's allowed, what tools exist) and a prompt;
it streams back Claude's thoughts, tool calls, and results. ReLife is essentially
a carefully-configured caller of this SDK.

### What is "MCP"?

**MCP (Model Context Protocol)** is the standard way to give Claude *new tools*.
An "MCP server" is just a program that advertises a set of tools (each with a
name and inputs). When attached, Claude can call them like any built-in. ReLife
uses MCP three ways:
- **browser** — an off-the-shelf server (Microsoft's Playwright) that lets Claude
  open web pages, click, and type.
- **relife_memory** — ReLife's *own* server exposing the memory tools
  (`memory_save`, `memory_recall`, `memory_forget`, `skill_write`/`skill_find`,
  `workflow_save`/`workflow_find`, `memory_consolidate`, `memory_dream`). It runs
  **in-process** (same Python program), but is dressed up as an MCP server so it
  could be split into a separate service later without changing how the agent calls
  it. That bet paid off twice: once for the memory daemon, and again when other AI
  apps needed to use the same memory (see §6d).
- **relife_build** — a per-build server exposing the ledger tools (only during
  `relife build`).

And it works the other way round too: `relife mcp --agent NAME` turns ReLife's memory
into an MCP server *for other programs* (Cursor, Claude Desktop, Gemini CLI, …).

---

## 3. The three ways you talk to it

```sh
relife do "<task>"     # one shot: do this task to completion, then stop
relife chat            # back-and-forth conversation in one workspace
relife work [ISSUE]    # list your assigned GitHub issues, or fix one → branch → PR (see §6c)
relife build "<spec>"  # BIG job: plan → split into milestones → build each
relife build --resume  # continue a build that was interrupted
relife serve           # ALWAYS-ON: agent server + web console + schedules (see §6b)
relife doctor          # is everything set up? (login, node, gh, connectors, server, schedules, crews)
relife consolidate     # run the cheap "sleep" pass now (fade/merge/learn) — no AI
relife dream           # opt-in DEEP review: AI critiques & tidies memory (spends budget)
relife memory stats    # peek at what's remembered and what has faded
relife memory search|list|show|forget   # look at — and correct — what it learned
relife memory spaces|export|import      # each agent's memory; move it between machines
relife agent create|list|show|promote|… # register agents and hand memory between them (§6d)
relife mcp --agent NAME                 # give another AI app ReLife's memory (§6d)
relife crew "<task>"   # CrewAI plans a team; ReLife staffs and runs it (§6d)
relife crews [ID]      # what past crews did
```

All the commands that run the agent take `--workspace PATH` (default `./workspace`)
— **the only folder the agent is allowed to freely write into.** Think of the
workspace as the agent's desk: it can do whatever it wants on its own desk, but needs
your nod to touch anything off it. `do` and `chat` also take `--agent NAME`, to run
as one of your registered agents with *its* memory instead of yours.

`do` and `chat` are for normal-sized tasks (one Claude conversation is enough).
`build` exists for projects too large to fit in a single conversation — see §6.
`serve` is for when you want the agent *around* — in a browser tab, and running
tasks on a schedule while you're not — see §6b. `work` is for GitHub issues — §6c.
`agent`, `mcp` and `crew` are the platform side — §6d.

---

## 4. How ONE request flows through the code (the golden path)

Let's trace `relife do "scaffold a weather CLI"` end to end. This is the single
most useful thing to understand; everything else is a variation.

```
 1. cli.py (do)                     You typed the command. Typer parses it.
        │                           Resolves the workspace folder, makes sure it exists.
        ▼
 2. permissions.make_permission_callback(workspace)
        │                           Builds the "can Claude do X?" gatekeeper,
        │                           locked to THIS workspace.
        ▼
 3. config.default_mcp_servers()    Attaches the browser + memory tool-servers.
        │
        ▼
 4. hooks.memory_hooks()            Sets up the "before each prompt, inject
        │                           relevant memories" hook.
        ▼
 5. agent.run_task(...)             Bundles everything into ClaudeAgentOptions
        │                           (model, system prompt, permissions, tools,
        │                           hooks) and opens a streaming session.
        ▼
 6. ── UserPromptSubmit hook fires ──
        │                           hooks._recall_hook looks up memories + skills +
        │                           workflows matching "scaffold a weather CLI",
        │                           silently prepends them as extra context, AND
        │                           reinforces them (recall = a use → they get stronger).
        ▼
 7. Claude works.                   It thinks, then calls tools: Write a file,
        │                           run `pytest`, `git init`, etc. EVERY tool call
        │                           is intercepted by step 2's gatekeeper:
        │                              • Read/Write-in-workspace/test/git → allowed
        │                              • email/publish/write-outside-workspace → ASK you
        │                           A PostToolUse hook also journals each call to the
        │                           event log (raw material for learning workflows).
        ▼
 8. agent._render(msg)              Each streamed message is pretty-printed:
        │                           "→ Write weather/cli.py", "✓ done", etc.
        ▼
 9. (optional) Claude calls memory_save / skill_write / workflow_save to record a
    lesson, so the next run benefits.
        ▼
10. ── consolidation ("sleep") ──   When the run ends, if enough has happened,
    ReLife fades unused memories, merges duplicates, and turns repeated tool
    sequences into new workflows — automatically. Then the session ends.
```

The whole architecture is just: **assemble options → stream Claude → gate every
tool call → render → consolidate.** `do` and `chat` differ only in step 5 (chat
loops, asking you for the next message each time). With `--agent NAME`, steps 3–4
hand Claude that agent's memory instead of yours; nothing else changes.

---

## 5. The files, one by one (low level)

The package lives in `relife/`. Here's what each file is responsible for. They're
listed in rough order of how central they are.

### `cli.py` — the front door
Defines every `relife …` command using **Typer** (a library that turns Python
functions into a CLI). The agent-running commands do the same three-step setup
(resolve workspace → build the permission callback → attach MCP servers + hooks)
then call into `agent.py`, `build/orchestrator.py` or `crew/runner.py`. Thin glue,
no logic.

### `agent.py` — the engine
The heart of the "drive Claude" loop. Two things to know:
- **`build_options(...)`** — assembles a `ClaudeAgentOptions` object: which model
  (`claude-opus-4-8`), the system prompt (persona), the workspace as working dir,
  the permission callback, the MCP servers, the hooks, and (for builds) subagents
  + resume id + budget cap. This is the single place all the pieces get wired
  together.
- **`run_task` / `run_chat`** — open a `ClaudeSDKClient` (streaming) and pump
  messages through `_render`. *Streaming* matters: the permission callback only
  works in streaming mode (a noted gotcha).
- It also reconfigures the Windows console to UTF-8 up top, because Rich would
  otherwise crash printing `→`/`✓` on a default Windows code page.

### `permissions.py` — the gatekeeper (the most important policy file)
Decides, for every single tool call, **"allow" or "ask".** The core is
`classify()`, a **pure function** (no side effects → trivially unit-testable):

```
Read/Glob/Grep/WebFetch/Task ...........→ ALLOW   (read-only / planning)
Write/Edit a file INSIDE the workspace ..→ ALLOW
Write/Edit a file OUTSIDE the workspace .→ ASK
Bash/PowerShell command .................→ ALLOW, unless it matches the
                                           "outward/destructive" regex, or it
                                           writes/deletes OUTSIDE the workspace
                                                                        →  ASK
mcp__relife* / mcp__browser* ............→ ALLOW   (ReLife's own trusted tools)
anything else (unknown tool) ............→ ASK     (fail closed — safe default)
```

The "outward/destructive" regex (`_OUTWARD_SHELL`) is the first safety net: it
catches email senders, file uploads via curl/wget, `scp/ssh/rsync`, package
publishing (`npm publish`, `twine upload`…), `sudo`, and `rm -rf /` — **and their
PowerShell equivalents** (`Send-MailMessage`, `Invoke-RestMethod -Method POST`,
`Enter-PSSession`, `Start-Process -Verb RunAs`, `Format-Volume`), which matters
because PowerShell is the shell the agent reaches for on Windows. Installing packages
*globally* (outside a project `.venv`) asks too. The second net is containment: a
shell command that *writes to* or *deletes* something outside the workspace asks
too, so a redirect (`echo x > ~/.bashrc`) can't route around the file-write rule.
**Note git is deliberately NOT in either** — you authorized git including `git
push`, so commits and pushes run without asking.

The GitHub CLI (`gh`) and the Gmail / Calendar / Drive connectors are judged by
**what the command says it does**: reading (`gh issue list`, `gh pr view`, searching
your mail) runs on its own; anything that changes something (`gh pr create`, `gh repo
delete`, sending an email, creating an event) asks; and anything it doesn't
recognize asks too.

`make_permission_callback()` wraps `classify` for real use: on "ask" it prints a
yellow prompt and waits for `y/N`. In a **non-interactive** run (no real
terminal), "ask" becomes an automatic **deny** — so an unattended run never hangs
*and* never takes an unapproved outward action. (This is why the earlier live
builds completed without pausing: nothing they did was outward-facing.) The one
exception is a schedule's **pre-approvals** — see §6b.

### `config.py` — the settings drawer
One small module holding everything tunable: the model + effort level, all the
filesystem paths (`data/`, `workspace/`, prompts — `RELIFE_HOME` moves them),
`agent_env()` (prepends the GitHub CLI dir to PATH if `gh` isn't found), and
`default_mcp_servers()` (the browser + memory servers attached to every run). Also sets `setting_sources=None`
indirectly — ReLife refuses to inherit the surrounding repo's Claude Code config,
so it behaves identically wherever it's run.

### `hooks.py` — automatic memory recall
A **hook** is a callback the SDK fires at lifecycle moments. ReLife registers three:
- **before your prompt reaches Claude** (`UserPromptSubmit`) it looks up memories,
  skills and workflows matching your prompt and, if anything relevant turns up,
  **injects it as hidden extra context** — so the agent "remembers" relevant past
  lessons *without having to decide to look them up*;
- **after every tool call** (`PostToolUse`) it writes one line to the event journal;
- **when the run stops** (`Stop`) it saves a short "episode" — what the task was and
  how it was approached.

The agent still has the manual `memory_recall` tool for explicit lookups. If memory
is unreachable for some reason, the hooks quietly do nothing rather than break the
run.

### `prompts/system.md` — the persona
A Markdown file appended onto Claude Code's built-in "preset" system prompt. It
defines ReLife's personality, its safety rules, and — crucially — *tells the agent
to save durable lessons to memory/skills* after finishing work. (The build
orchestrator swaps in `build/prompts/orchestrator.md` instead.)

### The memory layer — `relife/memory/` (works like a brain)

This is ReLife's notebook, and it's the cleverest part. The big idea: **a
memory's relevance isn't fixed.** It *rises* every time the memory gets used and
*fades* when it sits unused — just like human memory. Finished, never-touched-
again notes quietly sink out of view; the things ReLife keeps relying on stay
sharp. And it doesn't just store notes — it **watches what it does and invents
its own multi-step workflows** from repetition.

The four signals that decide what surfaces (all fused into one score):
*does it mean the same thing?* (semantic) · *do the words overlap?* (keyword) ·
*how strong is it right now?* (activation — the rise/fade) · *how important did we
mark it?* (importance).

- **`cognitive.py`** — the **pure math** of the brain model (no database, no AI,
  so it's trivially testable). It computes a memory's **activation**: more uses +
  more recent = stronger; long idle = weaker. It also decides when something has
  faded enough to **forget** (archive). This file is *why* memory behaves alive.
- **`store.py`** — the facts/preferences/episodes/patterns database
  (`data/relife.db`). `recall(query)` is **two-stage** so it stays fast even with
  huge memory: first a cheap **index** (SQLite FTS5) narrows millions of rows to a
  handful of candidates, then the full four-signal score ranks just those.
  Recalling a memory **reinforces** it (recall is a use). Re-saving the same text
  strengthens it instead of duplicating. Faded memories are **archived, not
  deleted** — reversible, like a memory you *could* still dredge up.
- **`embeddings.py`** — gives recall its *sense of meaning*. A small model runs
  **locally on your machine** (no API key, works offline) to turn text into
  vectors so "set up CI" can match "configure the test pipeline" even with no
  shared words. It's **optional**: not installed → memory just falls back to
  keyword matching, nothing breaks. (`pip install -e ".[embeddings]"` to enable.)
- **`skills.py`** — single reusable *procedures*, one Markdown file each under
  `data/skills/` ("how to push a new repo to GitHub").
- **`workflows.py`** — *multi-step* procedures: an ordered chain of stages
  ("scaffold → test → make repo → push"), under `data/workflows/`. The difference
  from a skill is that the value is in the **sequence**.
- **`events.py`** — a quiet journal of every tool the agent uses. On its own it's
  boring; it's the **raw material** the next file mines for patterns.
- **`consolidate.py`** — ReLife's **"sleep" pass.** Periodically (after runs, or
  via `relife consolidate`) it does brain-like housekeeping: **fades/archives**
  unused memories, **merges** duplicates, and — the magic part — scans the event
  journal for **action sequences it keeps repeating** and **writes them up as new
  workflows automatically.** So ReLife literally learns "whenever I do X I tend to
  do Y then Z" and saves that plan for next time. (Deliberately kept AI-free so
  it's cheap and safe to run on its own.)
- **`rem.py`** — ReLife's **"dream" pass** (`relife dream`), the brain-analogy
  taken one step further. Sleep (above) is cheap, automatic, and mechanical. REM
  is the **opt-in, AI-powered deep review** you run *on purpose* when you know you
  have budget to spare — because, unlike everything else in the memory layer, this
  one **calls the model.** It points Claude at your most recent memories as an
  **adversarial critic** and asks: *do any of these contradict each other? is any
  of this unsafe, hallucinated, or junk? is anything rated too important or not
  important enough?* The crucial safety design: **the AI only advises — it never
  has the keys.** Its suggestions are applied by plain, deterministic code, and:
    - the worst it can do is **archive** a memory (reversible — never a hard delete);
    - it **cannot edit the text** of a memory, only hide it or re-rank its importance;
    - low-confidence suggestions are **ignored**, and there's a hard **cap** on how
      much a single pass may archive — so even a bad review can't gut your memory;
    - every action is **logged to `data/rem_journal.jsonl`** with the reason, so you
      can see (and undo) exactly what it did.
  In short: **diminishing-returns polish, not a miracle.** It catches the
  qualitative problems the mechanical sleep pass can't — but it doesn't change how
  recall ranks things, so it makes memory *cleaner and safer*, not magically
  smarter. It's gated behind a manual command for **risk** reasons (an AI editing
  its own memory unsupervised is dangerous) as much as cost.
- **`tools.py`** — the memory tools, each written **once**: name, description,
  inputs, and what it does. Everything below serves these same definitions.
- **`server.py`** — wraps them as the **MCP server** `relife_memory`, exposing the
  tools Claude calls: `memory_save` (with an `importance` dial), `memory_recall`,
  `memory_forget`, `skill_write`/`skill_find`, `workflow_save`/`workflow_find`,
  `memory_consolidate`, and `memory_dream` (the REM pass). Names start with
  `relife` → the permission policy auto-trusts them.
- **`mcp_server.py`** — the same memory as a **standalone MCP server for other AI
  apps** (§6d), minus the two housekeeping tools and plus `memory_context`, which
  hands an app the same "here's what memory knows" block the hook gives Claude.
- **`context.py`** — builds that block (shared by the hook and `memory_context`).
- **`spaces.py`** — memory **spaces**: each agent's own corner of the memory
  (§6d). Everything from before spaces existed is in the `default` space — yours.
- **`_text.py`** — the shared tokenizer (lowercases, splits words, drops
  stop-words like "the"/"a") used by every keyword path.

**Fact vs. skill vs. workflow:** a *fact* is a thing that's true ("the user
prefers ruff"); a *skill* is one procedure you can replay ("scaffold a FastAPI
service"); a *workflow* is a multi-stage plan ("ship a new service end to end").

### The rest of the top-level files
- **`agents.py`** — the agent registry (`data/agents.json`): who each agent is, which
  memory it writes, which it may read, and the hand-over operations (§6d).
- **`workitems.py`** — the GitHub plumbing behind `relife work` (§6c).
- **`doctor.py`** — `relife doctor`'s checks (§6b).
- **`crew/`** — `relife crew` (§6d).

---

## 6. The build system — `relife/build/` (the part that's hard to follow)

This is the most sophisticated piece, and the one the live test exercised. Read
this section carefully; it answers "what actually happens when I run `relife build`."

### The problem it solves

Claude has a **limited context window** — a finite amount it can "hold in its
head" at once. A small task fits. But "build a full multi-service app" generates
so much code, test output, and back-and-forth that a single conversation would
overflow and the agent would start forgetting its own earlier decisions.

### The solution: decompose → delegate → resume

```
        relife build "build a todo app"
                  │
                  ▼
   ┌─────────────────────────────────────┐
   │  ORCHESTRATOR (one Claude session)  │   ← stays small & strategic.
   │  "I'm the architect / project mgr"  │     It plans and delegates;
   └───────────────┬─────────────────────┘     it does NOT write the code itself.
                  │
        1. Think through architecture.
        2. build_plan_set([...milestones])  ──► writes the LEDGER (plan) to disk
                  │
        3. For each milestone, in order:
                  │
                  ├─ build_milestone_update(id, "in_progress")
                  │
                  ├─ delegate via the Task tool ──►  ┌──────────────────────────┐
                  │                                  │  BUILDER subagent        │
                  │                                  │  (a FRESH Claude session)│
                  │   "implement milestone 3,        │  • reads existing code   │
                  │    verify it, report back        │  • writes this milestone │
                  │    a SHORT summary"              │  • runs the tests        │
                  │                                  │  • returns 2-3 sentences │
                  │   ◄──────── concise summary ─────┤  (NOT the full code)     │
                  │                                  └──────────────────────────┘
                  │
                  └─ build_milestone_update(id, "done", summary)  ──► updates LEDGER
                  │
        4. When all milestones done → build complete.
```

The trick is the **builder subagent**. Each milestone runs in its *own* fresh
context window (via the SDK's **Task** tool). The builder absorbs all the messy
implementation detail — file contents, test output, debugging — and hands the
orchestrator back only a **short summary**. So the orchestrator's context stays
lean no matter how big the project gets: it only ever holds the plan + a
paragraph per finished milestone.

### Why this also makes builds *resumable*

Everything important is written to disk in a **ledger** *as it happens*. So if the
run dies halfway (crash, you hit your Max session limit, you close the laptop),
nothing is lost. `relife build --resume` reloads the ledger, sees which
milestones are already `done`, and continues from the first unfinished one — even
if the original conversation is gone.

### The build files

- **`ledger.py` — `BuildLedger`**: the durable record. One per build at
  `data/builds/<build_id>/ledger.json`, with a human-readable `plan.md` mirror
  re-rendered on every change (that's the file you can open to watch progress). It
  holds the spec, the workspace path, a `session_id` (to resume the same Claude
  conversation), and the list of milestones with their status (`pending` →
  `in_progress` → `done`/`failed`) and summaries. **Pure and deterministic** — no
  AI calls — so it's fully unit-tested. Writes are atomic (temp file → rename) so
  a crash mid-write can't corrupt it.
- **`server.py` — the `relife_build` MCP server**: exposes three tools to the
  orchestrator — `build_plan_set` (record the milestones), `build_milestone_update`
  (change a milestone's status + summary), `build_status` (read the current
  ledger; the first thing it calls on resume). It's bound to *one specific ledger*
  for the run via a closure, so the tools mutate the right file. Tool names start
  with `mcp__relife_build__` → auto-trusted by the permission policy, no change
  needed.
- **`agents.py` — the `builder` definition**: describes the subagent the
  orchestrator delegates to — its instructions ("implement exactly ONE milestone,
  verify it, report back concisely, don't paste full files"), and the tools it's
  allowed (read/write/edit/shell/browser, but **not** the ledger tools — those
  belong to the orchestrator).
- **`orchestrator.py` — `run_build()`**: ties it all together. Creates or loads
  the ledger, attaches the `relife_build` server + the `builder` subagent + the
  orchestrator persona, streams the run, and persists the `session_id` after each
  message so `--resume` can continue. The resume prompt re-injects the ledger
  state, so resume works even if the live session handle is gone.
- **`prompts/orchestrator.md`** — the orchestrator's persona: "you are an
  architect / project manager; plan and delegate, don't build it yourself."

### A real example (the live test we just ran)

Spec: *"a small Python CLI named tempconv that converts between C/F/K…"*. The
orchestrator decomposed it into **4 milestones** (scaffold + core math → CLI →
packaging → tests), delegated each to a fresh builder, and finished — **41 tests
passing**, **$1.39** of usage, all tracked in
`data/builds/20260621-131239-a556/`. Open that folder's `plan.md` to see exactly
what each milestone produced. That's the whole machine working end to end.

---

## 6b. The always-on side — `relife serve` (server, console, schedules)

`do` and `chat` are **cold**: a process per task, and approvals happen in the
terminal. `relife serve` keeps one process running instead, and three things fall
out of that.

### The web console

Open `http://127.0.0.1:8600` and you get a plain, self-contained page: type a
task, watch the agent's text / tool calls / results stream in live, and — when it
wants to do something outward (send an email, post to GitHub) — an **approval
card** appears with the concrete details (recipient, subject, command). Click
allow or deny. If you don't click within five minutes the action is **denied**,
exactly like an unattended terminal run.

Behind the page: each browser session is one long-lived agent conversation (one
`claude` subprocess, kept open across turns), the page reattaches to it on
reload instead of starting a new one, and the server is **local-only** by design
(binding beyond `127.0.0.1` requires a token and is refused without one).

### Schedules — tasks that run themselves

In the console's *schedules* panel you add a task and a cadence — `every 30m`,
or `daily at 09:00 on mon, wed, fri`. When it's due, the server fires the task
**as a turn in its own session**, so it recalls memory, journals, learns and asks
for approvals exactly like something you typed. Nobody watching ⇒ an approval
times out ⇒ denied, and the agent is told up front to *report* that rather than
retry.

Every run's **outcome is recorded** (`data/runs/…`): the agent's closing summary,
how many tools it used, the cost, and — the important part — **anything it
needed you for** (each denied action, with what it was). The panel shows the
last outcome on the card and a *runs* list per schedule. Intervals under five
minutes are refused: every run spends your Max budget.

**Pre-approvals.** "Summarize my inbox and email me the digest" could never finish
unattended — nobody is there to approve the email. So a schedule can carry a few
narrow **pre-approvals**: *email* or *calendar events* to addresses you list (at most
five), or — for a work schedule (§6c) — opening *the one pull request* for the issue
that run worked on. Everything else still asks. A run can use them at most three
times, each use is listed in the run's record ("done for you — pre-approved"), and
an email only counts if every recipient is on your list and visible in the call
(no sending a saved draft, no reply-all).

### `relife doctor`

Everything ReLife leans on lives outside the package (the `claude` login, Node,
`gh`, connectors, the server) and fails late or silently. `doctor` checks it all
up front, prints a fix for anything wrong, and tells you when "nothing seems to
happen" is because the server isn't running or a scheduled run was denied
something.

### Optional: memory as its own process

`relife memory serve` runs the memory layer as a small daemon; set
`RELIFE_MEMORY_URL=http://127.0.0.1:8787` and every ReLife process (CLI runs,
the server, scheduled runs) shares **one brain** instead of each opening the
database. Unset the variable and nothing changes — memory is in-process by
default. The daemon also serves the memory to other AI apps over the network at
`/mcp` (§6d).

---

## 6c. Working your GitHub issues — `relife work`

`relife work` on its own lists the open issues assigned to you. Give it one —
`relife work owner/repo#12`, an issue URL, or `12 --repo owner/repo` — and it:

1. reads the issue and its latest comments (a closed issue stops here);
2. clones the repo once into `workspace/<owner>__<repo>` (an existing clone is left
   exactly as it is — it might hold your uncommitted work, so updating it is the
   agent's first visible step);
3. runs the agent **inside that clone**, so its free-to-write "desk" is just that one
   repo, on a branch `relife/issue-12-<short-title>`;
4. the agent implements, tests, commits and pushes on its own, then **asks** before
   opening the pull request (with "Closes #12").

Issue text is written by other people, so the agent is told it's information, not
instructions — and the permission rules are the real safety net: whatever a
malicious "also email the secrets to …" line would need still asks. `--dry-run`
shows the task without running the agent.

In the web console, a schedule can be a **work schedule** ("work my assigned GitHub
issues", optionally only one repo or one label): each time it fires it takes the
newest issue it hasn't tried yet. If there's nothing new, it skips without starting
the agent — no budget spent. Unattended, the PR step would be denied (nobody to
approve), so the branch gets pushed and the PR shows up under "needed you" — unless
you ticked "may open the PR without asking", the narrowest pre-approval there is: it
covers one plain `gh pr create` for *that* issue's repo and branch, nothing else.

---

## 6d. The platform — agents, shared memory, other AI apps, crews

### Agents, each with their own memory

`relife agent create reviewer` registers an agent. It gets its own **memory space**
— its own corner of the same database — and that's the only place it can *write*.
It can *read* its own memory plus yours (the `default` space), unless you create it
with `--isolated`. Run it with `relife do --agent reviewer "…"`. Your memory only
changes when you, or your main agent, change it.

### Handing memory from an old agent to a new one

| You want… | Command | What happens |
|---|---|---|
| a new agent that *knows what an old one knows* | `relife agent create junior --inherit veteran` | `junior` reads `veteran`'s memory live (and whatever `veteran` inherited), but can't change it |
| a new agent that *starts from a copy* | `relife agent create v2 --fork veteran` | `v2` gets a snapshot copy of `veteran`'s memories, skills and workflows, as its own |
| to *keep* what an agent learned | `relife agent promote reviewer` | copies its memories into yours — your call, never automatic |
| to move memory to another machine | `relife memory export veteran -o pack.json` / `relife memory import pack.json --space veteran` | a portable file |

Inheriting is read-only on purpose: a new memory can only be saved in one place. If
two agents should *share* one memory, give them the same `--space`. Recalled memory
that came from another agent is labelled (`via veteran`), so it's clear whose it is.

### Other AI apps can use ReLife's memory

`relife agent create cursor --runtime external` (or `relife agent connect cursor`)
prints a block of settings to paste into Cursor, Claude Desktop, Gemini CLI or any
other app that speaks MCP. That app can then save to and recall from *its* agent's
memory (and read yours, unless isolated) — locally via `relife mcp --agent cursor`,
or over the network from `relife memory serve` with a per-agent token (`relife agent
token cursor`; revoke it any time). The app can't pick a different agent's memory,
and it can't run the housekeeping or "dream" passes.

### Crews — a team for one task

`relife crew "add CSV export and review it"`:

1. **Plans a team.** Claude (on your subscription — no API key) proposes a small crew:
   which agents (reusing experienced ones, and letting new ones inherit their memory),
   which model each runs on, and which tasks in what order.
2. **Shows you the plan and asks** before anything runs (`--plan-only` stops here;
   `--yes` skips the question).
3. **Runs it with CrewAI.** Members of two kinds:
   - **ReLife agents** — full Claude agents with all of ReLife's tools and the same
     permission rules; each task runs in `workspace/crews/<id>/`, and anything outward
     still asks you in the terminal;
   - **agents on other models** (local Ollama models, or OpenAI/Gemini with keys you
     set in your environment — list them in `RELIFE_CREW_LLMS`) or `claude-max`
     (Claude through your subscription) — these get ReLife memory but **no shell, no
     files, no browser**, because ReLife's permission rules can't watch over CrewAI's
     own tool loop.
4. **Keeps the result** in `data/crews/<id>/` — `relife crews <id>` shows what each
   task produced, what it cost and anything that was denied. Each agent's lessons
   land in its own memory; `relife agent promote` brings the good ones into yours.

CrewAI doesn't run on Python 3.14 yet, so crews run from a Python 3.12 virtual
environment (`py -3.12 -m venv .venv`, then `.venv\Scripts\pip install -e ".[crewai]"`
and `.venv\Scripts\relife crew …`). `relife doctor` tells you this if you're on
3.14.

---

## 7. Where things live on disk

```
D:\ReLife\
├─ relife/                  the actual program (see §5, §6)
├─ workspace/               the agent's "desk" — where it builds your projects
│   ├─ tempconv-smoke/      the live-test output (a working CLI + 41 tests)
│   └─ todo-smoke/          an earlier full-app build
├─ data/                    runtime stuff (gitignored — not in version control)
│   ├─ relife.db            memory DB (facts/preferences/episodes/patterns + event log)
│   ├─ skills/              saved skill files (one .md each)
│   ├─ workflows/           learned multi-step workflows (one .md each)
│   ├─ consolidate_state.json   bookkeeping for the auto "sleep" pass
│   ├─ rem_state.json       bookkeeping for the "dream" pass (what's been reviewed)
│   ├─ rem_journal.jsonl    audit log of every change the AI critic made (undoable)
│   ├─ builds/<id>/         one folder per `relife build` (ledger.json + plan.md)
│   ├─ schedules.json       the schedules you set up in the console
│   ├─ runs/<schedule>/     one JSON per scheduled run (summary, cost, what needed you)
│   ├─ agents.json          your registered agents (who reads/writes which memory)
│   ├─ spaces/<agent>/      each agent's own skills/ and workflows/
│   ├─ crews/<id>/          one folder per `relife crew` run (plan + outcome)
│   └─ relife.db.daemon     marker left by a running `relife memory serve` (optional)
├─ .venv/                   Python 3.12 environment for crews (gitignored)
├─ tests/                   ~800 deterministic tests (no live AI — safe & fast)
├─ CLAUDE.md                instructions FOR the agent when editing this repo
├─ PROJECT_CONTEXT.md       the authoritative design/status doc (terse)
├─ MODULE_DEEP_DIVE.md      the architecture, module by module, with the reasoning
├─ RELEASE_TESTING.md       the release checklist (incl. the budget-spending smokes)
└─ HOW_IT_WORKS.md          ← you are here (the friendly guide)
```

---

## 8. Things that surprise people (worth knowing)

- **It costs subscription budget, not dollars-per-call.** ReLife uses your Claude
  Code **Max** subscription. `ANTHROPIC_API_KEY` is intentionally unset, and
  `relife doctor` warns if it's set (it would bill metered usage instead). Heavy
  runs (especially big builds and crews) draw on the *same* usage budget as your
  interactive Claude Code — so a giant build can hit "you've hit your session
  limit." That's exactly what `--resume` is for. Even inside a crew, Claude goes
  through your subscription; only the *other* models you choose use their own keys.
- **The tests never call the live model.** All ~800 tests are deterministic — they
  test the *policy and plumbing* (permission decisions, memory recall scoring,
  the cognitive activation/decay math, workflow learning, ledger persistence, the
  REM critic's *application* logic via a stubbed AI, the whole server — auth,
  streaming, approvals, schedules — against a scripted fake agent, the memory MCP
  server over the real protocol, and even a real CrewAI crew run with stand-in
  models), not Claude. So you can run them freely without spending budget.
- **There are two kinds of "memory cleanup", and only one uses AI.** The automatic
  **sleep** pass (`consolidate`) is mechanical and free — it runs itself. The
  **dream** pass (`relife dream`) is the only thing in the whole memory layer that
  calls the model, which is exactly why it's *opt-in* and never fires on its own.
  When you read the code, that line — deterministic-and-automatic vs.
  AI-and-manual — is the cleanest way to keep the two straight.
- **The agent has two shells on Windows** (`Bash` and `PowerShell`) and the
  permission policy gates both identically.
- **Memory fades and learns on its own.** Relevance rises with use and decays when
  ignored; the "sleep" pass forgets stale notes and invents workflows from
  repeated actions — all without you asking. Semantic (meaning-based) recall is a
  *local* model with no API key, and is optional: skip the install and memory
  gracefully falls back to keyword matching.
- **The orchestrator doesn't write your code.** During a build, the actual coding
  is done by the disposable `builder` subagents; the orchestrator only plans and
  tracks. If you watch a build and wonder why the "main" agent isn't typing
  code — that's by design.
- **Agents never write into your memory.** Each writes only its own space; the only
  ways in are you (or your main agent), `relife agent promote`, and `relife memory
  import`. The housekeeping pass also stays inside each space — one agent's habits
  never turn into workflows in another's memory.
- **Agents on other models don't get hands.** In a crew, anything that touches your
  machine (files, shell, git, browser) is done by a ReLife agent under ReLife's
  permission rules; models running inside CrewAI only think, recall and remember.

---

## 9. A 60-second recap

1. ReLife = **policy + plumbing around Claude**, not a model itself.
2. You run `relife do / chat / build`; it works inside a **workspace** (its desk).
3. **Permissions** let it act freely on code/git but stop at outward actions.
4. **MCP servers** give it extra hands: a **browser** and its own **memory**.
5. A **hook** auto-feeds it relevant past **memories + skills + workflows** before
   each prompt — and using a memory makes it **stronger** (unused ones **fade**).
6. After a job it **saves facts/skills/workflows**, and a **"sleep" pass** forgets
   stale notes and **learns new workflows** from repeated actions — so it improves.
   When you have budget to spare, an opt-in **"dream" pass** (`relife dream`) lets
   the AI critique and tidy its own memory — but reversibly, capped, and logged, so
   it can never corrupt itself.
7. For **big** jobs, `relife build` **plans milestones → delegates each to a
   fresh builder → records everything in a ledger**, which makes it **resumable**.
8. `relife serve` keeps it **always on**: a web console with approval cards, and
   **schedules** that run tasks unattended — including working your **GitHub
   issues** — and record what they did and what they needed you for, with narrow
   **pre-approvals** for the few things you trust it to do alone.
9. It's a **platform**: several **agents**, each with its own memory, that can
   **inherit** or **fork** an older agent's memory; **other AI apps** can use the
   memory over MCP; and **crews** let CrewAI plan a team that ReLife staffs.
10. It runs on your **Max subscription**; deterministic **tests** verify the
   plumbing without spending budget.

That's ReLife. When in doubt, open `plan.md` inside a build folder to *see* the
machine thinking, or re-read §4 (one request), §6 (a build), §6b (the server) and §6d
(the platform). For the *why* behind every design choice, read `MODULE_DEEP_DIVE.md`.
