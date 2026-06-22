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

That's the whole product. Everything below is *how* that paragraph is true.

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
   │  ReLife (this Python project — ~10 small files)         │
   │   • decides what Claude is allowed to do (permissions)  │
   │   • gives Claude extra abilities (MCP servers)          │
   │   • feeds Claude its past memories (hooks)              │
   │   • for big jobs, splits work into milestones (build)   │
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
- **relife_memory** — ReLife's *own* server exposing `memory_save` / `memory_recall` /
  `skill_write` / `skill_find`. It runs **in-process** (same Python program), but
  is dressed up as an MCP server so it can be split into a separate service later
  without changing how the agent calls it.
- **relife_build** — a per-build server exposing the ledger tools (only during
  `relife build`).

---

## 3. The three ways you talk to it

```sh
relife do "<task>"     # one shot: do this task to completion, then stop
relife chat            # back-and-forth conversation in one workspace
relife build "<spec>"  # BIG job: plan → split into milestones → build each
relife build --resume  # continue a build that was interrupted
relife consolidate     # run the cheap "sleep" pass now (fade/merge/learn) — no AI
relife dream           # opt-in DEEP review: AI critiques & tidies memory (spends budget)
relife memory stats    # peek at what's remembered and what has faded
```

All of them take `--workspace PATH` (default `./workspace`) — **the only folder
the agent is allowed to freely write into.** Think of the workspace as the
agent's desk: it can do whatever it wants on its own desk, but needs your nod to
touch anything off it.

`do` and `chat` are for normal-sized tasks (one Claude conversation is enough).
`build` exists for projects too large to fit in a single conversation — see §6.

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
loops, asking you for the next message each time).

---

## 5. The files, one by one (low level)

The package lives in `relife/`. Here's what each file is responsible for. They're
listed in rough order of how central they are.

### `cli.py` — the front door
Defines the `relife do` / `chat` / `build` commands using **Typer** (a library
that turns Python functions into a CLI). Each command does the same three-step
setup (resolve workspace → build the permission callback → attach MCP servers +
hooks) then calls into `agent.py` or `build/orchestrator.py`. Thin glue, no logic.

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
                                           "outward/destructive" regex →  ASK
mcp__relife* / mcp__browser* ............→ ALLOW   (ReLife's own trusted tools)
anything else (unknown tool) ............→ ASK     (fail closed — safe default)
```

The "outward/destructive" regex (`_OUTWARD_BASH`) is the safety net: it catches
email senders, `gh pr/issue/release/api/gist`, file uploads via curl/wget,
`scp/ssh/rsync`, package publishing (`npm publish`, `twine upload`…), `sudo`, and
`rm -rf /`. **Note git is deliberately NOT in it** — you authorized git including
`git push`, so commits and pushes run without asking.

`make_permission_callback()` wraps `classify` for real use: on "ask" it prints a
yellow prompt and waits for `y/N`. In a **non-interactive** run (no real
terminal), "ask" becomes an automatic **deny** — so an unattended run never hangs
*and* never takes an unapproved outward action. (This is why the earlier live
builds completed without pausing: nothing they did was outward-facing.)

### `config.py` — the settings drawer
One small module holding everything tunable: the model + effort level, all the
filesystem paths (`data/`, `workspace/`, prompts), `agent_env()` (prepends the
GitHub CLI dir to PATH if `gh` isn't found), and `default_mcp_servers()` (the
browser + memory servers attached to every run). Also sets `setting_sources=None`
indirectly — ReLife refuses to inherit the surrounding repo's Claude Code config,
so it behaves identically wherever it's run.

### `hooks.py` — automatic memory recall
A **hook** is a callback the SDK fires at lifecycle moments. ReLife registers one
on `UserPromptSubmit` (fires right before your prompt reaches Claude). It calls
`store.recall()` + `skills.find_skills()` on your prompt text and, if anything
relevant turns up, **injects it as hidden extra context.** The effect: the agent
"remembers" relevant past lessons *without having to decide to look them up*. It
still has the manual `memory_recall` tool for explicit lookups.

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
- **`server.py`** — wraps all of the above as the **MCP server** `relife_memory`,
  exposing the tools Claude calls: `memory_save` (with an `importance` dial),
  `memory_recall`, `memory_forget`, `skill_write`/`skill_find`,
  `workflow_save`/`workflow_find`, `memory_consolidate`, and `memory_dream` (the
  REM pass). Names start with `relife` → the permission policy auto-trusts them.
- **`_text.py`** — the shared tokenizer (lowercases, splits words, drops
  stop-words like "the"/"a") used by every keyword path.

**Fact vs. skill vs. workflow:** a *fact* is a thing that's true ("the user
prefers ruff"); a *skill* is one procedure you can replay ("scaffold a FastAPI
service"); a *workflow* is a multi-stage plan ("ship a new service end to end").

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
│   └─ builds/<id>/         one folder per `relife build` (ledger.json + plan.md)
├─ tests/                   77 deterministic tests (no live AI — safe & fast)
├─ CLAUDE.md                instructions FOR the agent when editing this repo
├─ PROJECT_CONTEXT.md       the authoritative design/status doc (terse)
└─ HOW_IT_WORKS.md          ← you are here (the friendly guide)
```

---

## 8. Things that surprise people (worth knowing)

- **It costs subscription budget, not dollars-per-call.** ReLife uses your Claude
  Code **Max** subscription. `ANTHROPIC_API_KEY` is intentionally unset; if it
  were set, ReLife would refuse to use it. Heavy runs (especially big builds) draw
  on the *same* usage budget as your interactive Claude Code — so a giant build
  can hit "you've hit your session limit." That's exactly what `--resume` is for.
- **The tests never call the live model.** All 77 tests are deterministic — they
  test the *policy and plumbing* (permission decisions, memory recall scoring,
  the cognitive activation/decay math, workflow learning, ledger persistence, and
  the REM critic's *application* logic via a stubbed AI), not Claude. So you can
  run them freely without spending budget.
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
8. It runs on your **Max subscription**; deterministic **tests** verify the
   plumbing without spending budget.

That's ReLife. When in doubt, open `plan.md` inside a build folder to *see* the
machine thinking, or re-read §4 (one request) and §6 (a build).
