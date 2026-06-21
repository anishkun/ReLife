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
job it can **write down what it learned** (facts + reusable how-to "skills") so
the next run is smarter. It runs on your **Claude Code Max subscription**, not a
paid-per-call API key.

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
        │                           hooks._recall_hook looks up memories + skills
        │                           matching "scaffold a weather CLI" and silently
        │                           prepends them to the prompt as extra context.
        ▼
 7. Claude works.                   It thinks, then calls tools: Write a file,
        │                           run `pytest`, `git init`, etc. EVERY tool call
        │                           is intercepted by step 2's gatekeeper:
        │                              • Read/Write-in-workspace/test/git → allowed
        │                              • email/publish/write-outside-workspace → ASK you
        ▼
 8. agent._render(msg)              Each streamed message is pretty-printed:
        │                           "→ Write weather/cli.py", "✓ done", etc.
        ▼
 9. (optional) Claude calls memory_save / skill_write to record a lesson, so the
    next run benefits. Then the session ends.
```

The whole architecture is just: **assemble options → stream Claude → gate every
tool call → render.** `do` and `chat` differ only in step 5 (chat loops, asking
you for the next message each time).

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

### The memory layer — `relife/memory/`

This is ReLife's notebook. **Two stores, both using plain keyword + recency
matching — no AI embeddings yet** (kept simple for v1; the API is designed so a
vector search can be slotted in later without callers noticing).

- **`store.py`** — facts/preferences/episodes in a SQLite database
  (`data/relife.db`). `save(text, kind, tags)` inserts a memory (exact duplicates
  just refresh their timestamp instead of piling up). `recall(query, k)` scores
  every memory by *how many words overlap* with your query plus a small bonus for
  *being recent*, and returns the top `k`. Zero-overlap memories are excluded, so
  an unrelated task surfaces nothing.
- **`skills.py`** — reusable *procedures*, one Markdown-with-frontmatter file each
  under `data/skills/`. A skill is a mini how-to ("how to push a new repo to
  GitHub"). Recall is the same keyword approach but **name matches count double**.
- **`server.py`** — wraps both stores as the **MCP server** named `relife_memory`,
  exposing four tools to Claude: `memory_save`, `memory_recall`, `skill_write`,
  `skill_find`. Because the name starts with `relife`, the permission policy
  auto-trusts them.
- **`_text.py`** — the shared tokenizer (lowercases, splits into words, drops
  stop-words like "the"/"a") used by both stores' recall scoring.

**Fact vs. skill:** a *fact* is a thing that's true ("the user prefers ruff");
a *skill* is a procedure you can replay ("steps to scaffold a FastAPI service").

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
│   ├─ relife.db            the memory database (facts/preferences/episodes)
│   ├─ skills/              saved skill files (one .md each)
│   └─ builds/<id>/         one folder per `relife build` (ledger.json + plan.md)
├─ tests/                   26 deterministic tests (no live AI — safe & fast)
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
- **The tests never call the live model.** All 26 tests are deterministic — they
  test the *policy and plumbing* (permission decisions, memory recall scoring,
  ledger persistence), not Claude. So you can run them freely without spending
  budget. Prefer them for checking changes.
- **The agent has two shells on Windows** (`Bash` and `PowerShell`) and the
  permission policy gates both identically.
- **Memory is "dumb" on purpose (for now).** Keyword + recency, no embeddings.
  It's good enough for v1 and the API is built so a smarter vector search can drop
  in later invisibly.
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
5. A **hook** auto-feeds it relevant past **memories + skills** before each prompt.
6. After a job it can **save facts/skills** so it improves over time.
7. For **big** jobs, `relife build` **plans milestones → delegates each to a
   fresh builder → records everything in a ledger**, which makes it **resumable**.
8. It runs on your **Max subscription**; deterministic **tests** verify the
   plumbing without spending budget.

That's ReLife. When in doubt, open `plan.md` inside a build folder to *see* the
machine thinking, or re-read §4 (one request) and §6 (a build).
