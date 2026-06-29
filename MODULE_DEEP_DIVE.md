# ReLife — Module-by-Module Deep Dive (Architectural Study Guide)

> A complete, ground-up explanation of the ReLife codebase organized as the **M1–M8
> tutoring curriculum**: what the system is, how data flows, *why* each decision was
> made over the obvious alternative, and where the trade-offs and weak points live.
> Written to be *read and re-read* — every module is covered in detail, with the
> causal reasoning and failure modes spelled out, not just the facts.
>
> Companions: `HOW_IT_WORKS.md` is the friendly plain-English walkthrough;
> `CLAUDE.md` is the working ruleset; `PROJECT_CONTEXT.md` is the locked decisions.
> This file explains the *reasoning behind the architecture*.

---

## Table of contents

- [M1. The 30,000-ft view — the central bet](#m1)
- [M2. The agent runner & the SDK seam](#m2)
- [M3. The permission model](#m3)
- [M4. The cognitive core (`cognitive.py`)](#m4)
- [M5. The memory store (`store.py`) + the seams around it](#m5)
- [M6. Hooks & the learning loop (`hooks.py`, `consolidate.py`, `rem.py`)](#m6)
- [M7. Build orchestration (`relife/build/`)](#m7)
- [M8. Trade-offs, failure modes & "why not X"](#m8)
- [Appendix A. The complete request lifecycle (one trace)](#appendix-a)
- [Appendix B. File-by-file index](#appendix-b)
- [Appendix C. Every tunable in `config.py`](#appendix-c)

---

<a name="m1"></a>
## M1. The 30,000-ft view — the central bet

### What ReLife is

ReLife is a **personal agent** built on the **Claude Agent SDK** (Python). It does
two things ordinary scripted agents don't:

1. It **acts** in the world through tools and MCP servers (shell, files, a browser,
   git, GitHub).
2. It **learns over time** — it accumulates *facts* (long-term memory) and
   *procedures* (skills + workflows), and it reshapes that knowledge with brain-like
   maintenance passes.

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
drives the CLI that is already authenticated against Max.

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
isn't contingent on the model "remembering to learn."

### How the code is laid out (the macro map)

```
relife/
  cli.py            entry point: do / chat / build / consolidate / dream / memory
  agent.py          build_options() + the streaming SDK loop + ask_model_oneshot
  permissions.py    classify() pure policy + can_use_tool callback
  hooks.py          UserPromptSubmit/PostToolUse/Stop lifecycle hooks
  config.py         every path, model id, and tunable in one place
  prompts/          system.md (persona), orchestrator.md, rem.md
  memory/
    cognitive.py    pure ACT-R math: activation, fused_score, forgetting
    store.py        SQLite store, two-stage recall, reinforce-on-write
    vector_index.py pluggable semantic candidate search (brute force / ANN)
    embeddings.py   soft-optional local ONNX embeddings
    service.py      MemoryService facade (the memory-only service seam)
    client.py       MemoryClient protocol + LocalMemoryClient (transport seam)
    skills.py       single reusable procedures (Markdown files)
    workflows.py    multi-step procedures (Markdown files)
    events.py       tool-event log (own table in the same SQLite db)
    consolidate.py  deterministic "sleep" pass (LLM-free)
    rem.py          opt-in LLM "dream" pass (adversarial critic)
    server.py       MCP server exposing memory/skills/workflow tools
    _text.py        shared stopword tokenizer
  build/
    ledger.py       durable plan + progress (resume source of truth)
    server.py       MCP server exposing the ledger to the orchestrator
    agents.py       the `builder` subagent definition
    orchestrator.py run_build(): decompose → delegate → resume
```

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

Look at `cli.py:do` (`cli.py:34`): it resolves the workspace, builds the
`can_use_tool` callback bound to that workspace, gets the default MCP servers, gets
the memory hooks, and hands all three to `run_task`. `chat` and `build` do the same
with their own variations. **The CLI's whole job is to assemble three pluggable
pieces and start the loop.**

### `build_options()` — the assembly point

`agent.build_options()` (`agent.py:64`) is the single funnel where a run's
configuration is assembled into `ClaudeAgentOptions`. It takes:

- `cwd` — the workspace,
- `permission_mode` + `can_use_tool` — the permission policy (M3),
- `mcp_servers` — browser + memory (+ build ledger for builds),
- `hooks` — the learning-loop hooks (M6),
- `system_prompt` — persona (default ReLife, or the orchestrator's),
- `agents` — subagent definitions (M7),
- `resume` — a session id to continue,
- `max_budget_usd` — an optional spend cap.

Two non-obvious but important choices live here:

- `setting_sources=None` (`agent.py:99`) — **deliberately do not inherit the
  surrounding repo's Claude Code settings.** ReLife is self-contained; it defines
  its own behavior and must not be silently reconfigured by whatever `.claude/`
  happens to be in the cwd.
- `env=config.agent_env()` (`agent.py:96`) — prepend the GitHub CLI dir to PATH for
  the agent subprocess, but *only* when `gh` isn't already resolvable (see
  `config.agent_env`, `config.py:151`). This exists because `gh` was winget-installed
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
   of subagents into the *same* `build_options`. One assembly funnel, many behaviors.
3. **A future out-of-process split.** Because consumers depend on injected
   interfaces, swapping the implementation behind one (e.g. memory becoming a
   daemon) changes only the injected object, not the consumers.

### The system prompt: preset + append

`preset_system_prompt()` (`agent.py:48`) returns
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

So `run_task` (`agent.py:192`) uses `ClaudeSDKClient` explicitly, and the comment at
`agent.py:204` says exactly why. This even applies to the deny-everything case:
`ask_model_oneshot` (`agent.py:108`) — used by the REM pass for a pure text-in/
text-out judgment — attaches `_deny_all_tools` as its `can_use_tool`, and *because*
it attaches a callback at all, it **must** run in streaming mode too
(`agent.py:117` comment). "No tools" is still enforced *through* the permission seam,
which forces streaming.

### `ask_model_oneshot` — the SDK-touching escape hatch

`ask_model_oneshot(system_prompt, prompt)` (`agent.py:108`) runs the model with **no
tools, no MCP, no hooks** and returns `(text, cost_usd)`. It exists for one caller:
the REM "dream" pass, which needs the model as a *pure advisor* with zero ability to
take action. Tools are hard-denied (`agent.py:103`), so the call can never touch the
filesystem or do anything but emit text. The cost is read off the `ResultMessage`
so REM can report spend.

> M2 mastery check: cli wires 3 pieces → `build_options` assembles them → streaming
> client runs the loop; the injection seam buys testability + persona swap + future
> service split; and `can_use_tool` forces streaming even in the deny-all one-shot.

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
  git), plus ReLife's own MCP tools.
- **Always-ask:** anything *outward-facing* — sending email, posting data off the
  machine, publishing packages, remote shells, GitHub PR/issue/release/api/gist,
  writing *outside* the workspace, and **any unrecognized tool** (fail-closed).

### `classify()` is a pure function

`classify(tool_name, tool_input, workspace)` (`permissions.py:79`) returns
`("allow" | "ask", reason)`. It has **no I/O and no side effects** — given the same
inputs it always returns the same decision. That's what makes the security policy
unit-testable in isolation, without ever spinning up an agent. The wrapper
`make_permission_callback()` (`permissions.py:107`) turns the pure decision into the
SDK's async `can_use_tool`, adding the interactive y/n prompt for the ask cases.

The classification order (`permissions.py:84`):

1. `_ALWAYS_ALLOW_TOOLS` (read-only / planning / shell control) → **allow**.
2. `_FILE_WRITE_TOOLS` → allow **only if** the target path is inside the workspace,
   else **ask**.
3. `_SHELL_TOOLS` (Bash/PowerShell) → **ask** if the command matches the outward/
   destructive regex, else **allow**.
4. Tools starting with a trusted MCP prefix (`mcp__relife`, `mcp__browser`) → allow.
5. **Everything else → ask** (the fail-closed default).

### The deepest idea: denylist for shells, allowlist for tools

This is the architectural insight worth grilling yourself on.

For **shell commands**, the policy uses a **denylist** (`_OUTWARD_BASH`,
`permissions.py:51`): allow by default, ask only for a *finite, enumerable set of
dangerous patterns* (email senders, `gh pr|issue|release|api|gist`, uploading curl/
wget, scp/sftp/rsync/ssh, package publish, `sudo`, root `rm -rf`).

Why denylist here and not an allowlist? Because **the set of safe shell commands is
effectively infinite** (every build invocation, every test runner, every git
subcommand, every file utility...). You cannot enumerate "all safe commands." But
the set of genuinely *dangerous, outward* shell patterns is **small and
enumerable.** When one side of a partition is infinite and the other is finite, you
must define the policy in terms of the finite side. So: enumerate the dangerous,
allow the rest.

For **tools in general** (the unknown-tool case), the policy is the opposite — an
**allowlist** with a fail-closed default (`permissions.py:104`): an unrecognized
tool is *asked*, not allowed. Why is fail-closed cheap *here* but would be
intolerable for shells? Because **unrecognized tools are rare** — they appear only
when a new tool is wired in, which is a development-time event. Prompting on
something that almost never happens costs almost nothing. Shell commands, by
contrast, are *constant* — fail-closed there would mean a prompt on every build and
test, which would make autonomy worthless. **The policy shape on each axis is chosen
by which side is finite and how often the "ask" path fires.**

### Path containment: defeating traversal and symlink escapes

File writes are allowed only `_under()` the workspace (`permissions.py:66`). The
check is not a string-prefix test (which `../` and symlinks would defeat). It:

1. resolves the path against the workspace if relative,
2. calls `.resolve()` on both the target and the workspace — which **collapses `..`
   segments and follows symlinks to their real location**,
3. and only then checks `target == workspace or workspace in target.parents`.

So `workspace/../../etc/passwd` resolves to `/etc/passwd`, which is not under the
resolved workspace → **ask**. A symlink inside the workspace pointing to `/etc`
resolves to `/etc` → **ask**. Resolving *before* comparing is what makes the
containment real instead of cosmetic.

### Non-interactive runs deny, never block

`make_permission_callback` (`permissions.py:107`) defaults `interactive` to whether
stdin is a TTY. In a non-interactive run, ask-cases are **denied**, not blocked
(`permissions.py:134`). This means an unattended run *never hangs waiting for input*
and *never takes an unapproved outward action* — it just declines and moves on.
There's even a defense for a pseudo-TTY where `isatty()` lies: if reading the prompt
raises, it **fails closed → deny** (`permissions.py:142`).

### Two shell tools, gated identically

On Windows the agent has both `Bash` and `PowerShell`. `_SHELL_TOOLS` contains both
(`permissions.py:38`), so the same outward/destructive gating applies to whichever
shell the model picks. *Any new shell tool must be added there* or it would fall
through to the fail-closed unknown-tool branch.

> M3 mastery check: pure `classify()`; shells use a denylist because the safe set is
> infinite and the dangerous set finite; the unknown-tool fail-closed default is
> cheap because it fires rarely (dev-time) unlike the constant shell path;
> `_under()`/`.resolve()` defeats `../` traversal and symlink escape by resolving
> before comparing.

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
  via `W_IMP`. This is intentional and documented at `config.py:64`: importance
  should *both* slow forgetting *and* act as a direct relevance signal.

The per-kind prior (`KIND_RECALL_BOOST`, `config.py:76`) gives durable kinds
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
cross. Forgetting becomes recoverable; deletion is the rare last resort.

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
**injectable class** `MemoryStore(db_path)` (`store.py:91`); module-level
`save`/`recall`/… are back-compat shims over a lazily-built default instance bound to
`_DB_PATH` (`store.py:474`). `_DB_PATH` stays a reassignable global so tests can point
the default at an isolated database — and the default is rebuilt whenever it changes
(`store.py:477`).

### Schema and migrations

One table, `memories` (`store.py:139`), with the cognitive columns:
`importance`, `last_used_at`, `use_count`, `status`, `embedding` (a BLOB of packed
floats). Schema version is tracked via **`PRAGMA user_version`** (`store.py:120`) with
ordered migrations:

- `version == 0` covers three cases at once (fresh DB, a v1 store, or a v2 store that
  never had the pragma set): create the table if absent, idempotently add the v2
  columns (`_migrate_to_v2`, `store.py:158`), backfill `last_used_at` from
  `created_at`, and stamp the version.
- `_apply_migrations()` (`store.py:180`) is the ordered upgrade path for *future*
  versions — empty today, but the structure means a later schema change is a single
  once-applied step on long-lived stores.

### Two-stage recall (the core algorithm)

This is the single most important mechanism in the memory layer. `recall()`
(`store.py:391`) does **not** score the whole table. It works in two stages:

**Stage 1 — candidate generation (`_candidates`, `store.py:345`).** Pull a *bounded*
set of candidates (capped at `CANDIDATE_TOPN` = 50) using indexes:

- **Keyword candidates** via the **FTS5** full-text index, ordered by `bm25`
  relevance (`store.py:354`). Scales to large stores because it's an index lookup,
  not a scan.
- **Semantic candidates** via the pluggable **vector index**, but only rows whose
  cosine clears `SEM_CANDIDATE_THRESHOLD` (0.60) and aren't already in the keyword
  set (`store.py:373`).
- If FTS5 is unavailable, a **fallback keyword scan** over the whole table
  (`store.py:366`) — correct, just slower; fine for small stores.

**Stage 2 — fuse-rank only the candidates (`store.py:416`).** For each candidate
compute the full `fused_score`, drop anything under `RECALL_FLOOR` (0.12), sort, and
return the top `k`.

### The property that makes recall safe: Stage 1 *is* the relevance gate

Internalize this, because it's the linchpin connecting M5 to M6.

A candidate enters Stage 1 **only** if it has keyword overlap **or** strong semantic
similarity to the query (the explicit gate at `store.py:421`). Activation and
importance are **not consulted in Stage 1 at all.** They only act in Stage 2, where
they tie-break *among rows that already passed the relevance gate.*

The consequence: **a strong-but-irrelevant memory can never surface.** No matter how
high its activation or importance, if it doesn't match the query topically it never
enters the candidate pool, so it's never ranked, so it's never returned. This is the
"unrelated query → nothing" guarantee. And — crucial for M6 — it's *why
reinforcement-on-recall is safe*: reinforcement can only ever strengthen memories
that were *relevant enough to surface*, so the rich-get-richer loop is structurally
starved of irrelevant fuel.

### Reinforce-on-write (and on recall)

`save()` (`store.py:229`) does **not** blindly insert. If the exact text already
exists, it **reinforces** the existing row (refreshes recency, bumps `use_count`,
keeps the higher importance, reactivates it) and returns its id (`store.py:257`).
With embeddings on, it also catches **near-duplicate paraphrases** via
`_semantic_duplicate` above `SAVE_DEDUP_SIM` (0.93) and reinforces those instead of
cloning (`store.py:269`). Saving the same knowledge twice makes it *stronger*, not
*duplicated* — exactly like recalling it. `recall(..., reinforce=True)`
(`store.py:438`) applies the same logic: surfacing is a use.

### Soft-optional everything: FTS5, embeddings, ANN

Three capabilities are **soft-optional** — present them if available, degrade
gracefully if not, never a hard dependency:

- **FTS5** (`_init_fts`, `store.py:190`) is *feature-detected* by trying to create
  the virtual table; if the SQLite build lacks it, `_fts_ok` stays False and recall
  falls back to a keyword scan. *Gotcha:* for an external-content FTS5 table,
  `COUNT(*)` reads the **content table**, not the index — so it can't tell you the
  index is empty. The rebuild decision keys off whether the table is being created
  for the first time (a `sqlite_master` check, `store.py:192`), not a count.
- **Embeddings** (`embeddings.py`) use **`fastembed`** (ONNX, CPU, offline, *no API
  key* — because Max, not API). `available()`/`embed()`/`cosine()` all degrade to
  `None`/keyword+activation if the package is absent or disabled. **Tests force
  embeddings OFF** via an autouse fixture so the suite stays deterministic, except
  tests marked `@pytest.mark.semantic`. The lazy model build is double-checked-locked
  (`embeddings.py:32`) so it's constructed at most once.
- **ANN index** (`vector_index.py`) — see next.

### The vector index seam (`vector_index.py`)

Semantic candidate search hides behind a `VectorIndex` protocol (`vector_index.py:38`)
with two backends:

- **`BruteForceIndex`** — an exhaustive cosine scan over the `embedding` column
  (`vector_index.py:53`). Always correct, no extra storage, the default and the
  source of truth.
- **`SqliteVecIndex`** — an ANN `vec0` virtual table via the optional `sqlite-vec`
  extension (`vector_index.py:73`). For large stores.

The defining safety mechanism is `get_index()` (`vector_index.py:170`): it returns
the ANN backend **only after a runtime self-test (`_self_test`, `vector_index.py:142`)
proves a correct round-trip on *this machine*.** This catches the case that "import
succeeded" doesn't — an extension that **loads but misbehaves** (wrong build, broken
distance metric). If the self-test fails, it silently falls back to brute force. So a
broken extension can *never* corrupt recall; the worst case is "slower, still
correct." The `embedding` BLOB column stays the source of truth either way — the ANN
table is a synced accelerator used only to *prune* the candidate set, never to *rank*
(ranking re-computes exact cosine from the column), which makes it robust to
distance-metric quirks (`vector_index.py:78`).

### The service / client seams (`service.py`, `client.py`)

Two layers of indirection sit between consumers and the store, both forward-looking:

- **`MemoryService`** (`service.py:26`) — the in-process *facade* for long-term
  memory: `save`/`recall`/`forget`/`consolidate`/`dream`/stats. It resolves the
  default store on each call so it honors `_DB_PATH` reassignment. Scope is
  *deliberately long-term memory only* — skills, workflows, and events stay
  in-process and aren't part of this seam.
- **`MemoryClient`** (`client.py:24`) — the *consumer-facing* protocol. The MCP tools,
  the recall/episode hooks, and the CLI all go through `default_client()`
  (`client.py:67`), today a `LocalMemoryClient` making direct in-process calls.

Why two layers that currently do nothing but forward? **Forward compatibility.** When
memory becomes a standalone process, an `HttpMemoryClient` implementing the same
protocol drops in and **only the transport changes** — no consumer is touched. The
indirection is a no-op *today* precisely so the migration is a no-op *later*.

### Procedural memory: skills and workflows

Beyond facts, ReLife stores **procedures** as human-readable Markdown files (so
they're diffable and the consolidation pass can write them mechanically):

- **Skills** (`skills.py`) — a *single* reusable procedure ("how I scaffold a Python
  CLI"). Frontmatter `name` + `when_to_use`, then steps. Recall is keyword overlap
  over name+when+body with the **name weighted 2×** (`skills.py:94`).
- **Workflows** (`workflows.py`) — a *multi-step* ordered chain ("scaffold → test →
  repo → push"), same file format plus a `trigger` field. Same weighted keyword
  recall.

### The MCP surface (`server.py`)

`memory_server()` (`server.py:241`) exposes the tools the agent calls directly:
`memory_save` (now takes `importance`), `memory_recall`, `memory_forget`,
`skill_write`/`skill_find`, `workflow_save`/`workflow_find`, `memory_consolidate`, and
`memory_dream`. Surfaced under the `relife_memory` server → `mcp__relife_memory__*`,
which the trusted `mcp__relife` prefix auto-allows (no permission change). Memory is
shipped *as an MCP server even though it's in-process* so the agent-facing contract is
identical when it's later split out.

> M5 mastery check: two-stage recall = indexed candidates → fuse-rank only those;
> Stage 1 IS the relevance gate, so importance/activation only tie-break among
> already-relevant rows and can never float an irrelevant-but-strong memory into the
> prompt; the vector-index self-test catches "loads but misbehaves," not just
> "absent," and falls back to brute force rather than erroring; the service/client
> seams are no-op indirections today so the out-of-process split is a no-op later.

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

Three hooks, registered in `memory_hooks()` (`hooks.py:147`):

| Event | Function | Loop beat | Job |
|---|---|---|---|
| `UserPromptSubmit` | `_recall_hook` | **recall** | inject relevant memory+skills+workflows before the prompt |
| `PostToolUse` | `_event_hook` | (feeds consolidate) | journal every tool call |
| `Stop` | `_episode_hook` | **reflect** (involuntary) | capture a deterministic episode of the run |

The symmetry: `_recall_hook` injects at the **start**, `_episode_hook` captures at the
**end**, `_event_hook` records the **middle**. The hooks bracket the whole turn.

#### Hook 1 — `_recall_hook` (UserPromptSubmit)

Before the prompt reaches the model (`hooks.py:44`), it:

1. **Stashes the prompt** in `_last_prompt[session_id]` (`hooks.py:48`) — a breadcrumb
   so the *Stop* hook can later pair intent with approach.
2. **Gathers candidates** from three sources (`hooks.py:52`): top-5 memories
   (`recall(..., reinforce=True)`), top-2 skills, top-1 workflow. Each carries
   `(section_label, key_text, rendered)`.
3. **De-duplicates across sources and caps size** (`hooks.py:65`): `_is_dup` rejects a
   block whose token Jaccard against anything already kept is ≥ `RECALL_DEDUP_JACCARD`
   (0.8), and a running byte count is held under `RECALL_INJECT_BUDGET` (2400). It's a
   **cross-section greedy knapsack** with priority order memory → skill → workflow.
4. **Returns `additionalContext`** (`hooks.py:84`) — the SDK splices it into the
   model's context for *this* prompt only. Nothing survived → return `{}`.

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

Deliberately dead simple (`hooks.py:102`): after *every* tool call, write one row to
the event log — tool name, a 120-char `_brief` of what it did, tagged with
`session_id` as `task_id`. Two tells:

- Wrapped in `try/except … pass` (`hooks.py:108`) — **journaling must never break a
  run.** Observability is strictly subordinate to the task.
- It stores a **brief, not just the tool name.** Three `Bash` calls are useless as
  "Bash, Bash, Bash"; the *command* is what lets consolidation's `_action_label` later
  distinguish `git-clone` from `test` from `git-push`. The journal captures *actions*,
  not just *tool types*.

#### Hook 3 — `_episode_hook` (Stop)

On run end (`hooks.py:126`): pop the stashed prompt for the session; if none, bail.
Pull this task's events; if fewer than `EPISODE_MIN_EVENTS` (3) happened, bail (too
little to be worth remembering). Collapse the tool sequence (consecutive dups removed,
`hooks.py:136`). Save a deterministic one-liner
`"Task: <intent> | Approach: <tool → tool>"` as `kind="episode"` (`hooks.py:141`).

**Why stash the prompt at the *start*?** Each hook callback only receives *its own
event's* payload. At `Stop`, the user prompt is **not in scope** — it belonged to the
`UserPromptSubmit` event, which already fired and is gone. The only way the episode
hook can know the intent is to have captured it earlier. And `_last_prompt` is a
**dict keyed by session_id**, not a single string, so **concurrent sessions don't
clobber each other's prompts** (and `.pop` keeps it from leaking).

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

#### `consolidate.run_consolidation()` (the sleep) — `consolidate.py:321`

Runs automatically when enough events accrued (`should_auto_run`:
`events.count() − last ≥ CONSOLIDATE_EVERY`, gated by `AUTO_CONSOLIDATE`,
`consolidate.py:68`), invoked by `_maybe_consolidate()` after each run
(`agent.py:160`). Four deterministic steps:

1. **Decay & forget (`_decay_and_archive`, `consolidate.py:77`).** Tier 1: archive
   active memories that `should_archive`. Tier 2: hard-delete rows that are *already
   archived* and `should_hard_delete`. The guard at `consolidate.py:93` ensures nothing
   goes active→deleted in one pass — only already-archived rows are deletion
   candidates.
2. **Dedupe (`_dedupe`, `consolidate.py:107`).** Merge near-duplicates: keyword Jaccard
   ≥ 0.9, **or** (embeddings on) semantic cosine ≥ `DEDUP_SIM` (0.90), which catches
   paraphrases sharing few tokens. The survivor is *reinforced*; the duplicate
   *deleted* — merging concentrates strength rather than losing it.
3. **Detect patterns (`_detect_patterns`, `consolidate.py:271`).** Two detectors:
   recurring **episodes** → `pattern` memory; recurring **action n-grams** from the
   event log → `pattern` memory **+ a synthesized workflow**.
4. **Synthesize workflows** — turning the journal into replayable plans.

**The n-gram cleverness (the part to really understand).**
- N-grams run over **action labels, not raw tool names** (`_tool_ngrams`,
  `consolidate.py:252` + `_action_label`, `consolidate.py:191`). For shell tools the
  *action* is derived from the command, so three different `Bash` calls become
  `git-clone`, `test`, `git-push` — distinct nodes — instead of collapsing into one
  meaningless `Bash → Bash → Bash` that would hide the real workflow and leave only
  trivial editor motions visible.
- A **meaningfulness gate** (`_is_meaningful_seq`, `consolidate.py:238`) only promotes
  a sequence that contains at least one *distinctive* action (git/test/build/docker, an
  MCP tool, browsing…). Pure `Write → Edit` editor motions are a real regularity but
  *not a workflow*, so they're skipped, not turned into noise.
- **Maximal sequences win** (`consolidate.py:289` + `_contains`, `consolidate.py:245`):
  process longest-first and drop any sequence already contained in an accepted longer
  one. So you get `clone→test→push` once, not also its sub-sequences as separate
  workflows.

So a recurring real procedure becomes an *automatically* synthesized workflow, which
`_recall_hook` then surfaces next time a matching task appears. **The journal from 6a
feeds workflow synthesis in 6b, which feeds injection in 6a. The loop closes.**

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
5. **Confidence-gated.** Verdicts below `REM_MIN_CONFIDENCE` (0.7) are ignored
   (`rem.py:198`).
6. **Capped.** `REM_MAX_PRUNE_FRACTION` (0.25) bounds how much one pass can archive;
   intents are applied most-confident-first until the cap, then skipped
   (`rem.py:230`).
7. **Journaled.** Every applied action is appended to `data/rem_journal.jsonl` for
   audit/recovery (`rem.py:86`).
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

> M6 mastery check: three hooks make recall/journal/reflect involuntary infrastructure;
> reinforcement is safe because the relevance gate precedes the loop; the prompt is
> stashed because it's out of scope at Stop; the episode floor prevents boilerplate
> false patterns; fail-soft is correct only where failure costs data, not safety;
> consolidate must be LLM-free because it auto-runs on the shared Max budget; REM is
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
   `summary`. Writes are **atomic-ish** (temp file then `replace`, `ledger.py:113`) so a
   crash mid-write can't corrupt the ledger. `latest_for(workspace)` (`ledger.py:88`)
   finds the most-recently-updated ledger for a workspace (for `--resume` with no id).
2. **`build_server` (`server.py`)** — an in-process MCP server `relife_build` exposing
   three tools to the orchestrator, **bound to one ledger via closure** (`server.py:21`)
   so they mutate the right ledger on disk: `build_plan_set` (record the
   decomposition), `build_milestone_update` (set status + summary), `build_status`
   (read the ledger). Surfaces as `mcp__relife_build__*` → already auto-allowed by the
   trusted `mcp__relife` prefix (no permission change).
3. **The `builder` subagent (`agents.py`)** — an `AgentDefinition` the orchestrator
   delegates each milestone to **via the Task tool**. Its prompt (`agents.py:18`)
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

The builder's tool list (`agents.py:37`) is read/search/edit/shell + browser + memory,
but **not** the build-ledger tools — those belong to the orchestrator alone.

### `run_build()` — wiring and resume (`orchestrator.py:43`)

A fresh build: create a ledger, give the orchestrator the initial prompt ("decompose
into milestones with `build_plan_set`, then delegate each to a `builder`",
`orchestrator.py:24`). A resume: load the ledger (by id, or the latest for the
workspace), and give the resume prompt ("here's the ledger; call `build_status`,
spot-check that 'done' milestones really exist, continue from the first unfinished
one", `orchestrator.py:33`).

Three robustness decisions worth understanding:

- **Resume runs in the ledger's own workspace, not the CLI's cwd** (`orchestrator.py:69`).
  The build lives where it was created; resuming from any directory must `cwd` into the
  *ledger's* workspace, or the agent couldn't see its prior work and would build in the
  wrong place.
- **The `session_id` is persisted to continue the same conversation**
  (`orchestrator.py:116`): each `ResultMessage` with a session id is written to the
  ledger, so the next `--resume` continues the same CLI session.
- **Expired-session fallback** (`orchestrator.py:100`). A persisted `session_id` can be
  *gone* — it expires across a Max session-limit reset, and the `claude` subprocess
  then fails to start. So resume tries the saved session, and on `ClaudeSDKError` it
  **drops the stale handle and starts a FRESH session**, re-injecting the full ledger
  in the resume prompt — so no milestone progress is lost even though the conversation
  handle died. This directly reflects the Max constraint from M1.

### The CLI subtlety (`cli.py:73`)

`--resume` is a **boolean flag**, and the positional `spec` argument *doubles* as the
optional build id on resume (`cli.py:99`). This is deliberate: if `--resume` took a
value, it would swallow the following option (e.g. `--workspace`). As a boolean, you
write `relife build --resume <id> -w <path>` unambiguously, and `--resume` alone
resumes the most recent build for the workspace.

> M7 mastery check: decompose → delegate (each milestone to a fresh `builder` via Task)
> → resume (BuildLedger + persisted session_id); fresh contexts keep the orchestrator
> small so builds scale past one window; resume is robust to a dead session because the
> ledger re-injects state and falls back to a fresh session.

---

<a name="m8"></a>
## M8. Trade-offs, failure modes & "why not X"

This module is the synthesis: the recurring design *principles*, the deliberate
*sacrifices*, and the *weak points*.

### The recurring principles (the "house style")

1. **Pure core, thin imperative shell.** The hard logic is pure and deterministic
   (`cognitive.py`, `classify()`, `ledger.py`, the hook callbacks, the REM `_apply`),
   tested without a live agent. The SDK loop is a thin wrapper around injected pieces.
   *Trade-off:* more indirection and more seams to understand, in exchange for a system
   you can test cheaply and reason about — essential when the expensive path (live runs)
   burns Max budget.

2. **Soft-optional dependencies, never hard.** FTS5, embeddings (`fastembed`), and the
   ANN extension (`sqlite-vec`) all degrade gracefully if absent, disabled, or
   *misbehaving*. The pattern is always: feature-detect or self-test → use if it works →
   silently fall back to the always-correct path otherwise. *Trade-off:* the fast path
   isn't guaranteed, and there are more code paths; in exchange the system always runs
   correctly on a bare install and can never be *broken* by an optional component.

3. **Reversibility over destruction.** Forgetting is two-tier (archive, then much-later
   delete). REM's only destructive action is archive, never delete, and every action is
   journaled. *Trade-off:* the store carries archived dead weight longer and there's
   more state to manage; in exchange a bad decision (a faded-but-seasonal memory, a
   wrong critic verdict) is recoverable.

4. **Budget-awareness as a first principle (the Max constraint).** Anything automatic
   and frequent is LLM-free; the one LLM memory path is opt-in, capped, and gated. No
   hosted embeddings. *Trade-off:* the cheap deterministic passes can't do qualitative
   judgement (a deterministic dedup can't notice a contradiction); that capability is
   quarantined into the opt-in REM pass.

5. **Fail-closed for safety, fail-soft for data.** Permissions and the recall gate fail
   *closed* (deny / surface nothing) because their failure mode is *wrong action*. The
   journaling and episode hooks fail *soft* (`except: pass`) because their failure mode
   is *lost learning*, which is recoverable and must never break a task.

6. **Seams for a future split that hasn't happened yet.** Memory is an MCP server even
   in-process; `MemoryService`/`MemoryClient` are no-op indirections today. The cost is
   paid now (extra layers) so the migration is free later (swap the transport).

### The sharpest trade-offs (state these crisply)

- **LLM-free consolidation buys affordability at the cost of intelligence.** The
  constant pass can only do *mechanical* maintenance (decay, dedupe, n-gram mining). It
  cannot tell that a memory is wrong or contradictory. That intelligence exists only in
  the opt-in REM pass — so between dreams, the store can hold contradictions and
  garbage that only a human-triggered pass will catch.
- **Reinforcement-on-recall makes useful memories self-strengthening — and depends
  entirely on the relevance gate to not become a popularity contest.** The safety of the
  whole feedback loop rests on M5's Stage-1 gate being correct. Weaken that gate and the
  reinforcement loop turns pathological.
- **Subagent delegation keeps the orchestrator small at the cost of cross-milestone
  context.** A builder sees only its one milestone + what already exists in the
  workspace; it can't see *why* a sibling milestone made a choice except through the
  concise summary. Parallel milestones are deliberately deferred for the same reason —
  coordination across fresh contexts is hard.

### The biggest risks / weak points

1. **Deterministic pattern mining can mint junk into long-term memory.** The episode/
   n-gram detectors are heuristic (Jaccard clustering, action labels, a meaningfulness
   gate). The `EPISODE_MIN_EVENTS` floor and `_is_meaningful_seq` gate exist precisely
   because, without them, boilerplate would cluster into false patterns. They mitigate
   but don't eliminate the risk of a spurious `pattern`/workflow that then gets surfaced
   by recall.
2. **The store has no human-facing contradiction control between dreams.** If REM is
   never run, wrong or stale memories only fade *if* they go unused — but a wrong memory
   that keeps matching queries gets *reinforced*, not corrected. The corrective path is
   strictly opt-in.
3. **Live verification is expensive and rationed.** Because live runs consume Max
   budget, the team relies on the deterministic test suite and avoids hammering live
   runs (`CLAUDE.md`). That means some end-to-end behaviors are validated less often
   than unit-level logic.
4. **Windows-specific sharp edges.** Console encoding must be forced to UTF-8
   (`agent.py:37`) or Rich crashes on glyphs under cp1252; there are two shell tools
   (Bash + PowerShell) that must be gated identically; `gh` PATH injection is a
   workaround for a mid-session install. These are handled, but they're the kind of
   environment coupling that can resurface.

### "Why not X" — alternatives consciously rejected

- **Why not a metered API key?** It would untie ReLife from the Max subscription and
  add per-token cost, defeating the central bet and removing the constraint that makes
  the design coherent. (And it would invite "just call an LLM here" everywhere.)
- **Why not LLM-driven consolidation?** It auto-runs on the shared Max budget;
  productivity would accelerate budget drain and stall long sessions (M6). Quarantined
  into opt-in REM instead.
- **Why not hosted embeddings?** No API key available; semantic recall uses a local
  offline ONNX model that degrades to keyword+activation if even *that* is absent.
- **Why not delete faded memories directly?** Irreversible; destroys cyclical/seasonal
  memories. Two-tier archive→delete makes the first step recoverable.
- **Why not let the REM model archive/edit memory directly?** It's fallible; a bad or
  prompt-injected verdict could corrupt the store. So it's advisor-only — verdicts
  applied deterministically, reversibly, capped, gated, journaled; text edits forbidden
  entirely.
- **Why not an allowlist for shell commands?** The safe set is infinite and the
  dangerous set finite — you can only define the policy over the finite side, so:
  denylist the dangerous, allow the rest.
- **Why not parallel milestones in build?** Coordinating writes/decisions across fresh,
  independent contexts is hard and error-prone; deferred deliberately.

---

<a name="appendix-a"></a>
## Appendix A. The complete request lifecycle (one trace)

Follow a single `relife do "add a /health endpoint and push"` from keypress to
consolidation:

1. **CLI** (`cli.py:do`) resolves the workspace, builds `can_use_tool` bound to it,
   gets `default_mcp_servers()` (browser + memory) and `memory_hooks()`, and calls
   `run_task`.
2. **`build_options`** (`agent.py:64`) assembles `ClaudeAgentOptions`: model, effort,
   preset+persona system prompt, the permission callback, MCP servers, hooks,
   `setting_sources=None`, `env` with `gh` on PATH.
3. **Streaming client** starts (`agent.py:213`). `client.query(prompt)` sends the task.
4. **`UserPromptSubmit` → `_recall_hook`** fires first: stashes the prompt by session,
   recalls top memories/skills/workflows (reinforcing them), de-dups + budget-caps,
   and **injects** the surviving context. Now the model sees the task *plus* "you
   previously learned: this project pushes with `git push origin main`" etc.
5. **The model acts.** Each tool call hits **`classify()`**: `Read`/`Grep`/`Edit`
   (inside workspace) → allow; `Bash "pytest"` → allow (build/test); `Bash "git push"`
   → allow (authorized). If it tried `gh pr create`, the outward regex → **ask** (or
   deny if non-interactive).
6. **`PostToolUse` → `_event_hook`** journals each call (tool + brief + session id) to
   `events`. The `git push` is recorded as action `git-push`.
7. The model may call **`memory_save`/`skill_write`** explicitly (reflect).
8. **`Stop` → `_episode_hook`** pops the prompt, sees ≥3 events, collapses the tool
   sequence, and saves `"Task: add a /health endpoint and push | Approach: Read → Edit
   → test → git-push"` as an episode.
9. **`_maybe_consolidate`** (`agent.py:160`) runs if enough events accrued: fades
   unused memories (two-tier), merges duplicates, and mines n-grams — if
   `Read → Edit → test → git-push` has now recurred ≥3 times, it synthesizes a workflow
   `auto-read-edit-test-git-push` that **future `_recall_hook` calls will surface** for
   similar tasks. The loop has closed.
10. **Later, manually:** `relife dream` runs REM — the model reviews recent memories as
    an adversarial critic and reversibly prunes/reweights, journaling every action.

---

<a name="appendix-b"></a>
## Appendix B. File-by-file index

| File | Responsibility |
|---|---|
| `cli.py` | Typer CLI: `do`/`chat`/`build`/`consolidate`/`dream`/`memory stats`; wires permissions + MCP + hooks. |
| `agent.py` | `build_options`, `run_task`/`run_chat`, streaming loop, `ask_model_oneshot`, `_maybe_consolidate`, Windows UTF-8 fix. |
| `permissions.py` | `classify()` pure policy, `_OUTWARD_BASH` denylist, `_under()` containment, `make_permission_callback`. |
| `config.py` | All paths, model id (`claude-opus-4-8`), every cognitive/recall/REM tunable, MCP server defs, `agent_env`. |
| `hooks.py` | `_recall_hook` (inject+reinforce), `_event_hook` (journal), `_episode_hook` (capture), `memory_hooks()`. |
| `memory/cognitive.py` | Pure ACT-R math: `activation`, `sigmoid`, `fused_score`, `should_archive`, `should_hard_delete`. |
| `memory/store.py` | `MemoryStore`: schema/migrations, two-stage `recall`, reinforce-on-`save`, candidate gate. |
| `memory/vector_index.py` | `VectorIndex` protocol, `BruteForceIndex`, `SqliteVecIndex`, `get_index` self-test. |
| `memory/embeddings.py` | Soft-optional local ONNX embeddings; `available`/`embed`/`cosine`. |
| `memory/service.py` | `MemoryService` facade (long-term memory only): save/recall/forget/consolidate/dream/stats. |
| `memory/client.py` | `MemoryClient` protocol + `LocalMemoryClient` + `default_client()` (transport seam). |
| `memory/skills.py` | Single procedures as Markdown; weighted keyword `find_skills`. |
| `memory/workflows.py` | Multi-step procedures as Markdown (+`trigger`); weighted keyword `find_workflows`. |
| `memory/events.py` | `EventLog`: append-only tool journal, `events_by_task`, `count`. |
| `memory/consolidate.py` | Deterministic "sleep": decay/archive/delete, dedupe, n-gram mining, workflow synthesis. |
| `memory/rem.py` | Opt-in LLM "dream": replay buffer, critic prompt, deterministic reversible `_apply`. |
| `memory/server.py` | MCP `relife_memory`: the 9 memory/skill/workflow tools the agent calls. |
| `memory/_text.py` | Shared stopword tokenizer used by every keyword path. |
| `build/ledger.py` | `BuildLedger`: durable plan/progress, atomic writes, `latest_for`, `status_brief`. |
| `build/server.py` | MCP `relife_build`: `build_plan_set`/`build_milestone_update`/`build_status`, ledger-bound. |
| `build/agents.py` | `builder` `AgentDefinition` + tool list + orchestrator prompt path. |
| `build/orchestrator.py` | `run_build`: fresh vs resume, ledger wiring, session persistence + expired-session fallback. |

---

<a name="appendix-c"></a>
## Appendix C. Every tunable in `config.py`

**Model:** `MODEL = claude-opus-4-8`, `EFFORT = high`.

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
- `SEM_CANDIDATE_THRESHOLD = 0.60` — min cosine for a zero-keyword row to be a candidate.

**De-dup:** `DEDUP_SIM = 0.90` (consolidation merge), `SAVE_DEDUP_SIM = 0.93` (save-time
paraphrase reinforcement).

**Injection (hooks):** `RECALL_INJECT_BUDGET = 2400` chars, `RECALL_DEDUP_JACCARD = 0.8`.

**Episodes / consolidation:** `EPISODE_MIN_EVENTS = 3`, `RECUR_THRESHOLD = 3`,
`AUTO_CONSOLIDATE = on`, `CONSOLIDATE_EVERY = 5`.

**REM:** `REM_BATCH_MAX = 40`, `REM_REFERENCE_MAX = 30`, `REM_MIN_CONFIDENCE = 0.7`,
`REM_MAX_PRUNE_FRACTION = 0.25`. (No AUTO flag — never auto-runs by design.)

**Embeddings:** `EMBED_MODEL = BAAI/bge-small-en-v1.5`, `EMBEDDINGS_ENABLED = auto`
(auto|on|off).

Many are overridable via `RELIFE_*` environment variables (see `config.py`).

---

*End of module deep dive. For the interactive Socratic grilling, the tutoring session
picks up at M6b (sleep/dream split) and continues through M7–M8.*
