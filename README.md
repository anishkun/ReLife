# ReLife

A personal agent that acts on the world through MCP servers and **learns over time** —
accumulating facts and reusable skills so it gets better at recurring tasks.

Built on the [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk) (Python),
running on the **Claude Code Max subscription** (no metered API key).

## Status

**1.0** — release-tested end-to-end (see `RELEASE_TESTING.md` and `CHANGELOG.md`):

- **CLI** — `relife do "<task>"` (one-shot) and `relife chat` (interactive).
- **GitHub work items** — `relife work` lists the issues assigned to you; `relife work owner/repo#12`
  clones, branches, fixes, tests and pushes, then asks before opening the PR. Issue text is treated
  as untrusted data (a release test planted a prompt injection in an issue; it was ignored).
- **Agent loop** — Claude Opus 4.8 with the Claude Code coding preset + ReLife persona,
  full built-in toolset (read/write/edit/bash/web), streaming output.
- **Permissions** — auto-allow reading, in-workspace edits, build/test, and git
  (incl. push); ask before outward/destructive actions, writes or deletes outside the
  workspace, and package installs into your global environment (use a project `.venv`).
  Fails closed when unattended. This is a policy against *accidental* outward actions,
  not a sandbox — code the agent writes and runs can do anything your user account can.
- **MCP** — Playwright **browser** server (navigate/read/click) + an in-process
  **memory/skills** server. GitHub via `gh` (build → commit → create repo → push).
- **Email / calendar / files** — the claude.ai **Gmail, Google Calendar and Google Drive**
  connectors ride the subscription (no keys, no setup beyond enabling them at claude.ai).
  Reads run on their own; anything that changes something asks you first, in the
  terminal or the web console. `relife doctor` shows whether they're enabled. The Gmail
  connector can send, reply and forward as well as draft — every one of those asks unless a
  schedule pre-approved it — and what actually goes through also depends on the Google
  permissions you granted when linking the account.
- **Memory that grows (like a brain)** — facts/skills/workflows recalled automatically
  before each task; relevance **rises with use and fades when ignored**; an automatic
  LLM-free **"sleep" pass** forgets stale notes, merges duplicates, and learns workflows
  from repeated actions; and an opt-in, AI-driven **"dream" pass** (`relife dream`)
  reversibly critiques and tidies memory when you have budget to spare. What it learned is
  inspectable and correctable (`relife memory search|list|show|forget`).
- **Always-on** — `relife serve` runs a long-lived agent server with a self-contained web
  console (streaming transcript, approval cards for outward actions) and **schedules**:
  tasks that fire on a cadence in their own session, with each run's outcome — summary,
  cost, and anything it needed you for — recorded durably. A **work schedule** runs
  `relife work` on your newest assigned issue each time; a schedule may carry narrow
  pre-approvals (email / create calendar events to listed addresses, or open the
  one PR for the issue it worked) so an unattended run can finish.
- **Memory as a service (optional)** — `relife memory serve` runs the memory layer as a
  standalone daemon so several ReLife processes share one brain (`RELIFE_MEMORY_URL`).
- **`relife doctor`** — checks the CLI login, Node, `gh`, FTS5, extras, connectors, the memory
  daemon, the agent server and whether your schedules will actually fire.

**Since 1.0 (unreleased — see `CHANGELOG.md`).** Tested deterministically (including a real
CrewAI run with stand-in models) but not yet live-tested with real model calls:

- **Many agents, handed-down memory** — register agents (`relife agent create`); each writes its
  own memory space and reads what it **inherited** from older agents (live, read-only), can start
  from a **fork** (snapshot copy), and only reaches your main memory when you **promote** it.
  Memory moves between installs as export/import packs.
- **Memory for any LLM** — ReLife memory is an MCP server: `relife mcp --agent NAME` (stdio) or
  `/mcp` on the memory daemon (per-agent token) attaches it to Claude Desktop, Cursor, Gemini CLI,
  a CrewAI or LangGraph app — any MCP client — scoped to that agent.
- **Crews (CrewAI)** — `relife crew "<task>"`: a planner (Claude via your subscription) designs a
  small team, CrewAI runs it, and ReLife staffs it with full ReLife agents (Claude + tools +
  memory) and, if you configure them, CrewAI agents on other models (Ollama, OpenAI, Gemini…) with
  ReLife memory attached. New team members inherit what experienced agents learned.

## Setup

Requires Python ≥ 3.11, Node.js (for MCP servers), and a logged-in Claude Code CLI
(`claude`) on a Max subscription.

```sh
pip install -e .                 # core
pip install -e ".[server]"       # + `relife serve` (agent server + web console)
pip install -e ".[daemon]"       # + `relife memory serve` (memory as a standalone daemon)
pip install -e ".[embeddings]"   # + local semantic recall (fastembed, offline, no API key)
pip install -e ".[vector]"       # + sqlite-vec index for very large memory stores
pip install -e ".[crewai]"       # + `relife crew` (CrewAI; needs Python <= 3.13, see below)
```

CrewAI doesn't support Python 3.14 yet. For crews, run ReLife from a 3.12/3.13 venv
(`py -3.12 -m venv .venv` then `.venv\Scripts\pip install -e ".[crewai]"`); on 3.14 the extra
installs nothing and `relife doctor` says so. Everything else works on 3.11–3.14.

Where state lives: in a source checkout, `data/` (memory, schedules, run records) and
`workspace/` sit next to the code; an installed (non-editable) copy uses `~/.relife/`.
Set `RELIFE_HOME` to put them anywhere else.

The web console binds `127.0.0.1` and, without a token, only answers requests addressed to
`localhost`/`127.0.0.1`. Set `RELIFE_AGENT_TOKEN` to require a token (mandatory for any
non-loopback bind — the server refuses otherwise). There is no TLS: put a reverse proxy in
front before exposing it beyond your machine.

Use the interpreter ReLife is installed into for everything below (if plain `python` lacks
`claude_agent_sdk`, try `py -3`).

## Usage

```sh
relife do "scaffold a Python CLI that prints the weather for a city"
relife chat
relife work                # your open assigned GitHub issues; `relife work owner/repo#12` to work one
relife build "<spec>"      # large, multi-milestone builds (resumable: --resume)
relife serve               # always-on agent server + web console on http://127.0.0.1:8600
relife doctor              # check the environment first (CLI login, node, gh, connectors, server…)
relife consolidate         # run the LLM-free memory "sleep" pass now
relife dream               # opt-in AI deep review of memory (spends Max budget)
relife memory stats        # what's remembered and what has faded
relife memory search "<q>" # what the agent would be shown for a query (doesn't reinforce)
relife memory list         # everything it has learned (--kind, --archived, --sort strong)
relife memory show <id>    # one memory in full
relife memory forget <id>  # archive a wrong or stale memory (reversible; --query works too)
relife memory serve        # optional: run memory as a standalone daemon (then set RELIFE_MEMORY_URL)
relife memory ping         # is the daemon up?
relife memory spaces       # memory per agent: what each space holds and who reads it
relife memory export SPACE -o pack.json   # portable pack; `relife memory import pack.json --space S`

relife agent create coder --inherit veteran   # a new agent that reads veteran's memory (read-only)
relife agent create alt --fork coder          # a new agent starting from a copy of coder's memory
relife agent list | show NAME | attach NAME OTHER | promote NAME | delete NAME
relife do "<task>" --agent coder              # run as that agent (its own memory space)

relife agent create cursor --runtime external --isolated   # prints the MCP config to paste
relife mcp --agent cursor                     # the stdio MCP server that config launches
relife agent token cursor                     # token for the daemon's HTTP /mcp endpoint

relife crew "<task>"       # CrewAI plans a team, shows it, runs it on confirm (--plan-only, --yes, --spec)
relife crews [ID]          # recent crew runs, or one run's per-task outcomes
```

`do`/`chat`/`build` accept `--workspace PATH` (default: `./workspace`) — the directory
the agent works in.

See **`HOW_IT_WORKS.md`** for a friendly, top-to-bottom walkthrough of the whole system, and
**`MODULE_DEEP_DIVE.md`** for the architecture module by module, with the reasoning behind it.

## Agents, memory handoff and crews

Every agent has a **memory space**. The main agent (plain `relife do`) uses `default` — your
memory. A registered agent *writes* only its own space and *reads* its own, the spaces it
inherited, and `default` (unless `--isolated`). The tools never take a space argument: scope
comes from who the agent is, so no model output can widen it. Consolidation dedupes and learns
workflows per space, never across.

- **inherit** — a new agent reads an older one's memory live, read-only (lineage is transitive);
- **fork** — it starts from a snapshot copy of the older agent's memories, skills and workflows;
- **promote** — you copy what an agent learned into your main memory (`relife agent promote`);
- **packs** — `relife memory export|import` moves a space between machines (provenance kept).

**Other LLMs.** Any MCP client attaches ReLife memory as a registered agent. Local (stdio):

```json
{"mcpServers": {"relife-memory": {"command": "<python>", "args": ["-m", "relife", "mcp", "--agent", "cursor"],
                                  "env": {"RELIFE_HOME": "<your relife home>"}}}}
```

Remote: run `relife memory serve` and point the client at `http://127.0.0.1:8787/mcp` with
`Authorization: Bearer <token from relife agent token NAME>`. `relife agent connect NAME` prints
both. A CrewAI app outside ReLife uses `MCPServerStdio(command=…, args=["-m", "relife", "mcp",
"--agent", NAME])` the same way.

**Crews.** `relife crew "<task>"` plans the team (one tool-less Claude call), shows it, and runs it
in `<workspace>/crews/<id>/` after you confirm. ReLife agents work under the normal permission
policy (outward actions ask). Agents on other models get ReLife memory but **no** shell/file tools —
list the models the planner may use in `RELIFE_CREW_LLMS` (e.g. `ollama/llama3.1,gpt-4.1`; their
keys, if any, are read by CrewAI/LiteLLM from your environment — ReLife never stores them);
`claude-max` (Claude through your subscription) is always available. Each run's plan and per-task
outcome is kept in `data/crews/<id>/record.json` (`relife crews ID`). CrewAI's own memory,
planning and telemetry are off: ReLife's memory is the memory.

## Layout

```
relife/
  cli.py        CLI entry (Typer): do / chat / build / serve / doctor / consolidate / dream / memory * /
                agent * / mcp / crew / crews
  agents.py     agent registry: identity, memory scope, inherit / fork / promote, MCP tokens
  agent.py      builds ClaudeAgentOptions, runs the SDK loop, the shared event taxonomy
  config.py     model, paths, permission mode, MCP servers, every tunable and env knob
  permissions.py allow/ask policy gating every tool call (shell, files, connectors)
  hooks.py      auto-recall before each prompt, tool journaling, episode capture on Stop
  workitems.py  `relife work`: GitHub issue plumbing (find, fetch, clone, branch, task prompt)
  doctor.py     `relife doctor`: environment + always-on checks, each with a fix
  prompts/      system prompt (persona + safety rules) + the REM critic prompt
  web/          the self-contained web console served by `relife serve`
  memory/       cognitive memory + skills/workflows, partitioned into per-agent spaces
    tools.py    the memory tools, defined once (in-process SDK server + standalone MCP server)
    mcp_server.py ReLife memory over MCP for any agent (stdio + the daemon's /mcp)
    remote/     the optional memory daemon (FastAPI) + its HTTP client
  crew/         `relife crew`: CrewAI planner/adapter, ReLife agents as crew members, run records
  build/        orchestrated, resumable large builds (decompose → delegate → resume)
  server/       always-on agent server: sessions, SSE, approvals, security, scheduler, run outcomes
data/           runtime db, agents, schedules, run and crew records, logs (gitignored)
tests/          ~800 tests (no live model calls; CI runs them on Windows + Linux, py3.11-3.13)
```
