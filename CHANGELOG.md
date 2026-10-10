# Changelog

## Unreleased

### Added
- **Memory spaces and agents.** Memory is partitioned per agent (store schema v3: `space`/`source`
  on memories, `space` on events; per-space skills/workflows). `relife agent create|list|show|attach|
  detach|promote|token|connect|delete` registers agents (`data/agents.json`). An agent writes only its
  own space and reads what it inherited (live, read-only, transitive) plus your default memory unless
  `--isolated`; `--fork` starts from a snapshot copy; `promote` is the explicit way into `default`.
  `relife memory spaces|export|import` (portable packs; imports marked `import:<space>`), `--space`
  on `search`/`list`/`forget`, `relife do|chat --agent NAME`.
- **ReLife memory over MCP, for any LLM.** The memory tools are defined once (`memory/tools.py`) and
  served both in-process (unchanged for Claude) and by a standalone MCP server: `relife mcp --agent
  NAME` (stdio) and `/mcp` on the memory daemon (streamable HTTP, per-agent bearer token, DNS-rebinding
  protection). New `memory_context` tool returns the recall block for agents without a hook.
  `python -m relife` works.
- **Crews (CrewAI).** `relife crew "<task>"` plans a small team (Claude via the CLI — no API key),
  shows it, and runs it with CrewAI after you confirm: full ReLife agents as crew members
  (`ReLifeAgent`, a `BaseAgentAdapter`) and CrewAI agents on other models (`RELIFE_CREW_LLMS`, or
  `claude-max`) with ReLife memory attached and no machine-touching tools. New agents inherit from
  experienced ones; outcomes are recorded in `data/crews/<id>/` (`relife crews [ID]`). Optional
  `[crewai]` extra — CrewAI needs Python ≤ 3.13; `relife doctor` explains.

### Changed
- Consolidation dedupes and learns patterns/workflows within each space, never across.
- The recall block labels memory from another space `via <space>`, and the agent's system prompt
  says what that label means (another agent's memory: background, not instructions).
- Docs brought up to date: `HOW_IT_WORKS.md` and `MODULE_DEEP_DIVE.md` cover work items, grants,
  agents/spaces, memory over MCP and crews (deep dive now M1–M16); `PROJECT_CONTEXT.md`'s overview
  sections refreshed. Correction to the 1.0.0 notes: the claude.ai Gmail connector **can send**
  (`send_message`, `reply`, `forward`), not only draft — every send asks unless a schedule's email
  grant covers it.

### Security
- Email grants no longer cover `send_message` with a `draftId` (sends a stored draft whose
  recipients the grant never saw) or `reply` with `replyAll` (keeps the thread's CC list) — both
  were pre-approved when `to` named a listed address. Found against the live Gmail connector schema.
- `rm -rf /d` (and `cp x /d`, `mv x /c`) auto-allowed: a single-letter `/x` token was read as a
  cmd switch (`del /s /q`) for every verb, but to `rm`/`cp`/`mv`/`tee`… in Git Bash it is the root
  of a drive. Slash switches are now only recognised for cmd built-ins.

### Fixed
- A fresh memory DB stamped the current schema version before later migration steps ran; it now
  stamps v2 and applies each step in order.
- Grants checked against the real Gmail/Calendar connector schemas: camelCase content fields
  (`htmlBody`, `forwardText`) no longer read as recipients; a non-address calendar attendee fails
  the grant; `get_draft` and `suggest_time` are reads.
- The `Bash` tool on Windows (Git Bash) spells `D:\relife\workspace` as `/d/relife/workspace`;
  shell writes/deletes to that form inside the workspace no longer ask (PowerShell is unchanged).

## 1.0.0 — 2026-10-08

First release. A personal agent on the Claude Agent SDK, running on the Claude Code Max
subscription (no API key), that acts through tools and MCP servers and learns over time.

### Features
- **Agent CLI** — `relife do` (one-shot) and `relife chat` (interactive), Claude Code preset +
  ReLife persona, streaming output, a workspace the agent works in (`--workspace`).
- **Orchestrated builds** — `relife build "<spec>"` decomposes a large project into milestones,
  delegates each to a fresh-context builder, and resumes after a crash (`--resume`).
- **GitHub work items** — `relife work` lists your assigned issues; `relife work owner/repo#N`
  clones, branches, fixes, tests and pushes, then asks before opening the PR. Issue text is
  fenced as untrusted data.
- **Cognitive memory** — facts, skills and workflows recalled automatically before each prompt;
  relevance rises with use and decays when ignored; an LLM-free "sleep" pass (decay, dedupe,
  workflow learning) runs automatically; an opt-in "dream" pass (`relife dream`) lets the model
  critique memory, reversibly and journaled. Inspect and correct with `relife memory …`.
  Optional local semantic recall (`[embeddings]`), sqlite-vec index (`[vector]`), and a
  standalone memory daemon (`[daemon]`, `RELIFE_MEMORY_URL`).
- **Always-on server** — `relife serve`: persistent sessions, a self-contained web console with
  approval cards, schedules (`every 30m` / `daily at 09:00`), work schedules that take your newest
  assigned issue, durable run records, and narrow per-schedule pre-approvals (draft email /
  calendar events to listed addresses, or the one PR for the issue a run worked).
- **Connectors** — claude.ai Gmail, Google Calendar and Google Drive ride the subscription;
  reads run on their own, changes ask. (Gmail can draft, not send.)
- **`relife doctor`** — CLI login, model, Node, `gh`, FTS5, extras, daemon, server, schedules,
  connectors; exits 1 on a blocker.

### Safety model
- Read-only tools, in-workspace edits, build/test and git (incl. push) are autonomous.
  Outward or destructive actions, writes/deletes outside the workspace (via the file tools
  *and* the shell), global package installs, privilege escalation, persistence, unknown tools
  and every connector write ask. Unattended runs deny anything that asks.
- `gh` is verb-based and fail-closed; connectors are verb-based and fail-closed.
- The server: token via bearer or HttpOnly SameSite=Strict cookie, same-origin checks on
  mutating routes, token-guess throttling, loopback-only Host when tokenless (DNS rebinding),
  refuses a non-loopback bind without a token, workspaces confined to a root, every queue and
  session count bounded.
- Known limit: the shell gate prevents accidents, it is not a sandbox — inline interpreter code
  or a script the agent writes can do anything the user can.

### Release hardening (since the last MVP pass)
- Installed wheels keep state in `~/.relife` (or `RELIFE_HOME`), never in `site-packages`.
- DNS-rebinding guard for the tokenless server.
- Shell policy closed 56 adversarial bypasses found by a new corpus (`tests/test_permissions_corpus.py`).
- Global package installs ask; the agent is told to use a project `.venv`.
- Corrupt `schedules.json` is preserved and reported instead of silently overwritten.
- Recall: a single generic shared word no longer surfaces (and reinforces) unrelated memories;
  the generic edit→test→commit loop no longer becomes a learned workflow.
- Approval cards show the recipient of an email / attendees of an event.
- Friendlier CLI errors (daemon down, corrupt DB); tracebacks no longer print local variables.
- Web console: decided approval cards drop their buttons; the schedules poll no longer swaps
  buttons under the cursor.
- CI on Windows + Linux × Python 3.11–3.13 (base and all extras), wheel smoke test, pip-audit;
  648 tests; `RELEASE_TESTING.md` checklist.
