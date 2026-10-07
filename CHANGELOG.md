# Changelog

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
