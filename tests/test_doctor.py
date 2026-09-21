"""Tests for `relife doctor` — run_checks() over scripted Probes (no subprocess,
no filesystem, no network)."""

from __future__ import annotations

from pathlib import Path

from relife.doctor import Check, Probes, find_claude_cli, parse_mcp_list, run_checks, worst

MCP_LIST = """Checking MCP server health…

claude.ai Google Calendar: https://calendarmcp.googleapis.com/mcp/v1 - ✔ Connected
claude.ai Google Drive: https://drivemcp.googleapis.com/mcp/v1 - ✔ Connected
claude.ai Gmail: https://gmailmcp.googleapis.com/mcp/v1 - ✔ Connected
"""

AUTH_OK = '{"loggedIn": true, "authMethod": "claude.ai", "email": "u@x.com", "subscriptionType": "max"}'


def healthy(**overrides) -> Probes:
    """A machine where everything is in order; tests knock one thing out."""
    tools = {"claude": "/bin/claude", "node": "/bin/node", "npx": "/bin/npx", "gh": "/bin/gh"}
    outputs = {
        ("/bin/claude", "--version"): (0, "2.1.183 (Claude Code)\n"),
        ("/bin/claude", "auth", "status"): (0, AUTH_OK),
        ("/bin/claude", "mcp", "list"): (0, MCP_LIST),
        ("/bin/node", "--version"): (0, "v22.0.0\n"),
        ("/bin/gh", "auth", "status"): (0, "Logged in to github.com account anishkun (keyring)\n"),
    }

    def run(cmd):
        return outputs.get(tuple(cmd), (1, f"no script for {cmd}"))

    base = dict(
        which=lambda name: tools.get(name),
        run=run,
        env={},
        python_version=(3, 12, 1),
        import_ok=lambda mod: True,
        bundled_cli=None,
        fts5_ok=lambda: True,
        data_dir=Path("/data"),
        data_dir_writable=lambda p: True,
        memory_url=None,
        http_get=lambda url: (200, "{}"),
    )
    base.update(overrides)
    p = Probes(**base)
    p._outputs = outputs  # type: ignore[attr-defined]  # let tests tweak scripts
    return p


def by_name(checks: list[Check]) -> dict[str, Check]:
    return {c.name: c for c in checks}


def test_healthy_machine_is_all_ok():
    checks = run_checks(healthy())
    assert worst(checks) == "ok"
    names = by_name(checks)
    assert names["claude login"].detail.startswith("u@x.com")
    assert names["connector gmail"].status == "ok"
    assert "approval" in names["connector gmail"].detail


def test_bundled_cli_preferred_over_path_like_the_sdk():
    bundled = Path("/site/claude_agent_sdk/_bundled/claude")
    assert find_claude_cli(healthy(bundled_cli=bundled)) == str(bundled)
    assert find_claude_cli(healthy()) == "/bin/claude"


def test_missing_cli_fails_and_skips_dependents():
    p = healthy(which=lambda name: None if name == "claude" else "/bin/" + name)
    names = by_name(run_checks(p))
    assert names["claude cli"].status == "fail"
    assert "pip install" in names["claude cli"].fix
    assert names["claude login"].status == "skip"
    assert names["connectors"].status == "skip"


def test_logged_out_is_a_fail_with_a_fix():
    p = healthy()
    p._outputs[("/bin/claude", "auth", "status")] = (1, '{"loggedIn": false}')
    names = by_name(run_checks(p))
    assert names["claude login"].status == "fail"
    assert "log in" in names["claude login"].fix
    assert names["connectors"].status == "skip"  # can't list connectors logged out


def test_api_key_in_env_warns():
    names = by_name(run_checks(healthy(env={"ANTHROPIC_API_KEY": "sk-ant-x"})))
    assert names["api key"].status == "warn"
    assert "unset" in names["api key"].fix


def test_missing_node_fails_missing_gh_only_warns():
    p = healthy(which=lambda n: None if n in {"node", "npx", "gh"} else "/bin/" + n)
    names = by_name(run_checks(p))
    assert names["node / npx"].status == "fail"
    assert names["github cli"].status == "warn"
    assert worst(run_checks(p)) == "fail"


def test_gh_found_via_extra_path_dirs(tmp_path):
    """gh installed mid-session (winget) isn't on the parent PATH; agent_env()
    prepends a known dir, so doctor looks there too."""
    (tmp_path / "gh.exe").write_text("")
    p = healthy(which=lambda n: None if n == "gh" else "/bin/" + n, extra_path_dirs=[str(tmp_path)])
    p._outputs[(str(tmp_path / "gh.exe"), "auth", "status")] = (0, "account anishkun")
    assert by_name(run_checks(p))["github cli"].status == "ok"


def test_gh_unauthenticated_warns():
    p = healthy()
    p._outputs[("/bin/gh", "auth", "status")] = (1, "You are not logged into any GitHub hosts")
    assert by_name(run_checks(p))["github cli"].status == "warn"


def test_old_python_fails():
    assert by_name(run_checks(healthy(python_version=(3, 10, 4))))["python"].status == "fail"


def test_no_fts5_is_a_warning_not_a_failure():
    names = by_name(run_checks(healthy(fts5_ok=lambda: False)))
    assert names["sqlite fts5"].status == "warn"


def test_unwritable_data_dir_fails():
    assert by_name(run_checks(healthy(data_dir_writable=lambda p: False)))["data dir"].status == "fail"


def test_missing_extras_are_skips_with_install_hint():
    names = by_name(run_checks(healthy(import_ok=lambda m: m != "fastembed")))
    assert names["extra [embeddings]"].status == "skip"
    assert names["extra [embeddings]"].fix == 'pip install -e ".[embeddings]"'
    assert names["extra [server]"].status == "ok"


def test_memory_daemon_checked_only_when_configured():
    assert by_name(run_checks(healthy()))["memory daemon"].status == "skip"

    ok = healthy(memory_url="http://127.0.0.1:8787", http_get=lambda u: (200, "{}"))
    assert by_name(run_checks(ok))["memory daemon"].status == "ok"

    def down(url):
        raise ConnectionError("refused")

    bad = healthy(memory_url="http://127.0.0.1:8787", http_get=down)
    c = by_name(run_checks(bad))["memory daemon"]
    assert c.status == "fail" and "relife memory serve" in c.fix


def test_connector_missing_from_account_warns_with_where_to_enable():
    p = healthy()
    p._outputs[("/bin/claude", "mcp", "list")] = (0, MCP_LIST.replace(
        "claude.ai Gmail: https://gmailmcp.googleapis.com/mcp/v1 - ✔ Connected\n", ""))
    names = by_name(run_checks(p))
    assert names["connector gmail"].status == "warn"
    assert "Connectors" in names["connector gmail"].fix
    assert names["connector google drive"].status == "ok"


def test_connector_not_connected_warns():
    p = healthy()
    p._outputs[("/bin/claude", "mcp", "list")] = (0, MCP_LIST.replace(
        "gmailmcp.googleapis.com/mcp/v1 - ✔ Connected", "gmailmcp.googleapis.com/mcp/v1 - ✘ Failed to connect"))
    assert by_name(run_checks(p))["connector gmail"].status == "warn"


def test_parse_mcp_list():
    servers = parse_mcp_list(MCP_LIST)
    assert servers == {
        "claude.ai Google Calendar": "✔ Connected",
        "claude.ai Google Drive": "✔ Connected",
        "claude.ai Gmail": "✔ Connected",
    }


def test_worst_orders_fail_over_warn_over_ok():
    assert worst([Check("a", "ok", ""), Check("b", "skip", "")]) == "ok"
    assert worst([Check("a", "ok", ""), Check("b", "warn", "")]) == "warn"
    assert worst([Check("a", "warn", ""), Check("b", "fail", "")]) == "fail"


# --- the always-on side -------------------------------------------------------
def _refused(url):
    raise ConnectionRefusedError("refused")


def _sched(name, status=None, denied=0, enabled=True):
    entry = {"at": 1.0, "status": status} if status else None
    if entry and denied:
        entry["denied"] = denied
    return {"name": name, "enabled": enabled, "last_status": status, "runs": [entry] if entry else []}


def test_model_is_reported_with_its_source():
    assert by_name(run_checks(healthy(model="claude-opus-4-8", effort="high")))["model"].detail == "claude-opus-4-8 · effort high"
    c = by_name(run_checks(healthy(model="claude-fable-5", effort="max", env={"RELIFE_MODEL": "claude-fable-5"})))["model"]
    assert c.status == "ok" and c.detail.endswith("(RELIFE_MODEL)")


def test_agent_server_not_running_is_a_skip_unless_schedules_exist():
    quiet = by_name(run_checks(healthy(http_get=_refused)))["agent server"]
    assert quiet.status == "skip" and "relife serve" in quiet.detail

    with_sched = by_name(run_checks(healthy(http_get=_refused, schedules=[_sched("inbox"), _sched("off", enabled=False)])))
    c = with_sched["agent server"]
    assert c.status == "warn" and "1 enabled schedule(s) will not fire" in c.detail and "relife serve" in c.fix


def test_agent_server_running_reports_health():
    body = '{"status":"ok","sessions":2,"schedules":3,"auth_required":true}'
    c = by_name(run_checks(healthy(http_get=lambda url: (200, body))))["agent server"]
    assert c.status == "ok" and "2 session(s), 3 schedule(s)" in c.detail and "auth on" in c.detail
    bad = by_name(run_checks(healthy(http_get=lambda url: (500, ""))))["agent server"]
    assert bad.status == "fail"


def test_agent_server_bind_without_token_fails_like_serve_would():
    c = by_name(run_checks(healthy(agent_host="0.0.0.0", agent_token=None)))["agent server"]
    assert c.status == "fail" and "RELIFE_AGENT_TOKEN" in c.fix
    # With a token it's allowed but flagged (no TLS), and a short token is called out.
    seen = {}
    def get(url):
        seen["url"] = url
        return 200, "{}"
    c = by_name(run_checks(healthy(agent_host="0.0.0.0", agent_token="abc", http_get=get)))["agent server"]
    assert c.status == "warn" and "short" in c.detail and "reverse proxy" in c.detail
    assert seen["url"] == "http://127.0.0.1:8600/health"  # 0.0.0.0 is probed via loopback


def test_schedules_summarise_last_outcomes():
    assert by_name(run_checks(healthy()))["schedules"].status == "skip"
    fine = by_name(run_checks(healthy(schedules=[_sched("a", "done"), _sched("b", enabled=False)])))["schedules"]
    assert fine.status == "ok" and fine.detail == "1 enabled, 1 paused — last runs fine"
    fresh = by_name(run_checks(healthy(schedules=[_sched("a")])))["schedules"]
    assert fresh.status == "ok" and fresh.detail.endswith("none run yet")

    c = by_name(run_checks(healthy(schedules=[
        _sched("inbox", "done", denied=2),
        _sched("nightly", "timeout"),
        _sched("weekly", "skipped: at most 8 concurrent sessions"),
        _sched("paused-broken", "error: x", enabled=False),  # paused: not nagged about
    ])))["schedules"]
    assert c.status == "warn"
    assert "last run failed: nightly (timeout), weekly (skipped: at most 8 concurrent sessions)" in c.detail
    assert "needed you: inbox (2 denied)" in c.detail
    assert "paused-broken" not in c.detail
    assert "runs" in c.fix


def test_workspace_root_must_be_writable():
    ok = by_name(run_checks(healthy(workspace_root=Path("/ws"))))["workspace root"]
    assert ok.status == "ok" and "confined" in ok.detail
    bad = by_name(run_checks(healthy(workspace_root=Path("/ws"), data_dir_writable=lambda p: p != Path("/ws"))))
    assert bad["workspace root"].status == "fail" and bad["data dir"].status == "ok"


def test_stray_daemon_sidecar_warns_when_memory_is_in_process():
    c = by_name(run_checks(healthy(sidecar_exists=True)))["memory daemon"]
    assert c.status == "warn" and "RELIFE_MEMORY_URL" in c.fix
    # Configured daemon + sidecar is the normal case.
    c2 = by_name(run_checks(healthy(sidecar_exists=True, memory_url="http://127.0.0.1:8787")))["memory daemon"]
    assert c2.status == "ok"


def test_to_json_shape():
    from relife.doctor import to_json

    out = to_json(run_checks(healthy(http_get=_refused)))
    assert out["overall"] == "ok"
    assert {"name", "status", "detail", "fix"} <= set(out["checks"][0])
    assert any(c["name"] == "agent server" and c["status"] == "skip" for c in out["checks"])


def test_connector_needing_auth_says_to_link_in_session_not_re_enable():
    p = healthy()
    p._outputs[("/bin/claude", "mcp", "list")] = (0, MCP_LIST.replace(
        "claude.ai Google Calendar: https://calendarmcp.googleapis.com/mcp/v1 - ✔ Connected",
        "claude.ai Google Calendar: https://calendarmcp.googleapis.com/mcp/v1 - ! Needs authentication",
    ))
    c = by_name(run_checks(p))["connector google calendar"]
    assert c.status == "warn" and "isn't linked" in c.detail
    assert "connect my calendar" in c.fix and "re-enable" not in c.fix
