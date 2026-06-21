"""Tests for the memory service seam and the preserved MCP tool contract."""

from relife.memory import server
from relife.memory.client import LocalMemoryClient
from relife.memory.service import MemoryService
from relife.memory.store import MemoryStore


def _client(tmp_path) -> LocalMemoryClient:
    # Bind the service to an isolated store (no global state touched).
    return LocalMemoryClient(MemoryService(MemoryStore(tmp_path / "svc.db")))


def test_client_save_recall_roundtrip(tmp_path):
    c = _client(tmp_path)
    c.save("The user prefers ruff and pytest.", kind="preference", tags="python")
    hits = c.recall("what python tools?")
    assert hits and "ruff" in hits[0].text.lower()
    assert c.count() == 1


def test_client_forget_archives(tmp_path):
    c = _client(tmp_path)
    c.save("The staging deploy runs nightly.", tags="deploy")
    forgotten = c.forget("staging deploy")
    assert forgotten is not None and "staging" in forgotten.text.lower()
    assert c.recall("staging deploy") == []                 # archived → not recalled
    assert c.count(include_archived=False) == 0


def test_client_forget_miss_returns_none(tmp_path):
    c = _client(tmp_path)
    c.save("Something unrelated entirely.")
    assert c.forget("nonexistent topic xyz") is None


def test_service_consolidate_returns_report(tmp_path):
    # Uses the default store; just assert the call wires through and reports.
    from relife.memory import store as store_mod
    from relife.memory import events as ev

    store_mod._DB_PATH = tmp_path / "relife.db"
    ev._DB_PATH = tmp_path / "relife.db"
    store_mod.init_db()
    report = MemoryService().consolidate()
    assert hasattr(report, "summary") and isinstance(report.summary(), str)


# --- contract: the agent-facing tool names must not drift ------------------
def test_mcp_tool_names_unchanged():
    cfg = server.memory_server()
    assert cfg["name"] == "relife_memory"
    # The four long-term-memory tools (plus skills/workflows) keep their names,
    # so they surface as the trusted-prefix `mcp__relife_memory__*` and stay
    # auto-allowed by the permission policy.
    expected = {
        "memory_save",
        "memory_recall",
        "memory_forget",
        "memory_consolidate",
        "skill_write",
        "skill_find",
        "workflow_save",
        "workflow_find",
    }
    for name in expected:
        tool_obj = getattr(server, name)
        assert tool_obj.name == name
        assert f"mcp__{cfg['name']}__{tool_obj.name}".startswith("mcp__relife_memory__")
