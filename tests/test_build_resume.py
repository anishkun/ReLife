"""Resume-path tests for orchestrated builds.

These lock in two fixes prompted by a live failure: a persisted session id can
expire across a Max session-limit reset, so resume must (a) run in the ledger's
own workspace, and (b) fall back to a fresh session instead of hard-crashing.
The SDK client is faked, so no live model is involved.
"""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest
from claude_agent_sdk import ProcessError

from relife import config
from relife.build import orchestrator
from relife.build.ledger import BuildLedger

STALE_SESSION = "dead-session-id"


@pytest.fixture
def builds_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDS_DIR", tmp_path / "builds")
    return tmp_path


class _FakeClient:
    """Records construction; connect() fails when asked to resume the stale
    session, mimicking the `claude` subprocess exiting on a missing conversation."""

    instances: list["_FakeClient"] = []

    def __init__(self, options):
        self.options = options
        self.disconnected = False
        type(self).instances.append(self)

    async def connect(self):
        if getattr(self.options, "resume", None) == STALE_SESSION:
            raise ProcessError("No conversation found with session ID", exit_code=1)

    async def query(self, prompt):
        self.prompt = prompt

    async def receive_response(self):
        return
        yield  # noqa: makes this an (empty) async generator

    async def disconnect(self):
        self.disconnected = True


@pytest.fixture
def fake_client(monkeypatch):
    _FakeClient.instances = []
    monkeypatch.setattr(orchestrator, "ClaudeSDKClient", _FakeClient)
    return _FakeClient


def _resume(ledger):
    anyio.run(
        lambda: orchestrator.run_build(
            None, cwd=Path("/some/other/dir"), can_use_tool=None, resume_id=ledger.build_id
        )
    )


def test_resume_falls_back_to_fresh_session_when_stale(builds_dir, fake_client):
    led = BuildLedger.create("harden the payment engine", str(builds_dir / "ws"))
    led.set_plan(["scaffold", "tests"])
    led.set_session_id(STALE_SESSION)

    _resume(led)

    # Connected twice: stale session first (fails), then a fresh one (resume=None).
    assert len(fake_client.instances) == 2
    assert fake_client.instances[0].options.resume == STALE_SESSION
    assert fake_client.instances[1].options.resume is None
    # The stale handle was dropped so future resumes don't keep retrying it.
    assert BuildLedger.load(led.build_id).session_id is None
    assert fake_client.instances[1].disconnected


def test_resume_runs_in_the_ledgers_workspace(builds_dir, fake_client):
    ws = str(builds_dir / "apexpay-prod")
    led = BuildLedger.create("spec", ws)
    led.set_plan(["m1"])  # no session id → fresh connect, no fallback

    _resume(led)

    assert len(fake_client.instances) == 1
    # cwd came from the ledger, NOT the bogus cwd passed on the command line.
    assert fake_client.instances[0].options.cwd == str(Path(ws))


def test_fresh_build_connect_error_propagates(builds_dir, monkeypatch):
    """A connect failure on a brand-new build (no session to fall back from)
    must surface, not silently retry."""
    class _AlwaysFails(_FakeClient):
        async def connect(self):
            raise ProcessError("boom", exit_code=1)

    _AlwaysFails.instances = []
    monkeypatch.setattr(orchestrator, "ClaudeSDKClient", _AlwaysFails)

    with pytest.raises(ProcessError):
        anyio.run(
            lambda: orchestrator.run_build(
                "build something new", cwd=Path(str(builds_dir / "ws")), can_use_tool=None
            )
        )
