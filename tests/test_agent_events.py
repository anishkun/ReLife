"""Unit tests for the streaming-message → structured-event converter.

``to_event`` is the single source of the streaming taxonomy, consumed by both the
terminal renderer and the web server's SSE stream. These are pure (no async, no
model) — they feed real SDK message objects in and assert the emitted event dicts.
"""

from __future__ import annotations

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from relife.agent import to_event


def _assistant(*blocks) -> AssistantMessage:
    return AssistantMessage(content=list(blocks), model="claude-opus-4-8")


def _user(*blocks) -> UserMessage:
    """How tool results really arrive: a ``user``-type frame from the CLI."""
    return UserMessage(content=list(blocks))


def test_text_block_emits_text():
    evs = to_event(_assistant(TextBlock(text="hello world")))
    assert evs == [{"type": "text", "text": "hello world"}]


def test_blank_text_is_dropped():
    assert to_event(_assistant(TextBlock(text="   \n "))) == []


def test_thinking_block_hides_content():
    evs = to_event(_assistant(ThinkingBlock(thinking="secret reasoning", signature="s")))
    assert evs == [{"type": "thinking"}]


def test_tool_use_carries_name_and_brief():
    block = ToolUseBlock(id="t1", name="Bash", input={"command": "pytest -q"})
    evs = to_event(_assistant(block))
    assert evs == [{"type": "tool_use", "name": "Bash", "brief": "pytest -q"}]


def test_tool_result_string_content():
    block = ToolResultBlock(tool_use_id="t1", content="all tests passed")
    evs = to_event(_user(block))
    assert evs == [{"type": "tool_result", "brief": "all tests passed"}]


def test_tool_result_list_content_and_error_prefix():
    block = ToolResultBlock(
        tool_use_id="t1",
        content=[{"type": "text", "text": "boom\nstack trace"}],
        is_error=True,
    )
    evs = to_event(_user(block))
    assert evs == [{"type": "tool_result", "brief": "error: boom stack trace"}]


def test_tool_result_on_assistant_message_still_handled():
    """The parser can place a tool_result on either carrier; both must map to
    the same event."""
    block = ToolResultBlock(tool_use_id="t1", content="ok")
    assert to_event(_assistant(block)) == [{"type": "tool_result", "brief": "ok"}]


def test_user_text_is_not_echoed_as_an_event():
    """A UserMessage carrying the caller's own turn yields nothing — the server
    publishes that itself on submit, so emitting it here would double it."""
    assert to_event(UserMessage(content="build me a thing")) == []
    assert to_event(_user(TextBlock(text="build me a thing"))) == []


def test_multiple_blocks_in_order():
    msg = _assistant(
        TextBlock(text="on it"),
        ToolUseBlock(id="t1", name="Edit", input={"file_path": "src/app.py"}),
    )
    evs = to_event(msg)
    assert [e["type"] for e in evs] == ["text", "tool_use"]
    assert evs[1]["brief"] == "src/app.py"


def test_result_message_carries_cost():
    msg = ResultMessage(
        subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
        num_turns=1, session_id="s", total_cost_usd=0.0421,
    )
    assert to_event(msg) == [{"type": "result", "cost_usd": 0.0421}]


def test_system_message_yields_nothing():
    assert to_event(SystemMessage(subtype="init", data={})) == []


def test_brief_falls_back_to_leading_fields_for_connector_calls():
    """A connector call has no `command`/`path`; the approval card must still
    show what is about to leave the machine."""
    from relife.agent import _tool_brief

    brief = _tool_brief(
        {"to": "boss@corp.com", "subject": "Q3 numbers", "body": "hi\n\nsee attached", "cc": ""},
        limit=400,
    )
    assert brief == "to=boss@corp.com  subject=Q3 numbers  body=hi see attached"
    assert _tool_brief({"command": "pytest -q"}) == "pytest -q"
    assert _tool_brief({}) == ""
    assert len(_tool_brief({"body": "x" * 500}, limit=80)) <= 80
