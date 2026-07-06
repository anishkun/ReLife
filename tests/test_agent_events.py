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
)

from relife.agent import to_event


def _assistant(*blocks) -> AssistantMessage:
    return AssistantMessage(content=list(blocks), model="claude-opus-4-8")


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
    evs = to_event(_assistant(block))
    assert evs == [{"type": "tool_result", "brief": "all tests passed"}]


def test_tool_result_list_content_and_error_prefix():
    block = ToolResultBlock(
        tool_use_id="t1",
        content=[{"type": "text", "text": "boom\nstack trace"}],
        is_error=True,
    )
    evs = to_event(_assistant(block))
    assert evs == [{"type": "tool_result", "brief": "error: boom stack trace"}]


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
