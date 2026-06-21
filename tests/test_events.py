"""Unit tests for the tool-event log (raw material for pattern detection)."""

from relife.memory import events as ev


def _fresh(tmp_path):
    ev._DB_PATH = tmp_path / "relife.db"
    ev.init_db()
    return ev


def test_log_and_count(tmp_path):
    e = _fresh(tmp_path)
    e.log_event("Read", "store.py", task_id="t1")
    e.log_event("Edit", "store.py", task_id="t1")
    assert e.count() == 2


def test_empty_tool_ignored(tmp_path):
    e = _fresh(tmp_path)
    assert e.log_event("") == 0
    assert e.count() == 0


def test_grouped_by_task_chronological(tmp_path):
    e = _fresh(tmp_path)
    e.log_event("Read", task_id="t1")
    e.log_event("Edit", task_id="t2")
    e.log_event("Bash", task_id="t1")
    grouped = e.events_by_task()
    assert [x.tool for x in grouped["t1"]] == ["Read", "Bash"]
    assert [x.tool for x in grouped["t2"]] == ["Edit"]
