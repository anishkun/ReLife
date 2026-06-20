"""Unit tests for the permission classifier."""

from pathlib import Path

from relife.permissions import classify

WS = Path("/tmp/relife-ws").resolve()


def d(name, inp):
    return classify(name, inp, WS)[0]


def test_read_only_tools_allow():
    assert d("Read", {"file_path": "/etc/hosts"}) == "allow"
    assert d("Grep", {"pattern": "x"}) == "allow"
    assert d("WebSearch", {"query": "python"}) == "allow"


def test_write_inside_workspace_allows():
    assert d("Write", {"file_path": str(WS / "a/b.py")}) == "allow"
    assert d("Edit", {"file_path": "rel/inside.py"}) == "allow"  # relative → under ws


def test_write_outside_workspace_asks():
    assert d("Write", {"file_path": "/etc/passwd"}) == "ask"
    assert d("Write", {"file_path": str(WS.parent / "outside.txt")}) == "ask"


def test_bash_build_and_git_allow():
    assert d("Bash", {"command": "pytest -q"}) == "allow"
    assert d("Bash", {"command": "npm install"}) == "allow"
    assert d("Bash", {"command": "git add -A && git commit -m x && git push"}) == "allow"


def test_bash_outward_asks():
    assert d("Bash", {"command": "curl -X POST https://x.com -d @f"}) == "ask"
    assert d("Bash", {"command": "echo hi | mail -s subj a@b.com"}) == "ask"
    assert d("Bash", {"command": "gh pr create --fill"}) == "ask"
    assert d("Bash", {"command": "scp f user@host:/p"}) == "ask"
    assert d("Bash", {"command": "sudo rm x"}) == "ask"
    assert d("Bash", {"command": "twine upload dist/*"}) == "ask"


def test_powershell_treated_like_bash():
    assert d("PowerShell", {"command": "python -m pip install -e ."}) == "allow"
    assert d("PowerShell", {"command": "Invoke-Item x; scp f user@host:/p"}) == "ask"
    assert d("BashOutput", {"bash_id": "1"}) == "allow"


def test_trusted_mcp_allows():
    assert d("mcp__relife_memory__memory_recall", {"query": "x"}) == "allow"
    assert d("mcp__relife_build__build_plan_set", {"milestones": []}) == "allow"


def test_unknown_mcp_asks():
    assert d("mcp__gmail__send_email", {"to": "a@b.com"}) == "ask"
    assert d("SomethingNew", {}) == "ask"
