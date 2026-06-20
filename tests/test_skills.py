"""Unit tests for the procedural skills store."""

from relife.memory import skills as sk


def _fresh(tmp_path):
    sk._SKILLS_DIR = tmp_path / "skills"
    return sk


def test_write_and_find(tmp_path):
    s = _fresh(tmp_path)
    s.write_skill(
        "scaffold-python-cli",
        "Setting up a new Python command-line project.",
        "1. create pyproject\n2. add Typer entry point\n3. pip install -e .",
    )
    hits = s.find_skills("set up a new python cli project")
    assert hits and hits[0].slug == "scaffold-python-cli"
    assert "Typer" in hits[0].body


def test_rewrite_updates_not_duplicates(tmp_path):
    s = _fresh(tmp_path)
    s.write_skill("push-repo", "Pushing a new repo.", "old steps")
    s.write_skill("push-repo", "Pushing a new repo to GitHub.", "new steps")
    assert s.count() == 1
    found = s.read_skill("push-repo")
    assert found and "new steps" in found.body


def test_unrelated_query_no_match(tmp_path):
    s = _fresh(tmp_path)
    s.write_skill("push-repo", "Pushing a repo.", "steps")
    assert s.find_skills("bake a chocolate cake") == []
