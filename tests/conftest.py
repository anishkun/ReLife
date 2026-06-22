"""Shared test fixtures.

ReLife's contract is that the test suite is **deterministic** and runs with
semantic embeddings **OFF** (see CLAUDE.md), so recall is pure keyword +
activation regardless of whether ``fastembed`` happens to be installed on the
machine. Historically this held only because the package was usually absent; we
now enforce it explicitly so a local install can't make tests slow or
non-deterministic.

Tests that specifically exercise the semantic path opt back in with
``@pytest.mark.semantic`` (and should themselves skip when embeddings are
genuinely unavailable).
"""

from __future__ import annotations

import pytest

from relife.memory import embeddings


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "semantic: test requires real local embeddings (not forced off)"
    )


@pytest.fixture(autouse=True)
def _embeddings_off(request, monkeypatch):
    """Force embeddings off for every test except those marked ``semantic``."""
    if request.node.get_closest_marker("semantic"):
        return
    monkeypatch.setattr(embeddings, "available", lambda: False)
    monkeypatch.setattr(embeddings, "embed", lambda texts: None)
    monkeypatch.setattr(embeddings, "embed_one", lambda text: None)
