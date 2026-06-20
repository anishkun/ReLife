"""Orchestrated, resumable large-project builds.

``relife build "<spec>"`` decomposes a big project into milestones, persists a
ledger (``ledger.py``), delegates each milestone to a fresh-context ``builder``
subagent (``agents.py`` + ``orchestrator.py``), and can resume after the run
dies (e.g. a Max session limit). See ``orchestrator.run_build``.
"""
