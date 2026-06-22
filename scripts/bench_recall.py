"""Recall scaling benchmark (non-CI).

Seeds a large store and times recall to confirm the two-stage design keeps
recall bounded — Stage 1 pulls at most ``CANDIDATE_TOPN`` candidates from the
FTS5 index, Stage 2 fuse-ranks only those, so latency stays roughly flat as the
store grows instead of scaling with total memory count.

    python scripts/bench_recall.py            # default: 10,000 memories
    python scripts/bench_recall.py 50000

Embeddings are left off (keyword/FTS path) so seeding is fast and the run
measures the keyword two-stage path; the vector index has its own backend and
is exercised separately.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

from relife import config
from relife.memory import store as store_mod

TOPICS = [
    "deploy pipeline staging branch release rollback",
    "python ruff pytest linting formatting tooling",
    "database sqlite migration schema index query",
    "auth token oauth session login security",
    "memory recall embedding cosine activation decay",
]


def seed(s: store_mod.MemoryStore, n: int) -> None:
    """Bulk-insert n varied memories directly (fast; bypasses per-save checks)."""
    s.init_db()
    now = time.time()
    with s._connect() as conn:
        rows = [
            (
                "fact",
                f"{TOPICS[i % len(TOPICS)]} note number {i} extra detail here",
                "bench",
                0.5,
                now,
                now,
            )
            for i in range(n)
        ]
        conn.executemany(
            "INSERT INTO memories (kind, text, tags, importance, created_at, "
            "last_used_at, use_count, status) VALUES (?,?,?,?,?,?,0,'active')",
            rows,
        )
        # Rebuild the FTS index over the bulk-inserted rows.
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    tmp = Path(tempfile.mkdtemp())
    s = store_mod.MemoryStore(tmp / "bench.db")

    t0 = time.time()
    seed(s, n)
    seed_s = time.time() - t0
    print(f"seeded {s.count():,} memories in {seed_s:.2f}s")
    print(f"CANDIDATE_TOPN={config.CANDIDATE_TOPN}  RECALL_FLOOR={config.RECALL_FLOOR}")

    queries = [
        "how does the deploy pipeline release work",
        "what python tooling for linting and tests",
        "sqlite schema migration approach",
    ]
    # Warm up, then time.
    s.recall(queries[0], k=5)
    times = []
    for q in queries * 4:
        t = time.time()
        hits = s.recall(q, k=5)
        times.append((time.time() - t) * 1000)
        assert len(hits) <= 5
    avg = sum(times) / len(times)
    worst = max(times)
    print(f"recall over {n:,} rows: avg {avg:.1f} ms, worst {worst:.1f} ms (k=5)")

    # Bound check: recall must stay cheap (well under a second) at this scale.
    assert worst < 750, f"recall too slow at {n} rows: {worst:.1f} ms"
    print("OK: recall stays bounded at scale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
