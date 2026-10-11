"""Atomic JSON-file writes that survive Windows' sharing rules.

Every durable record here (agents, schedules, run records, crew records) is
rewritten as tmp + ``os.replace`` while other threads — server routes, a crew's
worker, the scheduler's recorder — read the same file. On Windows a file that
is open for reading can't be replaced, and one being replaced can't be opened,
so either side may get a ``PermissionError`` for an instant. Retry briefly
instead of failing a request or losing a save. (CI caught it: concurrent agent
creates on windows-latest.)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")

RETRIES = 40
RETRY_DELAY = 0.025


def retrying(fn: Callable[[], T]) -> T:
    for attempt in range(RETRIES):
        try:
            return fn()
        except PermissionError:
            if attempt == RETRIES - 1:
                raise
            time.sleep(RETRY_DELAY)
    raise AssertionError("unreachable")  # pragma: no cover


def replace(tmp: Path, path: Path) -> None:
    retrying(lambda: os.replace(tmp, path))


def read_text(path: Path) -> str:
    return retrying(lambda: Path(path).read_text(encoding="utf-8"))
