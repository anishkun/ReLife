"""A cross-process exclusive lock on a sidecar file (``<file>.lock``).

The agent registry is one JSON file written by the CLI (`relife agent …`), the
server's routes and a crew's worker — possibly in different processes at once.
An in-process lock can't stop two processes interleaving read → modify →
write, so writers also take this OS-level lock: ``msvcrt.locking`` on Windows,
``fcntl.flock`` elsewhere. Held only for the few milliseconds of a write.

The lock is advisory (only cooperating writers respect it) and released when
the holder's file handle closes — including when the process dies, so a crash
can never leave it stuck.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

TIMEOUT = 10.0
POLL = 0.02


class LockTimeout(TimeoutError):
    """Another process held the lock for longer than ``TIMEOUT``."""


@contextmanager
def file_lock(target: Path, *, timeout: float | None = None) -> Iterator[None]:
    lock_path = Path(str(target) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + (TIMEOUT if timeout is None else timeout)
    try:
        while True:
            try:
                _acquire(fd)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise LockTimeout(f"{lock_path} is held by another process") from None
                time.sleep(POLL)
        try:
            yield
        finally:
            _release(fd)
    finally:
        os.close(fd)


if os.name == "nt":
    import msvcrt

    def _acquire(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _release(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _acquire(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _release(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)
