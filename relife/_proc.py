"""Is a process still running? (Cross-platform, read-only.)

Used to tell a record left "running" by a process that died (a crash, a closed
terminal, a killed server) from one whose owner is still at work. Never use
``os.kill(pid, 0)`` for this on Windows: there any signal other than the two
console events is ``TerminateProcess`` — the "probe" would kill the process.
"""

from __future__ import annotations

import os


def pid_alive(pid: object) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        return _alive_windows(pid)
    try:
        os.kill(pid, 0)  # POSIX: signal 0 only checks
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False
    return True


def _alive_windows(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # Access denied still means it exists; anything else (invalid
        # parameter) means there is no such process.
        return ctypes.get_last_error() == 5
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)
