"""Stop only the ToolVietSub server process tree on Windows or POSIX."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping


def _descendants(root_pid: int, parent_table: Mapping[int, int]) -> set[int]:
    """Return every descendant of ``root_pid`` from a PID -> PPID snapshot."""
    children: dict[int, set[int]] = {}
    for pid, ppid in parent_table.items():
        children.setdefault(int(ppid), set()).add(int(pid))

    found: set[int] = set()
    pending = list(children.get(int(root_pid), ()))
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        pending.extend(children.get(pid, ()))
    return found


def _process_table() -> dict[int, int]:
    """Read a small, platform-native process parent table."""
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid="],
        capture_output=True,
        text=True,
        check=True,
    )
    table: dict[int, int] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            table[int(fields[0])] = int(fields[1])
        except ValueError:
            continue
    return table


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _signal(pid: int, sig: signal.Signals) -> None:
    try:
        os.kill(pid, sig)
    except (OSError, ProcessLookupError):
        pass


def _depth(pid: int, root_pid: int, parent_table: Mapping[int, int]) -> int:
    depth = 0
    current = pid
    seen: set[int] = set()
    while current != root_pid and current not in seen:
        seen.add(current)
        parent = parent_table.get(current)
        if parent is None:
            break
        current = parent
        depth += 1
    return depth


def _terminate_windows_tree(root_pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(int(root_pid)), "/T", "/F"],
        capture_output=True,
        text=True,
        check=False,
    )


def _terminate_posix_tree(root_pid: int, *, grace_seconds: float = 0.6) -> None:
    try:
        table = _process_table()
    except (OSError, subprocess.SubprocessError):
        table = {}

    descendants = _descendants(root_pid, table)
    ordered = sorted(
        descendants,
        key=lambda pid: _depth(pid, root_pid, table),
        reverse=True,
    )
    for pid in ordered:
        _signal(pid, signal.SIGTERM)

    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline and any(_is_alive(pid) for pid in ordered):
        time.sleep(0.05)

    for pid in ordered:
        if _is_alive(pid):
            _signal(pid, signal.SIGKILL)

    if _is_alive(root_pid):
        _signal(root_pid, signal.SIGTERM)
        time.sleep(0.1)
        if _is_alive(root_pid):
            _signal(root_pid, signal.SIGKILL)


def terminate_process_tree(root_pid: int, *, grace_seconds: float = 0.6) -> None:
    """Terminate the ToolVietSub server and its descendants for this platform."""
    if os.name == "nt":
        _terminate_windows_tree(root_pid)
        return
    _terminate_posix_tree(root_pid, grace_seconds=grace_seconds)


def schedule_shutdown(root_pid: int | None = None, *, delay_seconds: float = 0.2) -> None:
    """Schedule shutdown after the HTTP response can leave the server."""
    target = os.getpid() if root_pid is None else int(root_pid)

    def worker() -> None:
        time.sleep(max(0.0, delay_seconds))
        terminate_process_tree(target)

    threading.Thread(target=worker, name="tool-shutdown", daemon=True).start()
