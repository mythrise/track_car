#!/usr/bin/env python3
"""Scoped process cleanup helpers for car runtime startup."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from typing import Iterable


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return False


def _sudo_kill_pid(pid: int, label: str, grace_s: float) -> bool:
    term = _run(["sudo", "-n", "kill", "-TERM", str(pid)])
    if term.returncode != 0:
        detail = (term.stderr or term.stdout).strip()
        print(
            f"[cleanup] sudo kill failed for pid={pid} ({label}): {detail}. "
            f"Run manually: sudo kill -9 {pid}",
            flush=True,
        )
        return False

    time.sleep(grace_s)
    if _pid_alive(pid):
        kill = _run(["sudo", "-n", "kill", "-KILL", str(pid)])
        if kill.returncode != 0:
            detail = (kill.stderr or kill.stdout).strip()
            print(
                f"[cleanup] sudo kill -9 failed for pid={pid} ({label}): {detail}. "
                f"Run manually: sudo kill -9 {pid}",
                flush=True,
            )
            return False

    print(f"[cleanup] killed pid={pid} ({label}) with sudo", flush=True)
    return True


def _kill_pid(pid: int, label: str, dry_run: bool = False, grace_s: float = 0.25) -> bool:
    if pid == os.getpid():
        return False
    if dry_run:
        print(f"[cleanup] would kill pid={pid} ({label})")
        return True

    if sys.platform == "win32":
        result = _run(["taskkill", "/PID", str(pid), "/F"])
        if result.returncode == 0:
            print(f"[cleanup] killed pid={pid} ({label})")
            return True
        print(f"[cleanup] failed to kill pid={pid} ({label}): {result.stderr.strip()}")
        return False

    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(grace_s)
        if _pid_alive(pid):
            os.kill(pid, signal.SIGKILL)
        print(f"[cleanup] killed pid={pid} ({label})")
        return True
    except PermissionError:
        print(f"[cleanup] permission denied killing pid={pid} ({label}); trying sudo", flush=True)
        return _sudo_kill_pid(pid, label, grace_s)
    except ProcessLookupError:
        return False
    except OSError as exc:
        print(f"[cleanup] failed to kill pid={pid} ({label}): {exc}")
    return False


def kill_pids(pids: Iterable[int], label: str, dry_run: bool = False) -> int:
    killed = 0
    for pid in sorted({int(pid) for pid in pids if int(pid) > 0}):
        if _kill_pid(pid, label=label, dry_run=dry_run):
            killed += 1
    return killed


def pids_by_name(pattern: str) -> list[int]:
    """Find POSIX processes matching a command-line pattern."""
    if sys.platform == "win32":
        return []

    result = _run(["ps", "-eo", "pid=,command="])
    pids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        if not pid_text.isdigit():
            continue
        if "process_cleanup.py" in command or " pgrep " in command or " grep " in command:
            continue
        if re.search(pattern, command):
            pids.append(int(pid_text))
    return pids


def cleanup_named_processes(names: Iterable[str], dry_run: bool = False) -> int:
    total = 0
    for name in names:
        pids = [pid for pid in pids_by_name(name) if pid != os.getpid()]
        if pids:
            total += kill_pids(pids, label=name, dry_run=dry_run)
    return total


def pids_on_port(port: int) -> list[int]:
    """Find processes listening on a TCP port on Windows/macOS/Linux."""
    if sys.platform == "win32":
        result = _run(["netstat", "-ano", "-p", "tcp"])
        pids: list[int] = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0].upper().startswith("TCP"):
                local_addr = parts[1]
                state = parts[3].upper()
                pid_text = parts[-1]
                if state == "LISTENING" and local_addr.endswith(f":{port}") and pid_text.isdigit():
                    pids.append(int(pid_text))
        return pids

    if _has_command("lsof"):
        result = _run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"])
        if result.returncode in (0, 1):
            return [int(x) for x in result.stdout.split() if x.isdigit()]

    if _has_command("ss"):
        result = _run(["ss", "-ltnp", f"sport = :{port}"])
        pids = []
        for match in re.finditer(r"pid=(\d+)", result.stdout):
            pids.append(int(match.group(1)))
        return pids

    print("[cleanup] cannot inspect ports: install lsof or iproute2/ss")
    return []


def _has_command(name: str) -> bool:
    return _run(["/bin/sh", "-lc", f"command -v {name}"]).returncode == 0


def cleanup_port(port: int, dry_run: bool = False) -> int:
    pids = [pid for pid in pids_on_port(port) if pid != os.getpid()]
    if not pids:
        print(f"[cleanup] no process is listening on tcp:{port}")
        return 0
    return kill_pids(pids, label=f"tcp:{port}", dry_run=dry_run)
