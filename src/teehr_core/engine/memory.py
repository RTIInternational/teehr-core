"""Memory limits for DuckDB inside containers.

Ported from VeriRace (``verirace/memory.py`` at 668ab7d).

DuckDB sizes itself from physical memory by default; in a container (or cgroup) with a
smaller limit that gets the process killed (scale test, finding 2). Every DuckDB
connection the engine opens uses :func:`default_memory_limit` instead.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "RUN_SHARE",
    "container_bytes",
    "default_memory_limit",
    "peak_mb",
    "reset_peak",
]

#: Share of the available memory one DuckDB connection may use: a process can have a
#: few open at once (lake reads, run staging, engine), plus Arrow/NumPy and Python.
_MEMORY_SHARE = 0.5


def _cgroup_limit() -> int | None:
    """Return the tightest cgroup v2 memory limit on this process in bytes (None: none)."""
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
    except (OSError, IndexError):
        return None
    limits = []
    path = Path("/sys/fs/cgroup") / rel.lstrip("/")
    for directory in (path, *path.parents):
        try:
            value = (directory / "memory.max").read_text().strip()
        except OSError:
            continue
        if value != "max":
            limits.append(int(value))
        if directory == Path("/sys/fs/cgroup"):
            break
    return min(limits) if limits else None


#: Share for each of the connections a run holds open at once (lake reads, staging,
#: engine): each DuckDB connection caches up to its own limit.
RUN_SHARE = 0.2


def container_bytes() -> int:
    """Return the memory this process's container has: its cgroup limit or physical memory."""
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return min(total, _cgroup_limit() or total)


def default_memory_limit(share: float | None = None) -> str:
    """Return a DuckDB memory limit that fits this process's container.

    ``TEEHR_CORE_DUCKDB_MEMORY_LIMIT`` if set, else ``share`` (default one half) of the
    cgroup limit or of physical memory, whichever is smaller. DuckDB sizes itself from
    physical memory by default, which in a container with a smaller limit gets the
    process killed (scale test, finding 2). Each connection caches up to its own limit,
    so a process holding several open passes a smaller ``share`` for each.
    """
    explicit = os.environ.get("TEEHR_CORE_DUCKDB_MEMORY_LIMIT")
    if explicit:
        return explicit
    return f"{int(container_bytes() * (share or _MEMORY_SHARE)) // 2**20}MB"


def reset_peak() -> bool:
    """Start measuring this process's peak memory afresh (Linux); False if it can't.

    A run's worker processes run many shards each, so each shard's peak needs its own
    start.
    """
    try:
        Path("/proc/self/clear_refs").write_text("5")
    except OSError:
        return False
    return True


def peak_mb() -> int | None:
    """Return this process's peak resident memory in MiB since the last reset (Linux)."""
    try:
        status = Path("/proc/self/status").read_text()
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) // 1024
    return None
