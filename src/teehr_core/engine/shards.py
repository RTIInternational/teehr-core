"""Splitting a run into shards that fit in memory.

Ported from VeriRace (``verirace/runs/executor.py`` at 668ab7d: ``plan_shards``,
``shard_rows`` and ``_in_list``), without its lakes: the caller supplies the keys and row
counts from its own catalog.

A shard is a contiguous range of observed series keys. Each brings every predicted
series linked to it, so per-series groups stay inside one shard, and since stored tables
are sorted by ``series_key``, a range of keys sits in few row groups: each shard reads
little beyond its own series (hash buckets spread every shard over the whole table: 13x
read amplification on the USGS-NWM crosswalk against 2.3x for ranges, VeriRace's
``docs/design/scaling-runs.md``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from teehr_core.engine.memory import RUN_SHARE, container_bytes
from teehr_core.sql import lit

__all__ = ["ROW_BYTES", "in_list", "key_ranges", "shard_rows"]

#: Engine memory a predicted row takes while a shard pairs it. Pairing doesn't reliably
#: spill, so a shard must fit its connection's limit: in VeriRace (2026-11-01) 2.07M
#: pairs paired every time at 600 MB (290 bytes a pair) and failed at random below that.
#: With a margin.
ROW_BYTES = 500


def shard_rows(memory_bytes: int | None = None, share: float = RUN_SHARE) -> int:
    """Return the predicted rows a shard may hold.

    ``share`` of ``memory_bytes`` (default: this process's container) at ``ROW_BYTES`` a
    row: the engine connection's limit when it's ``default_memory_limit(share)``.
    """
    total = container_bytes() if memory_bytes is None else memory_bytes
    return max(1, int(total * share) // ROW_BYTES)


def key_ranges(
    keys: Iterable[str],
    rows_per_key: float,
    rows_per_shard: int,
    min_shards: int = 1,
) -> list[tuple[str, ...]]:
    """Split ``keys`` (observed series keys) into contiguous ranges, in key order.

    Enough ranges that each holds at most ``rows_per_shard`` predicted rows at
    ``rows_per_key`` each (the predicted inputs' rows spread evenly over the observed
    series), and at least ``min_shards``; never more ranges than keys. Ranges hold equal
    numbers of keys. Empty when there are no keys.
    """
    ordered = sorted(set(keys))
    if not ordered:
        return []
    needed = math.ceil(rows_per_key * len(ordered) / max(1, rows_per_shard))
    n = min(max(1, min_shards, needed), len(ordered))
    size = len(ordered)
    return [tuple(ordered[i * size // n : (i + 1) * size // n]) for i in range(n)]


def in_list(keys: Iterable[str]) -> str:
    """``IN (...)`` with the keys as literals (matches nothing when empty).

    A literal list is pushed into a table scan, where sorted files prune by it;
    ``IN (SELECT k FROM <registered table>)`` is a join after the scan, which made shard
    reads 2-3x slower (VeriRace's scaling profile).
    """
    return f"IN ({', '.join(lit(k) for k in keys) or 'NULL'})"
