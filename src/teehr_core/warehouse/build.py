"""Building a derived dataset (climatology, percentiles, naive scale) into its Iceberg table.

A build reads the observed source at one snapshot, a batch of series at a time (memory holds
one batch), builds each batch with :mod:`teehr_core.derived`, writes each batch's rows as a
file, and then replaces the derived table with those files in **one commit**: readers see the
old dataset until the new one is whole, and a failed build leaves it as it was. The commit's
snapshot summary records the lineage: the source and its snapshot, the period read, the
parameters and who built it.

A long build outlives what its tables were loaded with: the storage credentials a catalog vends
with a table (often an hour) and, for the commit, the table's metadata. Given ``reload``, a
build loads both tables again every ``reload_every`` seconds and the target once more right
before it commits.

Needs the ``iceberg`` and ``engine`` extras.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pyarrow.compute as pc
from pyiceberg.expressions import And, BooleanExpression, GreaterThanOrEqual, In, LessThan
from pyiceberg.table import Table

from teehr_core import derived
from teehr_core.warehouse.write import PRODUCER, commit_files, write_file

log = logging.getLogger(__name__)

__all__ = ["BuildResult", "build_derived"]

#: Seconds a loaded table (and the storage credentials vended with it) is used before a build
#: with ``reload`` loads it again.
RELOAD_EVERY = 15 * 60

#: Snapshot summary properties a build records (with :data:`PRODUCER`).
SOURCE = "teehr.source"
SOURCE_SNAPSHOT = "teehr.source-snapshot"
PERIOD = "teehr.period"
PARAMETERS = "teehr.parameters"


@dataclass(frozen=True)
class BuildResult:
    """What a build wrote."""

    rows: int
    series: int  # series with rows (a percentiles build skips series too short)
    batches: int
    snapshot_id: int | None
    source_snapshot: int
    first: datetime | None  # the period of the source rows read
    last: datetime | None


def _name(table: Table) -> str:
    return ".".join(table.name()[-2:])  # namespace.table


def _check(source: Table, target: Table, kind: str) -> None:
    role = source.properties.get("teehr.role")
    if role is not None and role != "observed":
        raise ValueError(f"a {kind} is built from observations; {_name(source)} is {role!r}")
    declared = target.properties.get("teehr.type")
    if declared is not None and declared != kind:
        raise ValueError(f"{_name(target)} is a {declared} dataset, not a {kind}")


def build_derived(
    source: Table,
    target: Table,
    kind: str,
    series_keys: Sequence[str],
    *,
    params: dict[str, Any] | None = None,
    batch_series: int = 100,
    start: datetime | None = None,
    end: datetime | None = None,
    producer: str = "",
    reload: Callable[[Table], Table] | None = None,
    reload_every: float = RELOAD_EVERY,
) -> BuildResult:
    """Replace ``target`` with ``kind`` built from ``source``'s series ``series_keys``.

    ``source`` is read at its current snapshot, ``batch_series`` series at a time, over
    ``start``..``end`` (default: the whole record). ``params`` go to the builder (e.g.
    ``{"min_years": 10}``). Files written are deleted if the build fails before its commit.
    ``reload(table)`` loads a table again from its catalog (e.g. ``catalog.load_table(
    table.name())``): done every ``reload_every`` seconds, and for the target before its commit.
    """
    _check(source, target, kind)
    params = dict(params or {})
    current = source.current_snapshot()
    if current is None:
        raise ValueError(f"{_name(source)} has no data to build from")
    snapshot = current.snapshot_id
    keys = sorted(set(series_keys))
    batches = [keys[i : i + batch_series] for i in range(0, len(keys), batch_series)]
    when: list[BooleanExpression] = []
    if start is not None:
        when.append(GreaterThanOrEqual(term="value_time", value=start.isoformat()))
    if end is not None:
        when.append(LessThan(term="value_time", value=end.isoformat()))
    token = uuid.uuid4().hex[:12]
    paths: list[str] = []
    rows = series = 0
    first: datetime | None = None
    last: datetime | None = None
    started = loaded = time.monotonic()
    log.info(
        "%s %s: from %s at snapshot %d, %d series in %d batches",
        kind, _name(target), _name(source), snapshot, len(keys), len(batches),
    )  # fmt: skip
    try:
        for i, batch in enumerate(batches):
            if reload is not None and time.monotonic() - loaded > reload_every:
                source, target, loaded = reload(source), reload(target), time.monotonic()
            where: BooleanExpression = In(term="series_key", literals=batch)
            for w in when:
                where = And(where, w)
            data = source.scan(
                row_filter=where,
                selected_fields=("series_key", "value_time", "value"),
                snapshot_id=snapshot,
            ).to_arrow()
            if data.num_rows:
                bounds = pc.min_max(data["value_time"])
                lo, hi = bounds["min"].as_py(), bounds["max"].as_py()
                first = lo if first is None or lo < first else first
                last = hi if last is None or hi > last else last
            out = derived.build(kind, data, **params) if data.num_rows else None
            if out is not None and out.num_rows:
                paths.append(write_file(target, out, f"build-{token}-{i:05d}.parquet"))
                rows += out.num_rows
                series += len(pc.unique(out["series_key"]))
            log.info("%s: batch %d of %d (%d rows in)", _name(target), i + 1, len(batches),
                     data.num_rows)  # fmt: skip
        properties = {
            PRODUCER: producer,
            SOURCE: _name(source),
            SOURCE_SNAPSHOT: str(snapshot),
            PERIOD: f"{first.isoformat() if first else ''}/{last.isoformat() if last else ''}",
            PARAMETERS: json.dumps(params, sort_keys=True, default=str),
        }
        if reload is not None:  # fresh metadata and credentials for the commit
            target = reload(target)
        committed = commit_files(target, paths, properties=properties)
    except BaseException:
        for path in paths:
            try:
                target.io.delete(path)
            except Exception:  # best effort: orphan cleanup removes what's left
                log.warning("couldn't delete %s", path)
        raise
    log.info("%s: %d rows for %d series in %.0fs", _name(target), rows, series,
             time.monotonic() - started)  # fmt: skip
    return BuildResult(rows, series, len(batches), committed, snapshot, first, last)
