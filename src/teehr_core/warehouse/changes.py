"""What changed in a table since an earlier state of it, from Iceberg's own metadata.

Incremental runs need the months of ``value_time`` whose rows changed since the last run (ADR-0019
in teehr-cloud-core), whoever wrote them (the API, a job, a flow, a notebook). Rather than replay
the commits in between, which expiring snapshots throws away, a run records the table's
:class:`TableState` and the next run compares the table's live files with it:

- each live file records the sequence number of the commit that wrote it: a file from a later
  commit than the recorded state holds new or rewritten rows, in the months its ``value_time``
  bounds span;
- a partition whose older files now hold fewer rows than were recorded lost rows (deleted, or
  rewritten into new files): its recorded months changed.

Both survive snapshot expiry. A rewrite (compaction, sorting) after the recorded state makes its
files look new, so the months it rewrote count as changed: more than changed, never less. When
changes can't be told (delete files, files without bounds), :attr:`Changes.unknown` says why and
the caller treats the table as changed everywhere.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from pyiceberg.conversions import from_bytes
from pyiceberg.manifest import DataFile, ManifestContent
from pyiceberg.table import Table
from pyiceberg.table.snapshots import Snapshot
from pyiceberg.types import DateType, NestedField, PrimitiveType, TimestampType, TimestamptzType


@dataclass(frozen=True)
class TableState:
    """A table's state at one snapshot, compact enough to keep with a run.

    - ``sequence``: the snapshot's sequence number (0: no snapshot yet);
    - ``partitions``: per partition, its rows and the months (``YYYY-MM``) its files span.
    """

    sequence: int
    partitions: dict[str, tuple[int, tuple[str, ...]]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Return the state as JSON-ready data (for a run's result)."""
        return {
            "sequence": self.sequence,
            "partitions": {
                k: [rows, list(months)] for k, (rows, months) in self.partitions.items()
            },
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TableState":
        """Read a state :meth:`to_json` wrote."""
        return cls(
            sequence=int(data["sequence"]),
            partitions={
                k: (int(rows), tuple(months)) for k, (rows, months) in data["partitions"].items()
            },
        )


@dataclass(frozen=True)
class Changes:
    """What changed in a table since a recorded state.

    - ``new_files``: live files written by later commits;
    - ``added_only``: whether every row of the recorded state is still there unchanged;
    - ``months``: the months (``YYYY-MM``, UTC) of the column that changed, when the table has it;
    - ``unknown``: why what changed can't be told (then treat it as changed everywhere).
    """

    new_files: int = 0
    added_only: bool = True
    months: tuple[str, ...] = ()
    unknown: str | None = None

    @property
    def changed(self) -> bool:
        """Whether anything may have changed."""
        return self.new_files > 0 or not self.added_only or self.unknown is not None


def _unknown(why: str) -> Changes:
    return Changes(added_only=False, unknown=why)


def month_range(low: datetime, high: datetime) -> list[str]:
    """Return the months (``YYYY-MM``) from ``low``'s to ``high``'s, inclusive."""
    year, month = low.year, low.month
    out = []
    while (year, month) <= (high.year, high.month):
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def _as_datetime(value: object, kind: object) -> datetime:
    if isinstance(kind, DateType):
        assert isinstance(value, int)
        return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(days=value)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    assert isinstance(value, int)  # microseconds since the epoch
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=value)


def _time_field(table: Table, column: str | None) -> NestedField | None:
    if column is None:
        return None
    try:
        found = table.schema().find_field(column)
    except ValueError:
        return None
    if not isinstance(found.field_type, TimestampType | TimestamptzType | DateType):
        raise ValueError(f"column '{column}' isn't a time or date")
    return found


class _Unknown(Exception):
    pass


def _snapshot(table: Table, snapshot_id: int | None) -> Snapshot | None:
    if snapshot_id is None:
        return table.current_snapshot()
    found = table.snapshot_by_id(snapshot_id)
    if found is None:
        raise _Unknown(f"snapshot {snapshot_id} isn't in the table")
    return found


def _months(data: DataFile, time_field: NestedField | None) -> tuple[str, ...]:
    if time_field is None:
        return ()
    low = (data.lower_bounds or {}).get(time_field.field_id)
    high = (data.upper_bounds or {}).get(time_field.field_id)
    if low is None or high is None:
        raise _Unknown(f"a file has no {time_field.name} bounds")
    kind = time_field.field_type
    assert isinstance(kind, PrimitiveType)
    return tuple(
        month_range(
            _as_datetime(from_bytes(kind, low), kind), _as_datetime(from_bytes(kind, high), kind)
        )
    )


def _live_files(
    table: Table, snapshot: Snapshot, time_field: NestedField | None
) -> Iterator[tuple[str, int, int, tuple[str, ...]]]:
    """Each live data file: (partition, sequence number, rows, months)."""
    for manifest in snapshot.manifests(table.io):
        if manifest.content == ManifestContent.DELETES:
            if manifest.existing_files_count or manifest.added_files_count:
                raise _Unknown("the table has delete files")
            continue
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True):
            data: DataFile = entry.data_file
            if entry.sequence_number is None:
                raise _Unknown("a file has no sequence number (format version 1)")
            values = [data.partition[i] for i in range(len(data.partition))]
            partition = f"{data.spec_id}:{values}"
            yield partition, entry.sequence_number, data.record_count, _months(data, time_field)


def table_state(
    table: Table, snapshot_id: int | None = None, column: str | None = "value_time"
) -> TableState:
    """Record ``table``'s state at ``snapshot_id`` (default: the current snapshot).

    Raises ``ValueError`` when the state can't be told (delete files, files without bounds).
    """
    try:
        snapshot = _snapshot(table, snapshot_id)
        if snapshot is None:
            return TableState(sequence=0)
        partitions: dict[str, tuple[int, set[str]]] = {}
        for partition, _, rows, months in _live_files(table, snapshot, _time_field(table, column)):
            count, spans = partitions.get(partition, (0, set()))
            partitions[partition] = (count + rows, spans | set(months))
    except _Unknown as exc:
        raise ValueError(str(exc)) from None
    return TableState(
        sequence=snapshot.sequence_number or 0,
        partitions={k: (rows, tuple(sorted(m))) for k, (rows, m) in partitions.items()},
    )


def changes(
    table: Table,
    since: TableState | None,
    snapshot_id: int | None = None,
    column: str | None = "value_time",
) -> Changes:
    """Find what changed in ``table`` since the state ``since``, as of ``snapshot_id``.

    ``snapshot_id`` defaults to the current snapshot. ``months`` holds the months of ``column``
    that changed.
    """
    if since is None:
        return _unknown("no earlier state to compare with")
    months: set[str] = set()
    new_files = 0
    kept: dict[str, int] = {}
    try:
        snapshot = _snapshot(table, snapshot_id)
        if snapshot is not None:
            for partition, sequence, rows, spans in _live_files(
                table, snapshot, _time_field(table, column)
            ):
                if sequence > since.sequence:
                    new_files += 1
                    months.update(spans)
                elif partition not in since.partitions:
                    raise _Unknown(f"partition {partition} wasn't in the recorded state")
                else:
                    kept[partition] = kept.get(partition, 0) + rows
    except _Unknown as exc:
        return _unknown(str(exc))
    added_only = True
    for partition, (rows, spans) in since.partitions.items():
        if kept.get(partition, 0) < rows:  # rows deleted, or rewritten into new files
            added_only = False
            months.update(spans)
    return Changes(new_files=new_files, added_only=added_only, months=tuple(sorted(months)))
