"""Which months of a table changed between two of its snapshots, from Iceberg's own metadata.

Every commit records the data files it added and removed, with each file's column bounds, so the
months a commit touched are the months its files' ``value_time`` bounds span, whoever wrote it
(the API, a job, a flow, a notebook). Rewrites (``replace``: compaction, sorting) change no rows
and are skipped. Incremental runs use this to know what to recompute (ADR-0019 in
teehr-cloud-core).

The months can be more than what changed (a file spanning a month boundary counts both; an upsert
rewrites the stored rows it keeps), never fewer. When they can't be told (an earlier snapshot no
longer in the table's history, delete files, files without bounds), :attr:`Changes.unknown` says
why and the caller treats the table as changed everywhere.
"""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from pyiceberg.conversions import from_bytes
from pyiceberg.manifest import ManifestContent, ManifestEntryStatus
from pyiceberg.table import Table
from pyiceberg.table.snapshots import Operation, Snapshot
from pyiceberg.types import DateType, NestedField, PrimitiveType, TimestampType, TimestamptzType


@dataclass(frozen=True)
class Changes:
    """What changed in a table from one snapshot to a later one.

    - ``commits``: the commits between them that may change rows (rewrites aren't counted);
    - ``added_only``: whether those commits only added rows;
    - ``months``: the months (``YYYY-MM``, UTC) of the column the changed files span, when the
      table has that column;
    - ``unknown``: why what changed can't be told (then treat it as changed everywhere).
    """

    commits: int = 0
    added_only: bool = True
    months: tuple[str, ...] = ()
    unknown: str | None = None
    snapshots: tuple[int, ...] = field(default=(), repr=False)

    @property
    def changed(self) -> bool:
        """Whether anything may have changed."""
        return self.commits > 0 or self.unknown is not None


def _unknown(why: str) -> Changes:
    return Changes(added_only=False, unknown=why)


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


def month_range(low: datetime, high: datetime) -> list[str]:
    """Return the months (``YYYY-MM``) from ``low``'s to ``high``'s, inclusive."""
    year, month = low.year, low.month
    out = []
    while (year, month) <= (high.year, high.month):
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def _between(table: Table, since: int, until: int) -> list[Snapshot] | None:
    """Return the snapshots after ``since`` up to ``until``, oldest first.

    Follows ``until``'s ancestry; None when ``since`` isn't one of its ancestors (expired, or on
    another branch).
    """
    out = []
    snapshot = table.snapshot_by_id(until)
    while snapshot is not None and snapshot.snapshot_id != since:
        out.append(snapshot)
        parent = snapshot.parent_snapshot_id
        snapshot = table.snapshot_by_id(parent) if parent is not None else None
    if snapshot is None:
        return None
    return out[::-1]


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


def changes(
    table: Table, since: int | None, until: int | None = None, column: str | None = "value_time"
) -> Changes:
    """Find what changed in ``table`` after snapshot ``since``.

    Up to ``until`` (default: the current snapshot), with the months of ``column`` the changed
    files span.
    """
    if until is None:
        current = table.current_snapshot()
        until = current.snapshot_id if current else None
    if since == until:
        return Changes()
    if since is None:
        return _unknown("no earlier snapshot to compare with")
    if until is None or table.snapshot_by_id(until) is None:
        return _unknown(f"snapshot {until} isn't in the table")
    snapshots = _between(table, since, until)
    if snapshots is None:
        return _unknown(f"snapshot {since} is no longer in the table's history")
    time_field = _time_field(table, column)
    commits, added_only, months, ids = 0, True, set(), []
    for snapshot in snapshots:
        operation = snapshot.summary.operation if snapshot.summary else None
        if operation == Operation.REPLACE:
            continue
        commits += 1
        ids.append(snapshot.snapshot_id)
        for manifest in snapshot.manifests(table.io):
            if manifest.added_snapshot_id != snapshot.snapshot_id:
                continue  # an earlier commit's files
            if manifest.content == ManifestContent.DELETES:
                return _unknown(f"snapshot {snapshot.snapshot_id} has delete files")
            for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=False):
                if entry.snapshot_id != snapshot.snapshot_id:
                    continue
                if entry.status == ManifestEntryStatus.DELETED:
                    added_only = False
                elif entry.status != ManifestEntryStatus.ADDED:
                    continue
                if time_field is None:
                    continue
                low = (entry.data_file.lower_bounds or {}).get(time_field.field_id)
                high = (entry.data_file.upper_bounds or {}).get(time_field.field_id)
                if low is None or high is None:
                    return _unknown(
                        f"a file of snapshot {snapshot.snapshot_id} has no {column} bounds"
                    )
                kind = time_field.field_type
                assert isinstance(kind, PrimitiveType)
                months.update(
                    month_range(
                        _as_datetime(from_bytes(kind, low), kind),
                        _as_datetime(from_bytes(kind, high), kind),
                    )
                )
    return Changes(
        commits=commits, added_only=added_only, months=tuple(sorted(months)), snapshots=tuple(ids)
    )
