"""Write a timeseries' rows into its dataset, as a producer.

For anyone writing a TEEHR dataset directly rather than through the control plane's API: a
scheduled ingest flow, or a person writing a one-off ingest from a notebook. It works with any
PyIceberg catalog and writes as whoever the catalog authenticates as, so it can write any project
that identity may write (ADR-0019 and ADR-0020 in teehr-cloud-core).

The dataset must already be registered (through the API), which makes its table, its series table
and their properties. Everything the write needs is read from them: the columns, the key, the
unit, the series table and the location layer. Each batch is checked as the API checks uploads,
its new series are added to the series table, and it is upserted in one commit.

    from teehr_core.warehouse.ingest import write_timeseries

    written = write_timeseries(catalog, "public", "usgs_streamflow", batches,
                               unit="m3/s", producer="notebook:me")
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyarrow import fs
from pyiceberg.expressions import And, BooleanExpression, GreaterThanOrEqual, LessThan, Or
from pyiceberg.table import Table

from teehr_core.model.units import UnitRegistry, conversion_between
from teehr_core.warehouse.conform import DataMismatch, conform, duplicate_keys
from teehr_core.warehouse.layout import COLUMN_TYPES
from teehr_core.warehouse.write import PRODUCER, insert_missing, key_columns, upsert

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog

#: The table property naming a dataset's type, and its timeseries value.
TYPE = "teehr.type"
TIMESERIES = "timeseries"


@dataclass
class Written:
    """What :func:`write_timeseries` wrote, over every batch."""

    rows: int = 0
    series_added: int = 0
    batches: int = 0
    #: The last commit's snapshot (None: nothing committed).
    snapshot_id: int | None = None
    #: The UTC months (YYYY-MM) of ``value_time`` the rows fell in.
    months: list[str] = field(default_factory=list)


def columns_of(table: Table) -> list[dict[str, Any]]:
    """Return a table's declared columns ({name, type, required, doc}), from its schema.

    The inverse of :func:`~teehr_core.warehouse.layout.iceberg_schema`, so rows can be conformed
    to a table without the registry that declared it.
    """
    names = {str(t): name for name, t in COLUMN_TYPES.items()}
    columns = []
    for f in table.schema().fields:
        kind = names.get(str(f.field_type))
        if kind is None:
            raise DataMismatch(f"column '{f.name}' has a type TEEHR doesn't declare: "
                               f"{f.field_type}")  # fmt: skip
        columns.append({"name": f.name, "type": kind, "required": f.required, "doc": f.doc})
    return columns


def check(data: pa.Table, table: Table) -> pa.Table:
    """Return ``data`` conformed to ``table``'s columns.

    Raises :class:`DataMismatch` saying why not: wrong columns or types, a key repeated, an empty
    series key.
    """
    typed = conform(data, columns_of(table))
    problems = []
    repeated = duplicate_keys(typed, key_columns(table))
    if repeated:
        problems.append(f"{repeated} key values occur more than once")
    blank = pc.sum(pc.equal(typed["series_key"], "")).as_py() or 0
    if blank:
        problems.append(f"{blank} rows have an empty series key")
    if problems:
        raise DataMismatch("; ".join(problems))
    return typed


def location_keys(
    catalog: Catalog, namespace: str, name: str, where: str | None = None
) -> list[str]:
    """Return the location keys of timeseries dataset ``name``'s location layer, sorted.

    The locations an ingest fetches for. ``where`` filters on the layer's attributes, as a
    PyIceberg row filter (e.g. ``"is_active == True"``).
    """
    props = catalog.load_table((namespace, name)).properties
    layer = props.get("teehr.location_layer")
    if not layer:
        raise DataMismatch(f"{namespace}.{name} has no location layer (teehr.location_layer)")
    scan = catalog.load_table((namespace, layer)).scan(
        row_filter=where or "True", selected_fields=("location_key",)
    )
    return sorted(scan.to_arrow()["location_key"].to_pylist())


def combine(batches: Iterable[pa.Table], rows: int = 1_000_000) -> Iterator[pa.Table]:
    """Yield ``batches`` joined into tables of at least ``rows`` rows (the last may be fewer).

    Fewer, larger commits than one per fetched batch: a source's batches are often small
    (e.g. a request's worth of sites), and each write is a commit and its files.
    """
    held: list[pa.Table] = []
    count = 0
    for batch in batches:
        if not batch.num_rows:
            continue
        held.append(batch)
        count += batch.num_rows
        if count >= rows:
            yield pa.concat_tables(held, promote_options="default")
            held, count = [], 0
    if held:
        yield pa.concat_tables(held, promote_options="default")


def _time_column(data: pa.Table) -> str:
    """Return the column a dataset is partitioned on: issue time for forecasts, else value time."""
    return "reference_time" if "reference_time" in data.column_names else "value_time"


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    year, mon = (int(p) for p in month.split("-"))
    start = datetime(year, mon, 1, tzinfo=UTC)
    return start, datetime(year + mon // 12, mon % 12 + 1, 1, tzinfo=UTC)


def _month_filter(data: pa.Table) -> BooleanExpression:
    """Every row in the UTC months ``data``'s time column falls in.

    Consecutive months are one range, and the ranges are joined as a balanced tree: a
    chain of an ``Or`` per month overflows PyIceberg's recursive visitors (a backfill's
    commit can span centuries).
    """
    column = _time_column(data)
    stamps = pc.strftime(data[column], format="%Y-%m", locale="C")
    ranges: list[tuple[datetime, datetime]] = []
    for month in sorted(m for m in pc.unique(stamps).to_pylist() if m):
        start, end = _month_bounds(month)
        if ranges and ranges[-1][1] == start:
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((start, end))
    clauses: list[BooleanExpression] = [
        And(GreaterThanOrEqual(term=column, value=start.isoformat()),
            LessThan(term=column, value=end.isoformat()))
        for start, end in ranges
    ]  # fmt: skip
    while len(clauses) > 1:
        pairs = [Or(a, b) for a, b in zip(clauses[::2], clauses[1::2], strict=False)]
        clauses = pairs + clauses[len(pairs) * 2 :]
    return clauses[0]


def _current_snapshot(table: Table) -> int | None:
    table.refresh()
    snapshot = table.current_snapshot()
    return snapshot.snapshot_id if snapshot else None


def stage_by_month(batches: Iterable[pa.Table], directory: str) -> int:
    """Write ``batches`` under ``directory`` as Parquet, one folder per UTC month; return rows.

    The first step of a backfill: a source's batches are by site, or series, across their
    whole record, while :func:`staged_months` gives them back a month at a time for
    ``write_timeseries(..., replace_months=True)``. ``directory`` is a local path or a URI
    PyArrow can open (e.g. ``s3://bucket/prefix``).
    """
    filesystem, root = fs.FileSystem.from_uri(_uri(directory))
    rows = 0
    for batch in batches:
        if not batch.num_rows:
            continue
        stamps = pc.strftime(batch[_time_column(batch)], format="%Y-%m", locale="C")
        for month in pc.unique(stamps).to_pylist():
            part = batch.filter(pc.equal(stamps, month))
            folder = f"{root}/month={month}"
            filesystem.create_dir(folder, recursive=True)
            pq.write_table(part, f"{folder}/{uuid.uuid4().hex}.parquet", filesystem=filesystem)
        rows += batch.num_rows
    return rows


def staged_months(directory: str, months: Iterable[str] | None = None) -> Iterator[pa.Table]:
    """Yield what :func:`stage_by_month` wrote, one month (all its rows) at a time, in order.

    ``months`` (``YYYY-MM``) limits it to those.
    """
    filesystem, root = fs.FileSystem.from_uri(_uri(directory))
    found = filesystem.get_file_info(fs.FileSelector(root, allow_not_found=True))
    staged = sorted(
        i.base_name.split("=", 1)[1] for i in found
        if i.type == fs.FileType.Directory and i.base_name.startswith("month=")
    )  # fmt: skip
    wanted = set(months) if months is not None else None
    for month in staged:
        if wanted is not None and month not in wanted:
            continue
        parts = filesystem.get_file_info(fs.FileSelector(f"{root}/month={month}"))
        tables = [pq.read_table(p.path, filesystem=filesystem) for p in parts
                  if p.base_name.endswith(".parquet")]  # fmt: skip
        if tables:
            yield pa.concat_tables(tables, promote_options="default")


def _uri(directory: str) -> str:
    return directory if "://" in directory else f"file://{os.path.abspath(directory)}"


def _months(data: pa.Table) -> set[str]:
    stamps = pc.strftime(data["value_time"], format="%Y-%m", locale="C")
    return {m for m in pc.unique(stamps).to_pylist() if m}


def write_timeseries(
    catalog: Catalog,
    namespace: str,
    name: str,
    rows: pa.Table | Iterable[pa.Table],
    *,
    producer: str,
    unit: str | None = None,
    units: UnitRegistry | None = None,
    location_of: Callable[[str], str | None] | None = None,
    properties: dict[str, str] | None = None,
    replace_months: bool = False,
) -> Written:
    """Write ``rows`` (one table, or batches of them) into the timeseries dataset ``name``.

    Each batch is checked against the dataset's columns, its new series are added to the series
    table, and it is upserted on the dataset's key in one commit: a batch is written whole or
    not at all, and writing it again changes nothing. Values are in ``unit`` (default: the
    dataset's own); another unit is converted with ``units``. A new series is at the location
    ``location_of(series_key)`` in the dataset's location layer (e.g. ``lambda key: key`` when
    series keys are location keys; default: unset). ``producer`` names who wrote it (a flow
    run, a notebook) in each commit, with ``properties``.

    ``replace_months``: each batch holds whole UTC months (of ``reference_time`` for a forecast,
    else ``value_time``), and replaces those months: rows of any series not in the batch are
    deleted from them. For backfills and reloads of every series at once (:func:`staged_months`
    gives such batches): it drops whole partitions instead of rereading and rewriting stored
    rows, which an upsert does and which grows with what is stored.

    Raises :class:`DataMismatch` for rows that don't fit (before anything of that batch is
    written; earlier batches stay written) and the catalog's own errors (e.g. no permission).
    """
    table = catalog.load_table((namespace, name))
    props = table.properties
    if props.get(TYPE) != TIMESERIES:
        raise DataMismatch(f"{namespace}.{name} isn't a timeseries dataset ({TYPE}: "
                           f"{props.get(TYPE)})")  # fmt: skip
    dataset_unit = props.get("teehr.unit")
    if unit and dataset_unit and unit != dataset_unit:
        conversion = conversion_between(units, unit, dataset_unit)
    else:
        conversion = None
    series_name = props.get("teehr.series_table")
    series = catalog.load_table((namespace, series_name)) if series_name else None
    layer = props.get("teehr.location_layer")
    commit = {PRODUCER: producer, **(properties or {})}

    written, months = Written(), set()
    for batch in [rows] if isinstance(rows, pa.Table) else rows:
        if not batch.num_rows:
            continue
        typed = check(batch, table)
        if conversion is not None:
            i = typed.schema.get_field_index("value")
            typed = typed.set_column(i, typed.schema.field(i), conversion.apply(typed["value"]))
        if series is not None:
            keys = pc.unique(typed["series_key"])
            new = {"series_key": keys}
            if location_of is not None:
                new["location_key"] = pa.array([location_of(k) for k in keys.to_pylist()],
                                               pa.string())  # fmt: skip
            if layer:
                new["location_layer"] = pa.array([layer] * len(keys), pa.string())
            written.series_added += insert_missing(
                series, conform(pa.table(new), columns_of(series)), commit
            )
        if replace_months:
            where = _month_filter(typed)
            table.overwrite(typed, overwrite_filter=where, snapshot_properties=commit)
            written.snapshot_id = _current_snapshot(table)
        else:
            written.snapshot_id = upsert(table, typed, commit)
        written.rows += typed.num_rows
        written.batches += 1
        months |= _months(typed)
    written.months = sorted(months)
    return written
