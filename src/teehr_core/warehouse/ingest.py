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

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc
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
) -> Written:
    """Write ``rows`` (one table, or batches of them) into the timeseries dataset ``name``.

    Each batch is checked against the dataset's columns, its new series are added to the series
    table, and it is upserted on the dataset's key in one commit: a batch is written whole or
    not at all, and writing it again changes nothing. Values are in ``unit`` (default: the
    dataset's own); another unit is converted with ``units``. A new series is at the location
    ``location_of(series_key)`` in the dataset's location layer (e.g. ``lambda key: key`` when
    series keys are location keys; default: unset). ``producer`` names who wrote it (a flow
    run, a notebook) in each commit, with ``properties``.

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
        written.snapshot_id = upsert(table, typed, commit)
        written.rows += typed.num_rows
        written.batches += 1
        months |= _months(typed)
    written.months = sorted(months)
    return written
