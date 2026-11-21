"""Writes to TEEHR tables: upserts, inserts of missing keys, merges, and one-commit file writes.

Each takes a loaded table and the rows already conformed to it (:func:`~.conform.conform`), and
commits as whoever loaded the table. ``properties`` go into the commit's snapshot summary: who
wrote it (:data:`PRODUCER`) and anything else worth keeping with it (a flow run, a build's
lineage). Catalog errors (a refused commit, a conflict) are PyIceberg's own.
"""

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyiceberg.expressions import (
    AlwaysTrue,
    And,
    BooleanExpression,
    GreaterThanOrEqual,
    In,
    LessThanOrEqual,
)
from pyiceberg.table import Table
from pyiceberg.table.upsert_util import create_match_filter

from teehr_core.warehouse.conform import keep_unprovided
from teehr_core.warehouse.layout import LAYOUT_DEFAULTS

#: The snapshot summary property naming who wrote a commit (a user, a flow, a build).
PRODUCER = "teehr.producer"


def _snapshot_id(table: Table) -> int | None:
    snapshot = table.current_snapshot()
    return snapshot.snapshot_id if snapshot else None


def key_columns(table: Table) -> list[str]:
    """Return the table's key: its identifier fields, in schema order."""
    schema = table.schema()
    ids = set(schema.identifier_field_ids)
    return [f.name for f in schema.fields if f.field_id in ids]


def key_range(data: pa.Table, key_columns: list[str]) -> BooleanExpression:
    """Return a filter keeping every stored row that could share a key with ``data``.

    The first text key column's values, within each other key column's range (times, dates,
    numbers). Small however many rows ``data`` has (PyIceberg's upsert ORs a clause per row).
    """
    parts: list[BooleanExpression] = []
    listed = False
    for name in key_columns:
        column = data[name]
        kind = column.type
        if pa.types.is_timestamp(kind):  # as microseconds, which Iceberg takes
            column = pc.cast(column.cast(pa.timestamp("us", kind.tz)), pa.int64())
        elif pa.types.is_date32(kind):  # as days
            column = pc.cast(column, pa.int32())
        if pa.types.is_integer(column.type) or pa.types.is_floating(column.type):
            bounds = pc.min_max(column)
            parts += [
                GreaterThanOrEqual(term=name, value=bounds["min"].as_py()),
                LessThanOrEqual(term=name, value=bounds["max"].as_py()),
            ]
        elif not listed and (
            pa.types.is_string(column.type) or pa.types.is_large_string(column.type)
        ):
            parts.append(In(term=name, literals=pc.unique(column).to_pylist()))
            listed = True
    out: BooleanExpression = AlwaysTrue()
    for part in parts:
        out = And(out, part)
    return out


def upsert(table: Table, data: pa.Table, properties: dict[str, str] | None = None) -> int | None:
    """Upsert ``data`` on the table's key, in one commit; return the new snapshot's id.

    The stored rows in its :func:`key_range` that it doesn't replace are kept, and that range is
    overwritten with them and ``data``.
    """
    key = key_columns(table)
    if not key:
        raise ValueError("upsert needs a table with key columns")
    where = key_range(data, key)
    stored = table.scan(row_filter=where).to_arrow()
    rows = data
    if stored.num_rows:
        stored = stored.select(data.column_names).cast(data.schema)
        kept = stored.join(data.select(key), keys=key, join_type="left anti")
        rows = pa.concat_tables([data, kept.select(data.column_names).cast(data.schema)])
    table.overwrite(rows, overwrite_filter=where, snapshot_properties=properties or {})
    return _snapshot_id(table)


def insert_missing(table: Table, data: pa.Table, properties: dict[str, str] | None = None) -> int:
    """Insert the rows whose key isn't in the table yet, leaving existing rows alone.

    Returns how many were inserted (none: no commit).
    """
    result = table.upsert(data, when_matched_update_all=False, snapshot_properties=properties or {})
    return result.rows_inserted


def merge(
    table: Table,
    data: pa.Table,
    key_columns: list[str],
    provided: list[str],
    properties: dict[str, str] | None = None,
) -> int | None:
    """Upsert where only the ``provided`` columns change; return the new snapshot's id.

    A matched row keeps its current values in the others.
    """
    current = table.scan(row_filter=create_match_filter(data, key_columns)).to_arrow()
    table.upsert(
        keep_unprovided(data, current, key_columns, provided),
        snapshot_properties=properties or {},
    )
    return _snapshot_id(table)


def write_file(table: Table, data: pa.Table, file_name: str) -> str:
    """Write rows as a Parquet file in the table's data location; return its path.

    The file isn't part of the table until :func:`commit_files` adds it.
    """
    path = f"{table.metadata.location.rstrip('/')}/data/{file_name}"
    rows = int(
        table.properties.get(
            "write.parquet.row-group-limit", LAYOUT_DEFAULTS["write.parquet.row-group-limit"]
        )
    )
    with table.io.new_output(path).create(overwrite=True) as out:
        pq.write_table(data, out, row_group_size=rows, compression="zstd")
    return path


def commit_files(
    table: Table,
    paths: list[str],
    overwrite_filter: BooleanExpression | None = None,
    properties: dict[str, str] | None = None,
) -> int | None:
    """Replace rows with those of files :func:`write_file` wrote, in one transaction.

    Replaces the rows ``overwrite_filter`` keeps (all, when None). Returns the new snapshot's id.
    """
    with table.transaction() as tx:
        tx.delete(overwrite_filter or AlwaysTrue(), snapshot_properties=properties or {})
        if paths:
            tx.add_files(paths, snapshot_properties=properties or {})
    return _snapshot_id(table)
