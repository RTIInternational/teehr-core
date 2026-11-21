"""TEEHR's warehouse rules on Iceberg tables: layout, conforming rows, writes and change detection.

Shared by everything that writes TEEHR tables (the control plane, its jobs, and platform producers
such as ingest and build flows), so the rules live in one place (ADR-0019 in teehr-cloud-core).
Needs the ``iceberg`` extra (PyIceberg). Catalogs and credentials are the caller's: functions take
a loaded :class:`pyiceberg.table.Table`.

- :mod:`.layout`: a table's schema, partitioning, sort order and properties from declared columns.
- :mod:`.conform`: fitting submitted rows to a dataset's columns, or saying why not.
- :mod:`.write`: upserts, inserts of missing keys, merges, and files committed in one transaction.
- :mod:`.changes`: which months of a timeseries changed between two snapshots.
"""

from teehr_core.warehouse.changes import Changes, changes
from teehr_core.warehouse.conform import DataMismatch, conform, duplicate_keys, keep_unprovided
from teehr_core.warehouse.layout import (
    COLUMN_TYPES,
    LAYOUT_DEFAULTS,
    Layout,
    iceberg_schema,
    partition_spec,
    sort_order,
    table_properties,
)
from teehr_core.warehouse.write import (
    PRODUCER,
    commit_files,
    insert_missing,
    key_columns,
    key_range,
    merge,
    upsert,
    write_file,
)

__all__ = [
    "COLUMN_TYPES",
    "LAYOUT_DEFAULTS",
    "PRODUCER",
    "Changes",
    "DataMismatch",
    "Layout",
    "changes",
    "commit_files",
    "conform",
    "duplicate_keys",
    "iceberg_schema",
    "insert_missing",
    "keep_unprovided",
    "key_columns",
    "key_range",
    "merge",
    "partition_spec",
    "sort_order",
    "table_properties",
    "upsert",
    "write_file",
]
