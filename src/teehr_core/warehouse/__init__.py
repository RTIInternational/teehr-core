"""TEEHR's warehouse rules on Iceberg tables: layout, conforming rows, writes and change detection.

Shared by everything that writes TEEHR tables (the control plane, its jobs, and platform producers
such as ingest and build flows), so the rules live in one place (ADR-0019 in teehr-cloud-core).
Needs the ``iceberg`` extra (PyIceberg). Catalogs and credentials are the caller's: functions take
a loaded :class:`pyiceberg.table.Table`.

- :mod:`.layout`: a table's schema, partitioning, sort order and properties from declared columns.
- :mod:`.conform`: fitting submitted rows to a dataset's columns, or saying why not.
- :mod:`.write`: upserts, inserts of missing keys, merges, and files committed in one transaction.
- :mod:`.ingest`: a producer's write into a registered timeseries (a flow, or a person's
  one-off ingest): checked, new series added, upserted; from any catalog.
- :mod:`.changes`: which months of a timeseries changed since a recorded state of it.
- :mod:`.build`: building a derived dataset into its table (also needs the ``engine``
  extra; import it on its own).
"""

from teehr_core.warehouse.changes import Changes, TableState, changes, table_state
from teehr_core.warehouse.conform import DataMismatch, conform, duplicate_keys, keep_unprovided
from teehr_core.warehouse.ingest import (
    Written,
    columns_of,
    combine,
    location_keys,
    stage_by_month,
    staged_months,
    write_timeseries,
)
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
    "TableState",
    "Written",
    "changes",
    "columns_of",
    "combine",
    "commit_files",
    "conform",
    "duplicate_keys",
    "iceberg_schema",
    "insert_missing",
    "keep_unprovided",
    "key_columns",
    "key_range",
    "location_keys",
    "merge",
    "partition_spec",
    "sort_order",
    "stage_by_month",
    "staged_months",
    "table_properties",
    "table_state",
    "upsert",
    "write_file",
    "write_timeseries",
]
