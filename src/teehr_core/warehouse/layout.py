"""How a TEEHR table is laid out: schema, partitioning, sort order and properties."""

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table.sorting import UNSORTED_SORT_ORDER, SortField, SortOrder
from pyiceberg.transforms import IdentityTransform, MonthTransform, YearTransform
from pyiceberg.types import (
    BinaryType,
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IcebergType,
    IntegerType,
    LongType,
    NestedField,
    StringType,
    TimestampType,
    TimestamptzType,
)

#: The column types a dataset may declare, by name.
COLUMN_TYPES: dict[str, IcebergType] = {
    "string": StringType(),
    "int32": IntegerType(),
    "int64": LongType(),
    "float32": FloatType(),
    "float64": DoubleType(),
    "boolean": BooleanType(),
    "date": DateType(),
    "timestamp": TimestampType(),  # no time zone
    "timestamptz": TimestamptzType(),  # UTC
    "binary": BinaryType(),
    "decimal(5,4)": DecimalType(5, 4),  # quantiles (4 decimal places)
}

PartitionTransform = Literal["month", "year"]

#: How every new table's files are laid out, unless its layout says otherwise: files of about
#: 64 MB, in row groups of a few series each, so a read of a range of series keys skips most of a
#: file (run shards are key ranges). Spark (sorted rewrites) sizes row groups in encoded bytes,
#: PyIceberg in rows; 1 MB keeps Spark's groups under 1M rows of series data. Starting points.
LAYOUT_DEFAULTS = {
    "write.target-file-size-bytes": str(64 * 1024**2),
    "write.parquet.row-group-size-bytes": str(1024**2),
    "write.parquet.row-group-limit": "150000",
}
# PyIceberg warns that it doesn't size row groups in bytes; the row limit applies.
warnings.filterwarnings(
    "ignore", message=r".*write\.parquet\.row-group-size-bytes.*not implemented"
)


@dataclass(frozen=True)
class Layout:
    """How a new table is laid out.

    Partitioned by a time transform of one column, sorted by others, with table properties (the
    dataset's declaration, mirrored from the registry as ``teehr.*``).
    """

    partition: tuple[str, PartitionTransform] | None = None
    sort: tuple[str, ...] = ()
    properties: dict[str, str] | None = None


def table_properties(layout: Layout) -> dict[str, str]:
    """Return a new table's properties: the layout defaults, then the layout's own."""
    return {**LAYOUT_DEFAULTS, **(layout.properties or {})}


def iceberg_schema(
    columns: Iterable[dict[str, Any]], key_columns: Iterable[str] | None = None
) -> Schema:
    """Return the Iceberg schema for declared columns ({name, type, required, doc}).

    The key columns become its identifier fields.
    """
    fields = [
        NestedField(
            field_id=i,
            name=col["name"],
            field_type=COLUMN_TYPES[col["type"]],
            required=bool(col.get("required")),
            doc=col.get("doc"),
        )
        for i, col in enumerate(columns, start=1)
    ]
    ids = {f.name: f.field_id for f in fields}
    return Schema(*fields, identifier_field_ids=[ids[name] for name in key_columns or ()])


def partition_spec(
    schema: Schema, partition: tuple[str, PartitionTransform] | None
) -> PartitionSpec:
    """Partitioning by a month or year transform of one column (or none)."""
    if partition is None:
        return UNPARTITIONED_PARTITION_SPEC
    column, transform = partition
    field = schema.find_field(column)
    return PartitionSpec(
        PartitionField(
            source_id=field.field_id,
            field_id=1000,
            transform=MonthTransform() if transform == "month" else YearTransform(),
            name=f"{column}_{transform}",
        )
    )


def sort_order(schema: Schema, columns: tuple[str, ...]) -> SortOrder:
    """Sorted by ``columns``, in order (or unsorted)."""
    if not columns:
        return UNSORTED_SORT_ORDER
    return SortOrder(
        *(
            SortField(source_id=schema.find_field(c).field_id, transform=IdentityTransform())
            for c in columns
        )
    )
