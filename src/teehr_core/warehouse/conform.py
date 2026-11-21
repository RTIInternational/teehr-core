"""Fitting submitted rows to a dataset's declared columns, or saying why not."""

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from teehr_core.warehouse.layout import iceberg_schema


class DataMismatch(ValueError):
    """Submitted data (or a filter) doesn't fit the dataset's declared columns."""


def duplicate_keys(data: pa.Table, key_columns: list[str]) -> int:
    """How many key values occur more than once in ``data``."""
    if not key_columns or data.num_rows == 0:
        return 0
    counts = data.group_by(key_columns).aggregate([([], "count_all")])
    return pc.sum(pc.greater(counts.column("count_all"), 1)).as_py() or 0


def keep_unprovided(
    data: pa.Table, current: pa.Table, key_columns: list[str], provided: list[str]
) -> pa.Table:
    """Fill the columns ``data`` didn't provide from ``current`` (the matching stored rows).

    Null for new rows. Same schema and row order as ``data``.
    """
    missing = [c for c in data.column_names if c not in provided]
    if not missing:
        return data
    order = "__row"
    left = data.select([*key_columns, *[c for c in provided if c not in key_columns]])
    left = left.append_column(order, pa.array(range(data.num_rows), pa.int64()))
    right = current.select([*key_columns, *missing]).cast(
        pa.schema([data.schema.field(c) for c in [*key_columns, *missing]])
    )
    joined = left.join(right, keys=key_columns, join_type="left outer")
    joined = joined.sort_by(order)
    return pa.Table.from_arrays([joined.column(c) for c in data.column_names], schema=data.schema)


_HAS_OFFSET = r"(?:[Zz]|[+-]\d{2}(?::?\d{2})?)$"


def _localize(column: pa.ChunkedArray, field: pa.Field, timezone: str) -> pa.ChunkedArray:
    """Read zone-less times in ``timezone`` for a timestamptz column, then UTC.

    Values that carry an offset keep it. A local time that doesn't exist or occurs twice
    (daylight-saving changes) is refused rather than guessed.
    """
    naive_type = pa.timestamp(field.type.unit)
    with_offset = None
    if pa.types.is_timestamp(column.type):
        naive = column.cast(naive_type, safe=True)
    else:  # text
        has_offset = pc.match_substring_regex(column, _HAS_OFFSET)
        nulls = pa.nulls(len(column), column.type)
        with_offset = pc.if_else(has_offset, column, nulls).cast(field.type, safe=True)
        naive = pc.if_else(has_offset, nulls, column).cast(naive_type, safe=True)
    try:
        local = pc.assume_timezone(naive, timezone, ambiguous="raise", nonexistent="raise").cast(
            field.type
        )
    except pa.ArrowInvalid as exc:
        raise DataMismatch(
            f"column '{field.name}': a time doesn't exist or is ambiguous in "
            f"{timezone} (daylight-saving change); send it with an offset ({exc})"
        ) from exc
    return local if with_offset is None else pc.coalesce(with_offset, local)


def _decimal_places_error(field: pa.Field) -> DataMismatch:
    return DataMismatch(
        f"column '{field.name}': values must have at most {field.type.scale} decimal places"
    )


def conform(data: pa.Table, columns: list[dict[str, Any]], timezone: str | None = None) -> pa.Table:
    """Fit submitted data to the dataset's declared columns, or say why not.

    Column order doesn't matter, and a missing optional column becomes nulls. Types convert only
    where nothing is lost (e.g. int32 -> int64, a time zone -> UTC); a timestamp's
    time-zone-ness must match its column, unless ``timezone`` says what zone-less times in a
    timestamptz column are in.
    """
    target = iceberg_schema(columns).as_arrow()
    declared = {f.name for f in target}
    extra = sorted(set(data.column_names) - declared)
    if extra:
        raise DataMismatch(f"columns not in the dataset: {', '.join(extra)}")
    if len(set(data.column_names)) != len(data.column_names):
        raise DataMismatch("duplicate column names")
    arrays = []
    for field in target:
        if field.name not in data.column_names:
            if not field.nullable:
                raise DataMismatch(f"missing required column '{field.name}'")
            arrays.append(pa.nulls(data.num_rows, type=field.type))
            continue
        column = data.column(field.name)
        if not field.nullable and column.null_count:
            raise DataMismatch(
                f"column '{field.name}' is required but has {column.null_count} nulls"
            )
        source = column.type
        is_text = pa.types.is_string(source) or pa.types.is_large_string(source)
        zoned_target = pa.types.is_timestamp(field.type) and field.type.tz
        naive_source = pa.types.is_timestamp(source) and source.tz is None
        if timezone and zoned_target and (naive_source or is_text):
            try:
                arrays.append(_localize(column, field, timezone))
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                raise DataMismatch(
                    f"column '{field.name}': can't read times as {field.type} ({exc})"
                ) from exc
            continue
        both_times = pa.types.is_timestamp(source) and pa.types.is_timestamp(field.type)
        if both_times and (source.tz is None) != (field.type.tz is None):
            kind = "time-zone-aware" if field.type.tz else "time-zone-naive"
            hint = " (or pass timezone=...)" if field.type.tz else ""
            raise DataMismatch(f"column '{field.name}' must be {kind}{hint}")
        try:
            converted = column.cast(field.type, safe=True)
            if pa.types.is_decimal(field.type) and pa.types.is_floating(source):
                # Arrow rounds floats to the scale; refuse rather than round.
                back = converted.cast(pa.float64())
                same = pc.or_kleene(pc.equal(back, column.cast(pa.float64())), pc.is_null(column))
                if not pc.all(same).as_py():
                    raise _decimal_places_error(field)
            arrays.append(converted)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            if pa.types.is_decimal(field.type) and "Rescaling" in str(exc):
                raise _decimal_places_error(field) from exc
            if is_text and pa.types.is_timestamp(field.type):
                raise DataMismatch(
                    f"column '{field.name}': can't parse a value as {field.type} "
                    "(timestamps with a time zone need an offset such as Z, or "
                    f"timezone=...; without one must have none) ({exc})"
                ) from exc
            if is_text:
                raise DataMismatch(
                    f"column '{field.name}': can't parse a value as {field.type} ({exc})"
                ) from exc
            raise DataMismatch(
                f"column '{field.name}': can't convert {source} to {field.type} "
                f"without losing values ({exc})"
            ) from exc
    return pa.Table.from_arrays(arrays, schema=target)
