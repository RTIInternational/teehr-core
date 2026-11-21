"""Dataset types (warehouse data model v1): what each type's table looks like.

Ported from the TEEHR control plane (``teehr-cloud-core``, ``control-plane/src/
dataset_types.py``), which imports them from here. Columns are declared as
:class:`Column` dicts (name, type, required, doc); :func:`arrow_schema` turns them into
a pyarrow schema, and each engine maps the type names to its own (the control plane to
Iceberg, the engine to DuckDB).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import NotRequired, TypedDict

import pyarrow as pa

__all__ = [
    "BBOX",
    "CLIMATOLOGY",
    "COLUMN_TYPES",
    "CROSSWALK",
    "CROSSWALK_COLUMNS",
    "DERIVED",
    "LAYER",
    "LAYER_COLUMNS",
    "NAIVE_SCALE",
    "PERCENTILES",
    "SERIES",
    "TABLE",
    "TIMESERIES",
    "TIMESERIES_KEY",
    "TYPED",
    "Column",
    "arrow_schema",
    "crosswalk_columns",
    "derived_columns",
    "layer_columns",
    "series_columns",
    "timeseries_columns",
]

TABLE = "table"  # untyped: a test path, not part of the model
TIMESERIES = "timeseries"
SERIES = "series"  # a timeseries' series table, created with it
LAYER = "layer"  # locations: geometry + attributes
CROSSWALK = "crosswalk"  # links between two series tables' series
# Derived from an observed timeseries, per its series (role auxiliary; ADR-0019 in
# teehr-cloud-core): built by teehr_core.derived, read by the engine.
CLIMATOLOGY = "climatology"  # mean per series, day of year and hour
PERCENTILES = "percentiles"  # values at fixed percentiles of each series' record
NAIVE_SCALE = "naive_scale"  # mean |y(t) - y(t - lag)| per series and lag (MASE's scale)
DERIVED = (CLIMATOLOGY, PERCENTILES, NAIVE_SCALE)
TYPED = (TIMESERIES, SERIES, LAYER, CROSSWALK, *DERIVED)

BBOX = ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
LAYER_COLUMNS = frozenset({"location_key", "name", "geometry", *BBOX})
CROSSWALK_COLUMNS = frozenset({"left_key", "right_key"})
#: A timeseries' key columns, in order (those it has).
TIMESERIES_KEY = ("series_key", "reference_time", "value_time", "member", "quantile")

#: The column types a dataset may declare, by name, as pyarrow types.
COLUMN_TYPES: dict[str, pa.DataType] = {
    "string": pa.string(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "float32": pa.float32(),
    "float64": pa.float64(),
    "boolean": pa.bool_(),
    "date": pa.date32(),
    "timestamp": pa.timestamp("us"),  # no time zone
    "timestamptz": pa.timestamp("us", tz="UTC"),
    "binary": pa.binary(),
    "decimal(5,4)": pa.decimal128(5, 4),  # quantiles (4 decimal places)
}


class Column(TypedDict):
    """A declared column."""

    name: str
    type: str  # a COLUMN_TYPES key
    required: bool
    doc: NotRequired[str]


def _col(name: str, type_: str, required: bool, doc: str) -> Column:
    return {"name": name, "type": type_, "required": required, "doc": doc}


def arrow_schema(columns: Iterable[Column]) -> pa.Schema:
    """Return the pyarrow schema for declared columns (required: not nullable)."""
    return pa.schema(
        pa.field(
            c["name"],
            COLUMN_TYPES[c["type"]],
            nullable=not c.get("required"),
            metadata={"doc": c["doc"]} if c.get("doc") else None,
        )
        for c in columns
    )


def timeseries_columns(
    optional: Iterable[str], period_form: str, extra: Iterable[Column] = ()
) -> tuple[list[Column], list[str]]:
    """Return a timeseries dataset's columns and key.

    ``optional``: any of ``reference_time``, ``member``, ``quantile``, ``quality``.
    ``period_form`` ``absolute`` adds the window columns. ``extra``: the dataset's own.
    """
    on = set(optional)
    columns = [_col("series_key", "string", True, "The series a value belongs to")]
    if "reference_time" in on:
        columns.append(_col("reference_time", "timestamptz", True, "Issue time"))
    columns.append(_col("value_time", "timestamptz", True, "When the value applies"))
    if "member" in on:
        columns.append(_col("member", "string", True, "Ensemble member"))
    if "quantile" in on:
        columns.append(_col("quantile", "decimal(5,4)", True, "Quantile, 0 to 1"))
    columns.append(_col("value", "float64", False, "Null: a missing value"))
    if "quality" in on:
        columns.append(_col("quality", "string", False, "The source's code(s)"))
    if period_form == "absolute":
        columns.append(_col("period_start", "timestamptz", True, "Window start"))
        columns.append(_col("period_end", "timestamptz", True, "Window end"))
    columns.extend(extra)
    names = {c["name"] for c in columns}
    return columns, [c for c in TIMESERIES_KEY if c in names]


def series_columns(attributes: Iterable[Column] = ()) -> list[Column]:
    """Return a series table's columns: one row per series_key (ADR-0010)."""
    return [
        _col("series_key", "string", True, "Unique series id"),
        _col("location_key", "string", False, "Where the series is"),
        _col("location_layer", "string", False, "Layer of location_key"),
        _col("period", "string", False, "Overrides the dataset's period"),
        _col("resolution", "string", False, "How often values recur (ISO 8601)"),
        *attributes,
    ]


def layer_columns(attributes: Iterable[Column] = ()) -> list[Column]:
    """Return a layer's columns (layers-and-crosswalks.md).

    Geometry as WKB in EPSG:4326, its bounds for file pruning, then attributes.
    """
    return [
        _col("location_key", "string", True, "Unique location id"),
        _col("name", "string", False, "Display name"),
        _col("geometry", "binary", True, "WKB, EPSG:4326 (longitude, latitude)"),
        _col("bbox_xmin", "float64", True, "Westmost longitude"),
        _col("bbox_ymin", "float64", True, "Southmost latitude"),
        _col("bbox_xmax", "float64", True, "Eastmost longitude"),
        _col("bbox_ymax", "float64", True, "Northmost latitude"),
        *attributes,
    ]


def crosswalk_columns(extra: Iterable[Column] = ()) -> list[Column]:
    """Return a crosswalk's columns: a left series key, a right one, then extras."""
    return [
        _col("left_key", "string", True, "A series key of the left series table"),
        _col("right_key", "string", True, "A series key of the right series table"),
        *extra,
    ]


def derived_columns(kind: str) -> tuple[list[Column], list[str]]:
    """Return a derived dataset's columns and key (``kind``: one of :data:`DERIVED`)."""
    series = _col("series_key", "string", True, "The observed series it's derived from")
    if kind == CLIMATOLOGY:
        columns = [
            series,
            _col("day_of_year", "int32", True, "1 to 366"),
            _col("hour", "int32", True, "0 to 23, UTC"),
            _col("value", "float64", True, "Mean of hourly means in the window"),
            _col("count", "int64", True, "Hourly means averaged"),
        ]
        return columns, ["series_key", "day_of_year", "hour"]
    if kind == PERCENTILES:
        columns = [
            series,
            _col("percentile", "float64", True, "Non-exceedance percentile, 0 to 100"),
            _col("value", "float64", True, "Value at the percentile, of hourly means"),
            _col("count", "int64", True, "Hourly means it's of"),
            _col("years", "float64", True, "Days with data / 365.25"),
        ]
        # No key: Iceberg can't key on a float column, and builds replace the table whole.
        return columns, []
    if kind == NAIVE_SCALE:
        columns = [
            series,
            _col("lag", "string", True, "The lag as given (ISO 8601)"),
            _col("lag_seconds", "int64", True, "The lag in seconds"),
            _col("scale", "float64", True, "Mean absolute change over the lag"),
            _col("count", "int64", True, "Differences averaged"),
        ]
        return columns, ["series_key", "lag_seconds"]
    raise ValueError(f"not a derived kind: {kind!r} (one of {', '.join(DERIVED)})")
