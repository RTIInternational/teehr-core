"""The time series data contract: validation and normalization of Arrow tables.

Ported from VeriRace (``verirace/contract.py`` at 668ab7d).

Every dataset is normalized to the contract before it is used (see design doc §3):

- time series: ``series_key``, ``value_time``, ``value``, plus ``reference_time`` for
  forecasts and optional ``member`` / ``quantile``; extra columns pass through unchanged.
  A dataset has one ``variable`` and one ``unit``, kept in its metadata (columns with
  those names are checked against it and dropped).
- series catalog: unique ``series_key``, optional ``period`` and free-form attributes.
- crosswalk: ``left_key`` → ``right_key``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from teehr_core.model.units import UnitError, UnitRegistry

__all__ = [
    "QUANTILE_DECIMALS",
    "QUANTILE_SLACK",
    "TIMESTAMP",
    "ContractError",
    "Role",
    "normalize_crosswalk",
    "normalize_series_catalog",
    "normalize_timeseries",
    "quantile_problems",
]

Role = Literal["observed", "simulated", "forecast"]
TIMESTAMP = pa.timestamp("us", tz="UTC")
#: Quantile levels have this many decimal places (the shared contract), so they compare
#: exactly across systems: stored rounded to it, and more precise ones are refused.
QUANTILE_DECIMALS = 4
#: How far a level may sit from its rounded value and still count as that value (float
#: noise such as 0.05000000000000001).
QUANTILE_SLACK = 1e-9


def quantile_problems(nulls: int, outside: int, finer: int) -> list[str]:
    """Describe quantile levels that break the contract, from counts of each kind."""
    problems = []
    if nulls:
        problems.append(f"{nulls} rows have no quantile")
    if outside:
        problems.append(f"{outside} rows have a quantile outside 0 to 1")
    if finer:
        problems.append(
            f"{finer} rows have a quantile with more than {QUANTILE_DECIMALS} decimal places "
            "(e.g. 0.05 or 0.975 are fine; levels are compared exactly)"
        )
    return problems


class ContractError(ValueError):
    """A table does not satisfy the data contract. ``problems`` lists every issue found."""

    def __init__(self, what: str, problems: Iterable[str]) -> None:
        self.what = what
        self.problems = list(problems)
        super().__init__(f"{what} violates the data contract:\n  - " + "\n  - ".join(self.problems))


def _set(table: pa.Table, name: str, array: pa.Array | pa.ChunkedArray) -> pa.Table:
    return table.set_column(table.schema.get_field_index(name), name, array)


def _to_string(table: pa.Table, name: str, problems: list[str]) -> pa.Table:
    col = table[name]
    if pa.types.is_dictionary(col.type):
        col = pc.cast(col, col.type.value_type)
    if pa.types.is_string(col.type):
        return _set(table, name, col)
    if pa.types.is_large_string(col.type):
        return _set(table, name, pc.cast(col, pa.string()))
    problems.append(f"column {name!r} must be a string, got {col.type}")
    return table


def _to_timestamp(table: pa.Table, name: str, problems: list[str]) -> pa.Table:
    col = table[name]
    if not pa.types.is_timestamp(col.type):
        problems.append(f"column {name!r} must be a timestamp, got {col.type}")
        return table
    if col.type.tz is None:
        problems.append(f"column {name!r} must be timezone-aware (UTC); got a naive timestamp")
        return table
    try:
        return _set(table, name, pc.cast(col, TIMESTAMP))
    except pa.ArrowInvalid as exc:
        problems.append(f"column {name!r} cannot be converted to microsecond UTC: {exc}")
        return table


def _to_float(table: pa.Table, name: str, problems: list[str]) -> pa.Table:
    col = table[name]
    if not (pa.types.is_floating(col.type) or pa.types.is_integer(col.type)):
        problems.append(f"column {name!r} must be numeric, got {col.type}")
        return table
    return _set(table, name, pc.cast(col, pa.float64()))


def _to_levels(table: pa.Table, problems: list[str]) -> pa.Table:
    """Check quantile levels and round them to QUANTILE_DECIMALS places."""
    col = table["quantile"]
    rounded = pc.round(col, QUANTILE_DECIMALS)
    outside = pc.or_(pc.less(col, 0.0), pc.greater(col, 1.0))
    finer = pc.greater(pc.abs(pc.subtract(col, rounded)), QUANTILE_SLACK)
    problems += quantile_problems(
        col.null_count,
        pc.sum(pc.cast(outside, pa.int64())).as_py() or 0,
        pc.sum(pc.cast(finer, pa.int64())).as_py() or 0,
    )
    return _set(table, "quantile", rounded)


def _dataset_level(
    table: pa.Table, name: str, expected: str | None, problems: list[str]
) -> pa.Table:
    """Check a dataset-level value (``variable``, ``unit``) and drop its column.

    A dataset has one variable and one unit, kept in its metadata, not on every row. A
    column with the same name is accepted as a check: it must hold that one value.
    """
    if expected is None:
        problems.append(f"the dataset needs a {name}")
    if name not in table.column_names:
        return table
    col = table[name]
    if pa.types.is_dictionary(col.type):
        col = pc.cast(col, col.type.value_type)
    values = sorted(str(v) for v in pc.unique(col.drop_null()).to_pylist())
    if len(values) > 1:
        problems.append(f"a dataset has one {name}; column {name!r} has {values}")
    elif values and expected is not None and values != [expected]:
        problems.append(f"column {name!r} is {values[0]!r}, not the dataset's {expected!r}")
    return table.drop_columns([name])


def _check_duplicates(table: pa.Table, keys: list[str], problems: list[str]) -> None:
    counts = table.group_by(keys, use_threads=False).aggregate([([], "count_all")])
    dupes = counts.filter(pc.greater(counts["count_all"], 1))
    if dupes.num_rows:
        example = {k: dupes[k][0].as_py() for k in keys}
        problems.append(
            f"{dupes.num_rows} duplicate key(s) on ({', '.join(keys)}); first: {example}"
        )


def normalize_timeseries(
    table: pa.Table,
    *,
    role: Role,
    variable: str | None = None,
    unit: str | None = None,
    name: str = "time series",
    units: UnitRegistry | None = None,
) -> pa.Table:
    """Validate a time series table and return it normalized to the contract.

    Parameters
    ----------
    table
        Input table.
    role
        Dataset role; ``forecast`` requires ``reference_time``.
    variable, unit
        The dataset's variable and unit (required). Columns with these names are checked
        against them and dropped.
    name
        Dataset name, used in error messages.
    units
        Unit registry; when given, the unit must be a known unit.

    Returns
    -------
    pa.Table
        The table with contract columns cast to their canonical types.

    Raises
    ------
    ContractError
        Listing every problem found.
    """
    registry = units
    problems: list[str] = []
    for col in ("series_key", "value_time", "value"):
        if col not in table.column_names:
            problems.append(f"missing required column {col!r}")
    if role == "forecast" and "reference_time" not in table.column_names:
        problems.append("forecast datasets require a 'reference_time' column")
    if problems:
        raise ContractError(name, problems)

    table = _to_string(table, "series_key", problems)
    table = _to_timestamp(table, "value_time", problems)
    table = _to_float(table, "value", problems)
    if "reference_time" in table.column_names:
        table = _to_timestamp(table, "reference_time", problems)
    if "member" in table.column_names:
        table = _to_string(table, "member", problems)
    if "quantile" in table.column_names:
        table = _to_float(table, "quantile", problems)
        if pa.types.is_floating(table["quantile"].type):
            table = _to_levels(table, problems)
    table = _dataset_level(table, "variable", variable, problems)
    table = _dataset_level(table, "unit", unit, problems)
    if unit is not None and registry is not None:
        try:
            registry.get(unit)
        except UnitError as exc:
            problems.append(str(exc))

    for col in ("series_key", "value_time"):
        if col in table.column_names and table[col].null_count:
            problems.append(f"column {col!r} has {table[col].null_count} null values")
    if problems:
        raise ContractError(name, problems)

    keys = [
        c
        for c in ("series_key", "reference_time", "value_time", "member", "quantile")
        if c in table.column_names
    ]
    _check_duplicates(table, keys, problems)
    if problems:
        raise ContractError(name, problems)
    return table


def normalize_series_catalog(table: pa.Table, *, name: str = "series catalog") -> pa.Table:
    """Validate a series catalog (unique ``series_key``, optional ``period`` names)."""
    problems: list[str] = []
    if "series_key" not in table.column_names:
        raise ContractError(name, ["missing required column 'series_key'"])
    table = _to_string(table, "series_key", problems)
    if not problems:
        _check_duplicates(table, ["series_key"], problems)
    if "period" in table.column_names:  # names, checked against the periods when planned
        table = _to_string(table, "period", problems)
    if problems:
        raise ContractError(name, problems)
    return table


def normalize_crosswalk(table: pa.Table, *, name: str = "crosswalk") -> pa.Table:
    """Validate a crosswalk relation (``left_key`` → ``right_key``)."""
    problems: list[str] = []
    for col in ("left_key", "right_key"):
        if col not in table.column_names:
            problems.append(f"missing required column {col!r}")
    if problems:
        raise ContractError(name, problems)
    table = _to_string(table, "left_key", problems)
    table = _to_string(table, "right_key", problems)
    if not problems:
        _check_duplicates(table, ["left_key", "right_key"], problems)
    if problems:
        raise ContractError(name, problems)
    return table
