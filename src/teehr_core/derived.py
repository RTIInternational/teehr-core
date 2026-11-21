"""Datasets derived from observed series' records: climatology, percentiles, naive scales.

The engine reads them as auxiliary datasets (``generate: climatology``/``blend``,
``fn: percentile`` fields, ``fn: naive_scale`` fields); these functions make their rows. Each
takes a batch of an observed timeseries' rows (``series_key``, ``value_time``, ``value``: any
number of whole series) and returns the derived rows for those series, with the columns
:func:`teehr_core.model.dataset_types.derived_columns` declares. They know nothing of storage,
so a caller batches series as memory allows and writes the results where it keeps them
(``teehr_core.warehouse.build`` for Iceberg tables). Ported from VeriRace's builders, which
can use these.

Statistics are of **hourly means**: each hour of a record weighs the same whether a series
stores values hourly or every few minutes, or switches between them (USGS: hourly before
2023-09, 15-minute after). Non-finite values and nulls are left out.

Needs the ``engine`` extra (DuckDB).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pyarrow as pa

from teehr_core.durations import parse_duration
from teehr_core.engine.connect import arrow_table, memory_connection, register
from teehr_core.model.dataset_types import (
    CLIMATOLOGY,
    NAIVE_SCALE,
    PERCENTILES,
    arrow_schema,
    derived_columns,
)
from teehr_core.sql import lit

__all__ = [
    "BUILDERS",
    "DEFAULT_LAGS",
    "DEFAULT_PERCENTILES",
    "build",
    "climatology",
    "naive_scale",
    "percentiles",
]

#: Every whole percentile (a full flow-duration curve) plus the tails, for droughts and floods.
DEFAULT_PERCENTILES: tuple[float, ...] = (0.1, 0.5, *range(1, 100), 99.5, 99.9)
#: MASE's usual lags for hourly and daily forecasts.
DEFAULT_LAGS: tuple[str, ...] = ("PT1H", "P1D")

_FINITE = "value IS NOT NULL AND isfinite(value)"


def _hourly(source: str = "rows") -> str:
    """Return SQL for the hourly means of ``source``'s finite values."""
    return (
        "SELECT series_key, time_bucket(INTERVAL 1 HOUR, value_time) AS value_time, "
        f"avg(value) AS value FROM {source} WHERE {_FINITE} GROUP BY ALL"
    )


def _run(kind: str, rows: pa.Table, sql: str) -> pa.Table:
    """Run ``sql`` over ``rows`` (as ``rows``) and return it with ``kind``'s columns."""
    con = memory_connection()
    try:
        con.execute("SET TimeZone = 'UTC'")
        register(con, "rows", rows.select(["series_key", "value_time", "value"]))
        out = arrow_table(con, sql)
    finally:
        con.close()
    schema = arrow_schema(derived_columns(kind)[0])
    return out.select(schema.names).cast(schema)


def climatology(rows: pa.Table, *, window_days: int = 15) -> pa.Table:
    """Return each series' mean per day of year and hour (UTC).

    Each day's value averages the hourly means within ``window_days`` around it (a circular
    window, so late December and early January smooth into each other), at the same hour.
    Days with no data within the window get no row.
    """
    if window_days < 1 or window_days % 2 == 0:
        raise ValueError("window_days must be a positive odd number of days")
    half = window_days // 2
    sql = f"""
    WITH raw AS (
        SELECT series_key, dayofyear(value_time) AS doy, hour(value_time) AS hour,
               sum(value) AS s, count(*) AS n
        FROM ({_hourly()}) GROUP BY ALL
    ),
    dense AS (  -- every day of the year, so days without data of their own get values
        SELECT g.series_key, g.hour, days.doy, coalesce(r.s, 0) AS s, coalesce(r.n, 0) AS n
        FROM (SELECT DISTINCT series_key, hour FROM raw) g
        CROSS JOIN range(1, 367) days(doy)
        LEFT JOIN raw r USING (series_key, hour, doy)
    ),
    ext AS (  -- the year three times over, so the window wraps around the new year
        SELECT d.*, d.doy + k * 366 AS d FROM dense d, (VALUES (-1), (0), (1)) t(k)
    ),
    win AS (
        SELECT series_key, hour, d, sum(s) OVER w AS ss, sum(n) OVER w AS nn
        FROM ext
        WINDOW w AS (
            PARTITION BY series_key, hour ORDER BY d
            RANGE BETWEEN {half} PRECEDING AND {half} FOLLOWING
        )
    )
    SELECT series_key, d::INTEGER AS day_of_year, hour::INTEGER AS hour,
           ss / nn AS value, nn::BIGINT AS count
    FROM win WHERE d BETWEEN 1 AND 366 AND nn > 0
    ORDER BY series_key, day_of_year, hour
    """
    return _run(CLIMATOLOGY, rows, sql)


def percentiles(
    rows: pa.Table, *, levels: Sequence[float] = DEFAULT_PERCENTILES, min_years: float = 10
) -> pa.Table:
    """Return each series' values at ``levels`` (non-exceedance percentiles, 0 to 100).

    Of the hourly means, interpolating between them. The 90th is the value not exceeded 90%
    of the time (a high flow), unlike hydrology's Q90. Series with less than ``min_years`` of
    data (days with data / 365.25) get none.
    """
    wanted = sorted({float(p) for p in levels})
    if not wanted or not all(0 < p < 100 for p in wanted):
        raise ValueError(f"percentiles must be between 0 and 100 (exclusive): {list(levels)}")
    quantiles = "[" + ", ".join(repr(p / 100) for p in wanted) + "]"
    pcts = "[" + ", ".join(repr(p) for p in wanted) + "]"
    sql = (
        f"SELECT series_key, unnest({pcts}) AS percentile, unnest(q) AS value, count, years "
        f"FROM (SELECT series_key, quantile_cont(value, {quantiles}) AS q, count(*) AS count, "
        "count(DISTINCT value_time::DATE) / 365.25 AS years "
        f"FROM ({_hourly()}) GROUP BY series_key HAVING years >= {float(min_years)!r}) "
        "ORDER BY series_key, percentile"
    )
    return _run(PERCENTILES, rows, sql)


def naive_scale(rows: pa.Table, *, lags: Sequence[str] = DEFAULT_LAGS) -> pa.Table:
    """Return each series' mean absolute change over each lag: MASE's in-sample scale.

    Differences use values exactly ``lag`` apart (regularly sampled data), not hourly means.
    """
    if not lags:
        raise ValueError("give at least one lag")
    seconds = {lag: int(parse_duration(lag, allow_zero=False).total_seconds()) for lag in lags}
    per_lag = [
        f"SELECT x.series_key, {lit(lag)} AS lag, {sec}::BIGINT AS lag_seconds, "
        "avg(abs(x.value - y.value)) AS scale, count(*) AS count "
        "FROM s x JOIN s y ON y.series_key = x.series_key "
        f"AND y.value_time = x.value_time - to_seconds({sec}) GROUP BY ALL"
        for lag, sec in seconds.items()
    ]
    sql = (
        f"WITH s AS (SELECT series_key, value_time, value FROM rows WHERE {_FINITE}) "
        + " UNION ALL ".join(per_lag)
        + " ORDER BY series_key, lag_seconds"
    )
    return _run(NAIVE_SCALE, rows, sql)


#: Each derived kind's builder: rows of a batch of series and the build's parameters.
BUILDERS: dict[str, Callable[..., pa.Table]] = {
    CLIMATOLOGY: climatology,
    PERCENTILES: percentiles,
    NAIVE_SCALE: naive_scale,
}


def build(kind: str, rows: pa.Table, **params: Any) -> pa.Table:
    """Return ``kind``'s rows for a batch of series (see :data:`BUILDERS`)."""
    if kind not in BUILDERS:
        raise ValueError(f"not a derived kind: {kind!r} (one of {', '.join(BUILDERS)})")
    return BUILDERS[kind](rows, **params)
