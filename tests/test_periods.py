"""Periods (ADR-0009): definitions, their windows, and pairing on them."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pyarrow as pa
import pytest

from teehr_core import engine
from teehr_core.engine import Catalog, Dataset, PlanError
from teehr_core.model.contract import TIMESTAMP
from teehr_core.model.periods import Period, PeriodError, PeriodRegistry, parse_iso_duration

APR_JUL = Period("apr_jul_volume", "calendar", start="--04-01", end="--07-31")
WATER_YEAR = Period("water_year", "calendar", start="--10-01", end="--09-30")


def t(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def test_iso_durations() -> None:
    assert parse_iso_duration("PT1H") == timedelta(hours=1)
    assert parse_iso_duration("PT15M") == timedelta(minutes=15)
    assert parse_iso_duration("P1DT6H") == timedelta(days=1, hours=6)
    assert parse_iso_duration("P2W") == timedelta(weeks=2)
    for bad in ("1h", "P", "PT", "P1M", "P1Y", "PT0S", "P1DT"):
        with pytest.raises(ValueError):
            parse_iso_duration(bad)


def test_definitions_are_checked() -> None:
    PeriodRegistry([Period("hour_starting", "duration", "PT1H", "start"), APR_JUL, WATER_YEAR])
    bad = [
        (Period("x", "lead"), "form must be one of"),
        (Period("x", "duration", "PT1H"), "needs value_time_at"),
        (Period("x", "duration", "1h", "start"), "length: invalid duration"),
        (Period("x", "duration", "PT1H", "middle"), "start or end"),
        (Period("x", "instant", length="PT1H"), "has no length"),
        (Period("x", "calendar", start="--04-01"), "needs end"),
        (Period("x", "calendar", start="--02-29", end="--03-31"), "isn't a day in every year"),
        (Period("x", "calendar", start="04-01", end="--07-31"), "expected --MM-DD"),
    ]
    for period, match in bad:
        with pytest.raises(PeriodError, match=match):
            PeriodRegistry([period])
    with pytest.raises(PeriodError, match="defined twice"):
        PeriodRegistry([APR_JUL, APR_JUL])
    with pytest.raises(PeriodError, match="unknown period 'nope'"):
        PeriodRegistry([APR_JUL]).get("nope")


def test_the_shared_periods(periods: PeriodRegistry) -> None:
    common = ["day_ending", "day_starting", "hour_ending", "hour_starting", "instant"]
    assert periods.names() == common
    assert periods.get("hour_ending") == Period(
        "hour_ending", "duration", "PT1H", "end", description="The hour ending at value_time"
    )


def _windows(period: Period, times: list[str]) -> list[tuple[Any, Any]]:
    import duckdb

    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    start, end = period.window_sql("w")
    values = ", ".join(f"('{x}'::TIMESTAMPTZ)" for x in times)
    sql = f"SELECT {start}, {end} FROM (VALUES {values}) w(value_time)"
    return [tuple(r) for r in con.execute(sql).fetchall()]


def test_calendar_windows_contain_their_value_time() -> None:
    occurrence = (t("2025-04-01"), t("2025-08-01"))
    assert (
        _windows(APR_JUL, ["2025-04-01", "2025-05-15 13:00", "2025-07-31 23:59"])
        == [occurrence] * 3
    )
    assert _windows(APR_JUL, ["2025-03-31 23:00", "2025-08-01", "2025-12-01"]) == [(None, None)] * 3
    # A span across January 1 needs no water-year convention.
    wy2025 = (t("2024-10-01"), t("2025-10-01"))
    assert _windows(WATER_YEAR, ["2024-10-01", "2025-01-01", "2025-09-30 23:00"]) == [wy2025] * 3
    assert _windows(WATER_YEAR, ["2025-10-01"]) == [(t("2025-10-01"), t("2026-10-01"))]


def _volumes(name: str, role: str, times: list[str], values: list[float], **extra: Any) -> Dataset:
    cols: dict[str, Any] = {"series_key": ["basin-a"] * len(times)}
    cols.update({k: pa.array([t(x) for x in v], TIMESTAMP) for k, v in extra.items()})
    cols["value_time"] = pa.array([t(x) for x in times], TIMESTAMP)
    cols["value"] = pa.array(values, pa.float64())
    meta = {"variable": "volume", "unit": "mm", "statistic": "sum", "period": "apr_jul_volume"}
    return Dataset(name, role, pa.table(cols), **meta)  # type: ignore[arg-type]


def test_calendar_values_pair_on_their_occurrence(units: Any) -> None:
    """April-July volumes: a forecast pairs with the observation of the same season."""
    observed = _volumes("obs", "observed", ["2024-04-01", "2025-04-01"], [100.0, 200.0])
    # Labelled anywhere in the season; one for a September value pairs with nothing.
    forecast = _volumes(
        "fc",
        "forecast",
        ["2024-06-30", "2025-04-01", "2025-07-31 12:00", "2025-09-01"],
        [110.0, 190.0, 205.0, 999.0],
        reference_time=["2024-01-01", "2025-01-01", "2025-03-01", "2025-01-01"],
    )
    catalog = Catalog([observed, forecast], units, PeriodRegistry([APR_JUL]))
    spec = {
        "version": 1,
        "name": "volumes",
        "inputs": {"primary": {"dataset": "obs"}, "secondary": {"dataset": "fc"}},
        "pair": {"on_attribute": "series_key"},
        "aggregate": [
            {"group_by": ["secondary_dataset"], "metrics": ["count", "mean_absolute_error"]}
        ],
        "output": {"include_pairs": True},
    }
    with pytest.raises(PlanError, match="series attribute"):
        engine.evaluate(spec, catalog)  # pairing on an attribute needs series catalogs
    series = pa.table({"series_key": ["basin-a"]})
    catalog = Catalog(
        [Dataset(**{**d.__dict__, "series": series}) for d in (observed, forecast)],
        units,
        PeriodRegistry([APR_JUL]),
    )
    result = engine.evaluate(spec, catalog)
    assert result.table.to_pylist()[0]["count"] == 3
    assert result.table.to_pylist()[0]["mean_absolute_error"] == pytest.approx((10 + 10 + 5) / 3)
    assert result.pairs is not None
    pairs = sorted(result.pairs.to_pylist(), key=lambda p: p["secondary_value"])
    assert [(p["primary_value"], p["secondary_value"]) for p in pairs] == [
        (100.0, 110.0),
        (200.0, 190.0),
        (200.0, 205.0),
    ]
    assert pairs[2]["period_start"] == t("2025-04-01")
    assert pairs[2]["period_end"] == t("2025-08-01")
