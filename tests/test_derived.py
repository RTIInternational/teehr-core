"""Derived datasets: the builders (teehr_core.derived) and the build into Iceberg tables."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pyiceberg.catalog.sql import SqlCatalog

from teehr_core import derived
from teehr_core import warehouse as wh
from teehr_core.model.dataset_types import (
    CLIMATOLOGY,
    DERIVED,
    NAIVE_SCALE,
    PERCENTILES,
    arrow_schema,
    derived_columns,
    timeseries_columns,
)
from teehr_core.warehouse.build import PARAMETERS, SOURCE, SOURCE_SNAPSHOT, build_derived

T0 = datetime(2000, 1, 1, tzinfo=UTC)
TS = pa.timestamp("us", tz="UTC")


def series(key: str, times: list[datetime], values: list[float]) -> pa.Table:
    return pa.table(
        {"series_key": [key] * len(times), "value_time": pa.array(times, TS), "value": values},
    )


def hourly(key: str, hours: int, value=lambda h: float(h), start: datetime = T0) -> pa.Table:
    times = [start + timedelta(hours=h) for h in range(hours)]
    return series(key, times, [value(h) for h in range(hours)])


# --- kinds ---------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", DERIVED)
def test_each_kind_starts_with_series_key_and_keys_no_floats(kind: str) -> None:
    columns, key = derived_columns(kind)
    assert columns[0]["name"] == "series_key"
    types = {c["name"]: c["type"] for c in columns}
    assert all(types[k] not in ("float32", "float64") for k in key)  # Iceberg can't key on them


def test_not_a_derived_kind() -> None:
    with pytest.raises(ValueError, match="not a derived kind"):
        derived_columns("timeseries")
    with pytest.raises(ValueError, match="not a derived kind"):
        derived.build("timeseries", hourly("a", 2))


# --- builders ------------------------------------------------------------------------------


def test_climatology_means_each_day_and_hour_over_years() -> None:
    # Three years; the value at hour h on day d is h + 100 * year: the mean is h + 100.
    parts = []
    for year in range(3):
        start = datetime(2001 + year, 1, 1, tzinfo=UTC)
        parts.append(hourly("a", 24 * 365, lambda h, y=year: float(h % 24 + 100 * y), start))
    out = derived.climatology(pa.concat_tables(parts), window_days=1)
    assert out.schema == arrow_schema(derived_columns(CLIMATOLOGY)[0])
    rows = {(r["day_of_year"], r["hour"]): r for r in out.to_pylist()}
    assert rows[(10, 5)]["value"] == pytest.approx(105.0)
    assert rows[(10, 5)]["count"] == 3
    assert len(rows) == 365 * 24  # 2001-2003 have no 366th day


def test_climatology_window_wraps_around_the_new_year() -> None:
    out = derived.climatology(hourly("a", 24), window_days=15)  # January 1st only
    days = sorted({r["day_of_year"] for r in out.to_pylist()})
    assert days == [*range(1, 9), *range(360, 367)]  # 7 days either side, wrapping (366-day year)


def test_climatology_weighs_each_hour_once() -> None:
    # Hour 0 has four 15-minute values (mean 2.5); hour 1 one value (10): mean of means.
    times = [T0 + timedelta(minutes=15 * i) for i in range(4)] + [T0 + timedelta(hours=1)]
    out = derived.climatology(series("a", times, [1.0, 2.0, 3.0, 4.0, 10.0]), window_days=1)
    by_hour = {r["hour"]: r["value"] for r in out.to_pylist() if r["day_of_year"] == 1}
    assert by_hour == {0: pytest.approx(2.5), 1: pytest.approx(10.0)}


def test_climatology_window_must_be_odd() -> None:
    with pytest.raises(ValueError, match="odd"):
        derived.climatology(hourly("a", 2), window_days=14)


def test_percentiles_match_numpy_and_skip_short_records() -> None:
    days = 11 * 366  # over ten years of daily values (one hourly mean a day)
    times = [T0 + timedelta(days=d) for d in range(days)]
    values = list(np.random.default_rng(1).gamma(2.0, 50.0, days))
    rows = pa.concat_tables([series("long", times, values), hourly("short", 24 * 30)])
    out = derived.percentiles(rows, levels=[10, 50, 99.9], min_years=10)
    assert out.schema == arrow_schema(derived_columns(PERCENTILES)[0])
    assert set(out["series_key"].to_pylist()) == {"long"}
    got = {r["percentile"]: r["value"] for r in out.to_pylist()}
    for p in (10, 50, 99.9):
        assert got[p] == pytest.approx(np.percentile(values, p))
    assert out["count"][0].as_py() == days
    assert out["years"][0].as_py() == pytest.approx(days / 365.25)


def test_percentiles_leave_out_missing_values() -> None:
    rows = series("a", [T0, T0 + timedelta(hours=1), T0 + timedelta(hours=2)],
                  [1.0, None, float("nan")])  # fmt: skip
    out = derived.percentiles(rows, levels=[50], min_years=0)
    assert out.to_pylist()[0]["value"] == 1.0 and out.to_pylist()[0]["count"] == 1


def test_percentiles_levels_are_checked() -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        derived.percentiles(hourly("a", 2), levels=[0, 50])


def test_naive_scale_is_the_mean_absolute_change_per_lag() -> None:
    values = [float(h * h % 17) for h in range(24 * 5)]
    rows = hourly("a", len(values), lambda h: values[h])
    out = derived.naive_scale(rows, lags=["PT1H", "P1D"])
    assert out.schema == arrow_schema(derived_columns(NAIVE_SCALE)[0])
    got = {r["lag"]: r for r in out.to_pylist()}
    v = np.array(values)
    assert got["PT1H"]["scale"] == pytest.approx(np.mean(np.abs(v[1:] - v[:-1])))
    assert got["PT1H"]["count"] == len(v) - 1 and got["PT1H"]["lag_seconds"] == 3600
    assert got["P1D"]["scale"] == pytest.approx(np.mean(np.abs(v[24:] - v[:-24])))


def test_builders_keep_series_apart() -> None:
    rows = pa.concat_tables([hourly("a", 48), hourly("b", 48, lambda h: 2.0 * h)])
    out = derived.naive_scale(rows, lags=["PT1H"])
    assert {r["series_key"]: r["scale"] for r in out.to_pylist()} == {"a": 1.0, "b": 2.0}


# --- building into Iceberg -----------------------------------------------------------------


@pytest.fixture
def catalog(tmp_path: Path) -> SqlCatalog:
    catalog = SqlCatalog("t", uri=f"sqlite:///{tmp_path}/c.db", warehouse=f"file://{tmp_path}/w")
    catalog.create_namespace("public")
    return catalog


def create(catalog: SqlCatalog, name: str, columns, key, properties):
    schema = wh.iceberg_schema(columns, key)
    layout = wh.Layout(sort=tuple(key), properties=properties)
    return catalog.create_table(
        ("public", name), schema=schema, sort_order=wh.sort_order(schema, layout.sort),
        properties=wh.table_properties(layout),
    )  # fmt: skip


@pytest.fixture
def source(catalog: SqlCatalog):
    columns, key = timeseries_columns([], "instant")
    table = create(catalog, "obs", columns, key,
                   {"teehr.type": "timeseries", "teehr.role": "observed"})  # fmt: skip
    data = pa.concat_tables([hourly(k, 48, lambda h, i=i: float(h * (i + 1))) for i, k in
                             enumerate(["a", "b", "c"])])  # fmt: skip
    table.append(wh.conform(data, columns))
    return table.refresh()


def target(catalog: SqlCatalog, kind: str = NAIVE_SCALE, name: str = "obs_naive_scale"):
    columns, key = derived_columns(kind)
    return create(catalog, name, columns, key, {"teehr.type": kind})


def test_a_build_replaces_the_dataset_in_one_commit_with_its_lineage(catalog, source) -> None:
    table = target(catalog)
    result = build_derived(
        source, table, NAIVE_SCALE, ["a", "b", "c"],
        params={"lags": ["PT1H"]}, batch_series=2, producer="flow:test",
    )  # fmt: skip
    assert (result.rows, result.series, result.batches) == (3, 3, 2)
    table = table.refresh()
    got = {r["series_key"]: r["scale"] for r in table.scan().to_arrow().to_pylist()}
    assert got == {"a": 1.0, "b": 2.0, "c": 3.0}
    summary = table.current_snapshot().summary
    assert summary[SOURCE] == "public.obs" and summary[wh.PRODUCER] == "flow:test"
    assert summary[SOURCE_SNAPSHOT] == str(source.current_snapshot().snapshot_id)
    assert summary[PARAMETERS] == '{"lags": ["PT1H"]}'

    before = len(table.metadata.snapshots)
    build_derived(source, table, NAIVE_SCALE, ["a"], params={"lags": ["PT1H"]})
    table = table.refresh()
    assert table.scan().to_arrow()["series_key"].to_pylist() == ["a"]  # replaced
    assert len(table.metadata.snapshots) - before <= 2  # one commit (a delete and an add)


def test_a_failed_build_leaves_the_dataset_and_no_files(catalog, source, monkeypatch) -> None:
    table = target(catalog)
    build_derived(source, table, NAIVE_SCALE, ["a", "b", "c"], params={"lags": ["PT1H"]})
    table = table.refresh()
    snapshot = table.current_snapshot().snapshot_id
    data_dir = Path(table.metadata.location.removeprefix("file://")) / "data"
    files = set(data_dir.rglob("*.parquet"))
    calls = iter(range(10))
    real = derived.build

    def flaky(kind, rows, **params):
        if next(calls) == 1:
            raise RuntimeError("out of memory")
        return real(kind, rows, **params)

    monkeypatch.setattr(derived, "build", flaky)
    with pytest.raises(RuntimeError):
        build_derived(source, table, NAIVE_SCALE, ["a", "b", "c"], params={"lags": ["PT1H"]},
                      batch_series=1)  # fmt: skip
    assert table.refresh().current_snapshot().snapshot_id == snapshot
    assert set(data_dir.rglob("*.parquet")) == files


def test_a_build_reads_one_period(catalog, source) -> None:
    table = target(catalog)
    result = build_derived(source, table, NAIVE_SCALE, ["a"], params={"lags": ["PT1H"]},
                           start=T0 + timedelta(hours=24))  # fmt: skip
    assert result.first == T0 + timedelta(hours=24)
    assert table.refresh().scan().to_arrow()["count"].to_pylist() == [23]


def test_builds_are_from_observations_into_their_kind(catalog, source) -> None:
    columns, key = timeseries_columns(["reference_time"], "instant")
    forecast = create(catalog, "fcst", columns, key, {"teehr.role": "forecast"})
    with pytest.raises(ValueError, match="built from observations"):
        build_derived(forecast, target(catalog), NAIVE_SCALE, ["a"])
    with pytest.raises(ValueError, match="not a naive_scale"):
        build_derived(source, target(catalog, PERCENTILES, "p"), NAIVE_SCALE, ["a"])
