"""teehr_core.fetch.usgs against frames shaped like the Water Data API's (no network)."""

import os
from datetime import UTC, datetime

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("dataretrieval")

from teehr_core.fetch import usgs  # noqa: E402

START = datetime(2026, 11, 20, tzinfo=UTC)
END = datetime(2026, 11, 20, 2, tzinfo=UTC)


def frame(*rows, daily=False):
    """API rows of (site, time, value ft3/s[, time series id])."""
    times = [pd.Timestamp(r[1]) for r in rows]
    return pd.DataFrame({
        "time_series_id": [r[3] if len(r) > 3 else f"ts-{r[0]}" for r in rows],
        "monitoring_location_id": [f"USGS-{r[0]}" for r in rows],
        "time": pd.Series(times, dtype="datetime64[us]" if daily else "datetime64[us, UTC]"),
        "value": [r[2] for r in rows],
        "unit_of_measure": "ft^3/s",
    })  # fmt: skip


def as_tuples(table):
    return [(r["series_key"], r["value_time"].strftime("%Y-%m-%dT%H:%M"), round(r["value"], 4))
            for r in table.to_pylist()]  # fmt: skip


def test_keys():
    assert {usgs.series_key(s) for s in ("usgs-01646500", "USGS-01646500", "01646500")} == {
        "usgs-01646500"
    }
    assert usgs.site_number("usgs-01646500") == "01646500"


def test_rows_on_the_hour_in_m3s_without_nodata():
    rows = usgs.to_rows(frame(
        ("01646500", "2026-11-20T00:00Z", 100.0),
        ("01646500", "2026-11-20T00:15Z", 110.0),  # not on the hour
        ("01646500", "2026-11-20T01:00Z", -999999.0),  # no data
        ("02339495", "2026-11-20T01:00Z", None),  # missing
        ("02339495", "2026-11-20T00:00Z", 17.4),
    ))  # fmt: skip
    assert rows.schema == usgs.SCHEMA
    assert sorted(as_tuples(rows)) == [
        ("usgs-01646500", "2026-11-20T00:00", round(100 * usgs.CFS_TO_M3S, 4)),
        ("usgs-02339495", "2026-11-20T00:00", round(17.4 * usgs.CFS_TO_M3S, 4)),
    ]
    assert len(usgs.to_rows(frame(("01646500", "2026-11-20T00:15Z", 1.0)), hourly=False)) == 1


def test_quality_is_approval_then_qualifiers():
    assert usgs.quality("Provisional", None) == "Provisional"
    assert usgs.quality("Approved", ["ESTIMATED", "ICE"]) == "Approved,ESTIMATED,ICE"
    assert usgs.quality(None, float("nan")) is None
    api = frame(("01646500", "2026-11-20T00:00Z", 1.0), ("01646500", "2026-11-20T01:00Z", 2.0))
    api["approval_status"] = ["Approved", "Provisional"]
    api["qualifier"] = [["ESTIMATED"], None]
    assert usgs.to_rows(api)["quality"].to_pylist() == ["Approved,ESTIMATED", "Provisional"]


def test_daily_values_at_midnight_utc():
    rows = usgs.to_rows(frame(("01646500", "2026-11-18", 1610.0), daily=True), service="dv")
    assert as_tuples(rows) == [
        ("usgs-01646500", "2026-11-18T00:00", round(1610 * usgs.CFS_TO_M3S, 4))
    ]
    assert str(rows["value_time"].type) == "timestamp[us, tz=UTC]"


def test_one_time_series_per_site(caplog):
    rows = usgs.to_rows(frame(
        ("01646500", "2026-11-20T00:00Z", 1.0, "main"),
        ("01646500", "2026-11-20T01:00Z", 2.0, "main"),
        ("01646500", "2026-11-20T00:00Z", 9.0, "backup"),
    ))  # fmt: skip
    assert [r[2] for r in as_tuples(rows)] == [round(v * usgs.CFS_TO_M3S, 4) for v in (1.0, 2.0)]
    assert "01646500" in caplog.text


def test_the_series_still_reporting_wins_over_a_longer_old_one():
    rows = usgs.to_rows(frame(
        ("01646500", "2026-11-20T00:00Z", 1.0, "old"),
        ("01646500", "2026-11-20T01:00Z", 2.0, "old"),
        ("01646500", "2026-11-20T00:00Z", 8.0, "current"),  # fewer values ...
        ("01646500", "2026-11-20T03:00Z", 9.0, "current"),  # ... but the latest
    ))  # fmt: skip
    assert [r[2] for r in as_tuples(rows)] == [round(v * usgs.CFS_TO_M3S, 4) for v in (8.0, 9.0)]


def test_nothing_is_an_empty_batch():
    assert usgs.to_rows(pd.DataFrame()).num_rows == 0
    assert usgs.to_rows(None).schema == usgs.SCHEMA


def test_fetch_batches_sites(monkeypatch):
    calls = []

    def request(service, sites, start, end):
        calls.append((service, sites))
        return frame(*[(usgs.site_number(s), "2026-11-20T00:00Z", 1.0) for s in sites])

    monkeypatch.setattr(usgs, "_request", request)
    batches = list(usgs.fetch_streamflow(["usgs-1", "2", "usgs-1", "3"], START, END, batch_sites=2))
    assert calls == [("iv", ["usgs-1", "2"]), ("iv", ["3"])]
    assert [b.num_rows for b in batches] == [2, 1]


def test_fetched_rows_write(monkeypatch, tmp_path):
    """The rows fit a timeseries dataset as the control plane registers one."""
    pytest.importorskip("pyiceberg")
    from pyiceberg.catalog.sql import SqlCatalog

    from teehr_core import warehouse as wh
    from teehr_core.model.dataset_types import series_columns, timeseries_columns

    catalog = SqlCatalog("t", uri=f"sqlite:///{tmp_path}/c.db", warehouse=f"file://{tmp_path}/w")
    catalog.create_namespace("public")
    columns, key = timeseries_columns(["quality"], "instant")
    for name, cols, k, props in (
        ("usgs_streamflow", columns, key, {"teehr.type": "timeseries", "teehr.unit": usgs.UNIT,
                                           "teehr.series_table": "usgs_streamflow_series"}),
        ("usgs_streamflow_series", series_columns(), ["series_key"], {"teehr.type": "series"}),
    ):  # fmt: skip
        catalog.create_table(("public", name), schema=wh.iceberg_schema(cols, k), properties=props)
    monkeypatch.setattr(usgs, "_request", lambda *a: frame(("01646500", "2026-11-20T00:00Z", 1.0)))

    written = wh.write_timeseries(
        catalog, "public", "usgs_streamflow", usgs.fetch_streamflow(["01646500"], START, END),
        producer="test", unit=usgs.UNIT, location_of=lambda key: key,
    )  # fmt: skip
    assert (written.rows, written.series_added) == (1, 1)


@pytest.mark.skipif(not os.environ.get("TEEHR_LIVE_USGS"), reason="set TEEHR_LIVE_USGS=1")
def test_live():
    """The real API (opt in): two gauges, two hours."""
    rows = next(usgs.fetch_streamflow(["01646500", "02339495"], START, END))
    assert set(rows["series_key"].to_pylist()) == {"usgs-01646500", "usgs-02339495"}
    assert rows.num_rows == 4
