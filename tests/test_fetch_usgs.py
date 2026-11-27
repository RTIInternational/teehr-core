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


@pytest.fixture(autouse=True)
def no_metadata(monkeypatch):
    """No series metadata (no network): sites' several series fall back to the latest."""
    asked = []

    def metadata(sites, service):
        asked.append(sites)
        return pd.DataFrame()

    monkeypatch.setattr(usgs, "_series_metadata", metadata)
    return asked


def meta(*rows):
    """Series metadata rows of (site, id, primary, sublocation, description, begin, end)."""
    return pd.DataFrame({
        "time_series_id": [r[1] for r in rows],
        "monitoring_location_id": [f"USGS-{r[0]}" for r in rows],
        "primary": [r[2] for r in rows],
        "sublocation_identifier": [r[3] for r in rows],
        "web_description": [r[4] for r in rows],
        "begin_utc": pd.to_datetime([r[5] for r in rows], utc=True, format="ISO8601"),
        "end_utc": pd.to_datetime([r[6] for r in rows], utc=True, format="ISO8601"),
    })  # fmt: skip


NOW = "2026-11-27T09:00Z"


def choose(monkeypatch, *rows, pins=None):
    monkeypatch.setattr(usgs, "_series_metadata", lambda sites, service: meta(*rows))
    return usgs.choose_series(["x"], pins=pins)


def test_the_primary_series_over_a_secondary_sensor(monkeypatch):
    # The secondary's last value is later, and it has more (5-minute) values: still not kept.
    assert choose(monkeypatch,
        ("08155300", "primary", "Primary", None, "Primary Sensor", "2007-10-01", "2026-11-27T06Z"),
        ("08155300", "secondary", None, None, "Secondary Sensor 5-Min Update", "2026-07-27", NOW),
    ) == {"USGS-08155300": "primary"}  # fmt: skip


def test_a_series_still_reporting_over_an_old_one(monkeypatch):
    # Both primary; the old one has no sublocation and a longer record, but ended in 2008.
    assert choose(monkeypatch,
        ("07358280", "old", "Primary", None, None, "1970-10-06", "2008-10-01"),
        ("07358280", "outflow", "Primary", "OUTFLOW", None, "2007-10-01", NOW),
    ) == {"USGS-07358280": "outflow"}  # fmt: skip


def test_the_whole_site_over_its_meters(monkeypatch):
    assert choose(monkeypatch,
        ("08188590", "right", "Primary", "Right Vel Meter", None, "2016-05-05", NOW),
        ("08188590", "whole", "Primary", None, None, "2016-05-05", NOW),
        ("08188590", "left", "Primary", "Left Vel Meter", None, "2016-05-05", NOW),
    ) == {"USGS-08188590": "whole"}  # fmt: skip


def test_the_total_over_its_parts(monkeypatch):
    assert choose(monkeypatch,
        ("08072500", "c2", "Primary", None, "Conduit 2", "2026-08-06", NOW),
        ("08072500", "total", "Primary", None, "Total Outflow", "2026-08-06", NOW),
        ("08072500", "c1", "Primary", None, "Conduit 1", "2026-08-24", NOW),
    ) == {"USGS-08072500": "total"}  # fmt: skip


def test_otherwise_the_longest_record_with_a_warning(monkeypatch, caplog):
    assert choose(monkeypatch,
        ("04191500", "tailwater", "Primary", None, "Dam tailwater", "1990-10-01", NOW),
        ("04191500", "harding", "Primary", None, "at Harding Rd", "2016-10-01", NOW),
    ) == {"USGS-04191500": "tailwater"}  # fmt: skip
    assert "record length only" in caplog.text and "USGS-04191500" in caplog.text


def test_missing_times_still_choose_one(monkeypatch):
    rows = meta(("405327084555401", "b", "Primary", None, None, None, None),
                ("405327084555401", "a", "Primary", None, None, None, None))  # fmt: skip
    monkeypatch.setattr(usgs, "_series_metadata", lambda sites, service: rows)
    assert usgs.choose_series(["x"]) == {"USGS-405327084555401": "a"}


def test_a_pin_wins(monkeypatch, caplog):
    rows = [("02237700", "structure", "Primary", None, "Structure Discharge", "2005-10-01", NOW),
            ("02237700", "nurf", "Primary", None, "NURF Discharge", "2009-06-01", NOW)]  # fmt: skip
    assert choose(monkeypatch, *rows, pins={"usgs-02237700": "nurf"}) == {"USGS-02237700": "nurf"}
    assert "record length only" not in caplog.text


def test_a_pin_to_a_series_that_stopped_is_warned_about(monkeypatch, caplog):
    rows = [("09522720", "old", "Primary", None, None, "2006-10-01", "2009-10-01"),
            ("09522720", "current", "Primary", None, None, "2007-10-19", NOW)]  # fmt: skip
    assert choose(monkeypatch, *rows, pins={"09522720": "old"}) == {"USGS-09522720": "current"}
    assert "stopped reporting" in caplog.text


def test_a_pin_the_site_does_not_have_is_warned_about(monkeypatch, caplog):
    rows = [("01646500", "only", "Primary", None, None, "1972-06-09", NOW)]
    assert choose(monkeypatch, *rows, pins={"01646500": "nope"}) == {}
    assert "isn't one of its series" in caplog.text


def test_rows_keep_the_chosen_series():
    rows = usgs.to_rows(frame(
        ("08155300", "2026-11-20T00:00Z", 1.0, "primary"),
        ("08155300", "2026-11-20T00:00Z", 9.0, "secondary"),  # later, and more values ...
        ("08155300", "2026-11-20T01:00Z", 9.0, "secondary"),
    ), series={"USGS-08155300": "primary"})  # fmt: skip
    assert [r[2] for r in as_tuples(rows)] == [round(1.0 * usgs.CFS_TO_M3S, 4)]


def test_fetch_chooses_per_batch_unless_given(monkeypatch, no_metadata):
    monkeypatch.setattr(usgs, "_request", lambda service, sites, start, end: frame(
        *[(usgs.site_number(s), "2026-11-20T00:00Z", 1.0) for s in sites]))  # fmt: skip
    list(usgs.fetch_streamflow(["1", "2", "3"], START, END, batch_sites=2))
    assert no_metadata == [["1", "2"], ["3"]]
    no_metadata.clear()
    list(usgs.fetch_streamflow(["1", "2", "3"], START, END, batch_sites=2, series={}))
    assert no_metadata == []


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


def test_approval_status_and_qualifiers():
    assert usgs.qualifier(["ESTIMATED", "ICE"]) == "ESTIMATED,ICE"
    assert usgs.qualifier(None) is None and usgs.qualifier(float("nan")) is None
    api = frame(("01646500", "2026-11-20T00:00Z", 1.0), ("01646500", "2026-11-20T01:00Z", 2.0))
    api["approval_status"] = ["Approved", "Provisional"]
    api["qualifier"] = [["ESTIMATED"], None]
    rows = usgs.to_rows(api)
    assert rows["approval_status"].to_pylist() == ["Approved", "Provisional"]
    assert rows["qualifier"].to_pylist() == ["ESTIMATED", None]


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
    columns, key = timeseries_columns(["approval_status", "qualifier"], "instant")
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


def test_bulk_files_give_the_same_rows(monkeypatch, no_metadata):
    def bulk(service, site):
        if site == "missing":
            return None
        api = frame(("01646500", "2026-11-19T23:00Z", 1.0), ("01646500", "2026-11-20T00:00Z", 2.0),
                    ("01646500", "2026-11-20T02:00Z", 3.0))  # fmt: skip
        api["value"] = api["value"].astype(str)  # the files' values are strings
        api["approval_status"] = "Approved"
        api["qualifier"] = [None, ["ESTIMATED"], None]
        return api

    monkeypatch.setattr(usgs, "_bulk_file", bulk)
    tables = list(usgs.read_bulk(["01646500", "missing"], START, END))
    assert no_metadata == [["01646500", "missing"]]  # asked once, for all the sites
    assert [t.num_rows for t in tables] == [1, 0]  # 2026-11-20 00:00 up to 02:00
    assert tables[0].to_pylist()[0]["approval_status"] == "Approved"
    assert tables[0].to_pylist()[0]["qualifier"] == "ESTIMATED"
    assert tables[0].schema == usgs.SCHEMA


@pytest.mark.skipif(not os.environ.get("TEEHR_LIVE_USGS"), reason="set TEEHR_LIVE_USGS=1")
def test_live_bulk_daily():
    """A real (small) daily file: means only, at midnight UTC."""
    start, end = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 11, tzinfo=UTC)
    rows = next(usgs.read_bulk(["01646500"], start, end, service="dv"))
    assert rows.num_rows == 10
    assert {t.hour for t in rows["value_time"].to_pylist()} == {0}


def test_bulk_file_waits_out_rate_limits_with_the_key(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    import pyarrow as pa
    import pyarrow.parquet as pq

    buf = io.BytesIO()
    pq.write_table(pa.table({
        "time_series_id": ["a"], "monitoring_location_id": ["USGS-01646500"],
        "time": pa.array([pd.Timestamp("2026-11-20T00:00Z")]), "value": ["1.0"],
        "approval_status": ["Approved"], "qualifier": pa.array([None], pa.list_(pa.string())),
        "parameter_code": ["00060"],
    }), buf)  # fmt: skip
    seen, waits = [], []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout):
        seen.append(request.get_header("X-api-key"))
        if len(seen) < 3:
            after = {"Retry-After": "7"} if len(seen) == 1 else {}
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many", after, None)
        return Response(buf.getvalue())

    monkeypatch.setenv("API_USGS_PAT", "k")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(usgs.time, "sleep", waits.append)
    frame = usgs._bulk_file("iv", "01646500")
    assert len(frame) == 1
    assert seen == ["k", "k", "k"]
    assert waits == [7, 120]
