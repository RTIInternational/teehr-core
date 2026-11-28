"""teehr_core.fetch.nwps against responses shaped like the NWPS API's (no network)."""

import io
import json
import os
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta

import pytest

from teehr_core.fetch import nwps


def payload(*points, name="Flow", unit="kcfs"):
    """A stageflow/forecast response with (valid time, kcfs) points."""
    return {"issuedTime": "2026-11-27T17:32:00Z", "secondaryName": name,
            "secondaryUnits": unit,
            "data": [{"validTime": t, "generatedTime": "2026-11-27T17:39:59Z",
                      "primary": 3.0, "secondary": v} for t, v in points]}  # fmt: skip


def at(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def test_a_six_hour_forecast_starts_one_step_before_its_first_value():
    got = nwps.to_forecast("ANAW1", payload(
        ("2026-11-27T18:00:00Z", 13.5), ("2026-11-28T00:00:00Z", 13.6),
        ("2026-11-28T06:00:00Z", 13.9)))  # fmt: skip
    assert got.series_key == "nwpsrfc-anaw1"
    assert got.reference_time == at("2026-11-27T12:00:00Z")
    assert got.step == timedelta(hours=6) and nwps.resolution(got.step) == "PT6H"
    rows = got.rows.to_pylist()
    assert [r["value_time"] for r in rows] == [at("2026-11-27T18:00:00Z"),
                                               at("2026-11-28T00:00:00Z"),
                                               at("2026-11-28T06:00:00Z")]  # fmt: skip
    assert {r["reference_time"] for r in rows} == {at("2026-11-27T12:00:00Z")}
    assert rows[0]["value"] == pytest.approx(13.5 * 28.316846592)


def test_three_hour_steps_are_kept():
    got = nwps.to_forecast("XYZ1", payload(
        ("2026-11-28T03:00:00Z", 1.0), ("2026-11-28T06:00:00Z", 2.0)))  # fmt: skip
    assert (got.step, got.reference_time) == (timedelta(hours=3), at("2026-11-28T00:00:00Z"))


def test_gaps_and_no_data_leave_the_smallest_step():
    got = nwps.to_forecast("MUTC1", payload(
        ("2026-11-27T21:00:00Z", -999), ("2026-11-27T22:00:00Z", 1.0),
        ("2026-11-27T23:00:00Z", 1.1), ("2026-11-28T07:00:00Z", 1.2),
        ("2026-11-28T08:00:00Z", -9999)))  # fmt: skip
    assert got.step == timedelta(hours=1)
    assert got.reference_time == at("2026-11-27T21:00:00Z")
    assert got.rows.num_rows == 3


def test_cfs_is_converted_too():
    got = nwps.to_forecast(
        "A1",
        payload(("2026-11-28T01:00:00Z", 1000.0), ("2026-11-28T02:00:00Z", 1000.0), unit="cfs"),
    )
    assert got.rows["value"][0].as_py() == pytest.approx(28.316846592)


@pytest.mark.parametrize("response", [
    payload(("2026-11-28T01:00:00Z", 1.0), ("2026-11-28T02:00:00Z", 1.0), name="Stage",
            unit="ft"),
    payload(("2026-11-28T01:00:00Z", 1.0), ("2026-11-28T02:00:00Z", 1.0), unit="mgd"),
    payload(("2026-11-28T01:00:00Z", 1.0), ("2026-11-28T02:00:00Z", -999)),
    {"secondaryName": None, "data": []},
])  # fmt: skip
def test_no_flow_forecast(response):
    assert nwps.to_forecast("A1", response) is None


def test_keys_and_lids():
    assert nwps.series_key("anaW1") == "nwpsrfc-anaw1"
    assert nwps.lid_of("nwpsrfc-anaw1") == nwps.lid_of("anaw1") == "ANAW1"
    assert nwps.resolution(timedelta(minutes=15)) == "PT15M"


def test_forecasts_report_each_gauges_outcome(monkeypatch):
    bodies = {"GOOD1": payload(("2026-11-28T01:00:00Z", 1.0), ("2026-11-28T02:00:00Z", 2.0))}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout):
        lid = request.full_url.split("/gauges/")[1].split("/")[0]
        if lid == "GONE1":
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)
        if lid == "DOWN1":
            raise urllib.error.HTTPError(request.full_url, 503, "Unavailable", {}, None)
        return Response(json.dumps(bodies.get(lid, payload(name="Stage"))).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    got = dict(nwps.forecasts(["nwpsrfc-good1", "GONE1", "DOWN1", "STAGE1"], workers=2))
    assert isinstance(got["nwpsrfc-good1"], nwps.Forecast)
    assert got["GONE1"] is None and got["STAGE1"] is None
    assert isinstance(got["DOWN1"], urllib.error.HTTPError)


@pytest.mark.skipif(not os.environ.get("TEEHR_LIVE_NWPS"), reason="set TEEHR_LIVE_NWPS=1")
def test_live_forecasts():
    got = [
        f for _, f in nwps.forecasts(["ANAW1", "BRKI4", "MUTC1"]) if isinstance(f, nwps.Forecast)
    ]
    assert got
    for forecast in got:
        assert forecast.rows["value_time"][0].as_py() - forecast.step == forecast.reference_time
        assert forecast.reference_time < datetime.now(UTC) + timedelta(days=1)
