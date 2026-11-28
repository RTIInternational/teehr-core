"""Every source returns the same thing: v1 timeseries rows a dataset's writer takes as is.

Add a case for a new source: rows it returned, and the optional columns its dataset has.
"""

from datetime import UTC, datetime, timedelta

import pytest

pytest.importorskip("pandas")
pytest.importorskip("h5py")

from teehr_core.fetch import nwm, nwps, usgs
from teehr_core.model.dataset_types import arrow_schema, timeseries_columns
from tests.test_fetch_nwm import DAY, channel_rt
from tests.test_fetch_nwm import bucket as bucket
from tests.test_fetch_usgs import frame


def usgs_rows(_store):
    rows = usgs.to_rows(frame(("01646500", "2026-11-20T00:00Z", 100.0)))
    return usgs, rows, ["approval_status", "qualifier"]


def nwm_forecast_rows(store):
    name = "nwm.20261124/medium_range_mem1/nwm.t00z.medium_range.channel_rt_1.f001.conus.nc"
    store[name] = channel_rt(DAY + timedelta(hours=1), [1.0, 2.0, 3.0])
    (rows,) = nwm.read_points(nwm.files("medium_range_mem1", DAY, DAY + timedelta(days=1)), [101])
    return nwm, rows, ["reference_time", "member"]


def nwm_analysis_rows(store):
    name = (
        "nwm.20261124/analysis_assim_no_da/nwm.t01z.analysis_assim_no_da.channel_rt.tm00.conus.nc"
    )
    store[name] = channel_rt(DAY + timedelta(hours=1), [1.0, 2.0, 3.0])
    (rows,) = nwm.read_points(
        nwm.files("analysis_assim_no_da", DAY, DAY + timedelta(days=1)), [101]
    )
    return nwm, rows, []


def nwps_rows(_store):
    got = nwps.to_forecast("ANAW1", {"secondaryName": "Flow", "secondaryUnits": "kcfs", "data": [
        {"validTime": "2026-11-27T18:00:00Z", "secondary": 13.5},
        {"validTime": "2026-11-28T00:00:00Z", "secondary": 13.6}]})  # fmt: skip
    return nwps, got.rows, ["reference_time"]


@pytest.mark.parametrize("case", [usgs_rows, nwm_forecast_rows, nwm_analysis_rows, nwps_rows])
def test_rows_are_v1_timeseries_rows(case, bucket, units):
    source, rows, optional = case(bucket)
    assert rows.num_rows
    columns, _ = timeseries_columns(optional, "instant")
    expected = arrow_schema(columns)
    # The dataset's columns, in its order, with its types (nullability is the writer's check).
    assert rows.column_names == expected.names
    assert [f.type for f in rows.schema] == [f.type for f in expected]
    assert all(t >= datetime(1970, 1, 1, tzinfo=UTC) for t in rows["value_time"].to_pylist())
    assert source.UNIT in units
