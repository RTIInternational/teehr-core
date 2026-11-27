"""teehr_core.fetch.nwm against files shaped like NOAA's channel_rt NetCDF (no network)."""

import io
import urllib.error
from datetime import UTC, datetime, timedelta

import pytest

h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")

from teehr_core.fetch import nwm  # noqa: E402

DAY = datetime(2026, 11, 24, tzinfo=UTC)
FILL = -999900


def channel_rt(valid, flows, ids=(101, 202, 303), version="v3.1"):
    """A channel_rt file's bytes: ``flows`` (m3/s, None = missing) at ``ids``, for ``valid``."""
    buf = io.BytesIO()
    with h5py.File(buf, "w") as h5:
        h5.attrs["NWM_version_number"] = np.bytes_(version)
        h5["feature_id"] = np.array(ids, dtype=np.int64)
        raw = [FILL if v is None else round(v * 100) for v in flows]
        flow = h5.create_dataset("streamflow", data=np.array(raw, dtype=np.int32))
        flow.attrs["scale_factor"] = np.array([0.01], dtype=np.float32)
        flow.attrs["add_offset"] = np.array([0.0], dtype=np.float32)
        flow.attrs["_FillValue"] = np.array([FILL], dtype=np.int32)
        minutes = int((valid - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds() // 60)
        h5["time"] = np.array([minutes], dtype=np.int32)
    return buf.getvalue()


@pytest.fixture
def bucket(monkeypatch):
    """A fake GCS: object name -> bytes; listing by prefix."""
    objects: dict[str, bytes] = {}

    def get(url):
        name = url.removeprefix(nwm.FILE_URL.format(name=""))
        if name not in objects:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)  # type: ignore[arg-type]
        return objects[name]

    monkeypatch.setattr(nwm, "_get", get)
    monkeypatch.setattr(
        nwm, "_list", lambda prefix: sorted(n for n in objects if n.startswith(prefix))
    )
    return objects


def test_parse_names():
    sr = nwm.parse("nwm.20261124/short_range/nwm.t05z.short_range.channel_rt.f003.conus.nc")
    assert sr is not None and not sr.analysis and sr.member is None
    assert (sr.reference_time, sr.value_time) == (
        DAY + timedelta(hours=5),
        DAY + timedelta(hours=8),
    )
    mr = nwm.parse(
        "nwm.20261124/medium_range_mem1/nwm.t06z.medium_range.channel_rt_1.f240.conus.nc"
    )
    assert mr is not None and mr.member == "1"
    assert mr.value_time == DAY + timedelta(hours=6 + 240)
    hi = nwm.parse(
        "nwm.20261124/short_range_hawaii/nwm.t12z.short_range.channel_rt.f00015.hawaii.nc"
    )
    assert hi is not None and hi.value_time == DAY + timedelta(hours=12, minutes=15)
    aa = nwm.parse(
        "nwm.20261124/analysis_assim_no_da/nwm.t03z.analysis_assim_no_da.channel_rt.tm02.conus.nc"
    )
    assert aa is not None and aa.analysis and aa.value_time == DAY + timedelta(hours=1)
    hia = nwm.parse("nwm.20261124/analysis_assim_hawaii_no_da/"
                    "nwm.t03z.analysis_assim_no_da.channel_rt.tm0245.hawaii.nc")  # fmt: skip
    assert hia is not None and hia.value_time == DAY + timedelta(minutes=15)
    assert (
        nwm.parse("nwm.20261124/short_range/nwm.t05z.short_range.reservoir.f003.conus.nc") is None
    )


def test_forecast_rows_keep_their_cycle(bucket):
    for cycle in (0, 1):
        for lead in (1, 2):
            t = DAY + timedelta(hours=cycle + lead)
            name = (f"nwm.20261124/short_range/"
                    f"nwm.t{cycle:02d}z.short_range.channel_rt.f{lead:03d}.conus.nc")  # fmt: skip
            bucket[name] = channel_rt(t, [1.5 + cycle, None, 3.25 * lead])
    bucket["nwm.20261124/short_range/nwm.t00z.short_range.reservoir.f001.conus.nc"] = b"x"
    found = nwm.files("short_range", DAY, DAY + timedelta(hours=1))
    assert [f.name.split(".")[-3] for f in found] == ["f001", "f002"]  # the 00z cycle only
    rows = [
        r for t in nwm.read_points(found, ["nwm31-101", 202, "303", "404"]) for r in t.to_pylist()
    ]
    assert rows == [
        {"series_key": "nwm31-101", "reference_time": DAY, "value_time": DAY + timedelta(hours=1),
         "value": 1.5},
        {"series_key": "nwm31-303", "reference_time": DAY, "value_time": DAY + timedelta(hours=1),
         "value": 3.25},
        {"series_key": "nwm31-101", "reference_time": DAY, "value_time": DAY + timedelta(hours=2),
         "value": 1.5},
        {"series_key": "nwm31-303", "reference_time": DAY, "value_time": DAY + timedelta(hours=2),
         "value": 6.5},
    ]  # fmt: skip


def test_members_and_values_are_exact(bucket):
    name = "nwm.20261124/medium_range_mem1/nwm.t00z.medium_range.channel_rt_1.f003.conus.nc"
    bucket[name] = channel_rt(DAY + timedelta(hours=3), [123.45, 0.07, 98765.43])
    (table,) = nwm.read_points(nwm.files("medium_range_mem1", DAY, DAY + timedelta(days=1)),
                               [101, 202, 303])  # fmt: skip
    assert table.column_names == ["series_key", "reference_time", "value_time", "member", "value"]
    assert table["member"].to_pylist() == ["1"] * 3
    assert table["value"].to_pylist() == [123.45, 0.07, 98765.43]


def test_analysis_keeps_the_latest_cycle_per_time(bucket):
    """tm00..tm02 of hourly cycles: a time's value is from the latest cycle that has it."""
    for cycle in range(0, 4):
        for tm in range(3):
            valid = DAY + timedelta(hours=cycle - tm)
            name = (f"nwm.20261124/analysis_assim_no_da/nwm.t{cycle:02d}z."
                    f"analysis_assim_no_da.channel_rt.tm{tm:02d}.conus.nc")  # fmt: skip
            bucket[name] = channel_rt(valid, [cycle * 10 + tm, 0, 0])
    found = nwm.files("analysis_assim_no_da", DAY, DAY + timedelta(days=1))
    # Cycles 00..03z cover times from 22z the day before to 03z; each time once, latest cycle.
    assert [(f.reference_time.hour, f.name.split(".")[-3]) for f in found] == [
        (0, "tm02"), (1, "tm02"), (2, "tm02"), (3, "tm02"), (3, "tm01"), (3, "tm00"),
    ]  # fmt: skip
    rows = [r for t in nwm.read_points(found, [101]) for r in t.to_pylist()]
    assert [(r["value_time"] - DAY, r["value"]) for r in rows] == [
        (timedelta(hours=h), v) for h, v in [(-2, 2), (-1, 12), (0, 22), (1, 32), (2, 31), (3, 30)]
    ]
    assert all("reference_time" not in r for r in rows)


def test_missing_files_are_skipped_and_versions_checked(bucket):
    name = "nwm.20261124/short_range/nwm.t00z.short_range.channel_rt.f001.conus.nc"
    bucket[name] = channel_rt(DAY + timedelta(hours=1), [1.0, 2.0, 3.0])
    found = nwm.files("short_range", DAY, DAY + timedelta(hours=1))
    del bucket[name]
    assert list(nwm.read_points(found, [101])) == []
    bucket[name] = channel_rt(DAY + timedelta(hours=1), [1.0, 2.0, 3.0], version="v3.0")
    with pytest.raises(ValueError, match=r"v3\.0, not nwm31"):
        list(nwm.read_points(found, [101]))
    # Within a day of v3.1 starting, a file still saying v3.0 is accepted.
    start = nwm.STARTS["nwm31"]
    early = f"nwm.{start:%Y%m%d}/short_range/nwm.t00z.short_range.channel_rt.f001.conus.nc"
    bucket[early] = channel_rt(start + timedelta(hours=1), [1.0, 2.0, 3.0], version="v3.0")
    (table,) = nwm.read_points(nwm.files("short_range", start, start + timedelta(hours=1)), [101])
    assert table["series_key"].to_pylist() == ["nwm31-101"]


def test_times_must_be_utc():
    with pytest.raises(ValueError, match="time zone"):
        nwm.files("short_range", datetime(2026, 11, 24), datetime(2026, 11, 25))
