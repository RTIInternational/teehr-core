"""Generated predicted inputs: persistence forecasts built from the observations."""

from __future__ import annotations

import bisect
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import pyarrow.parquet as pq
import pytest
import yaml

from teehr_core import engine
from teehr_core.engine.plan import PlanError
from teehr_core.model.units import UnitRegistry
from tests.engine_support import HYDRO, SPECS

CATALOG = "tests/fixtures/hydro/datasets.yaml"


def _spec(**generated: Any) -> dict[str, Any]:
    spec: dict[str, Any] = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())
    spec["inputs"]["secondary"] = [
        {"dataset": "nwm31_short_range"},
        {"generate": "persistence", "like": "nwm31_short_range", **generated},
    ]
    spec["output"] = {"include_pairs": True}
    return spec


def _expected(max_age: timedelta | None = None) -> dict[tuple[str, Any, Any], float]:
    """Persistence computed directly: the last observation at or before each reference time."""
    obs = pq.read_table(HYDRO / "usgs_streamflow.parquet").to_pylist()
    xwalk = pq.read_table(HYDRO / "usgs_nwm31_crosswalk.parquet").to_pylist()
    fcst = pq.read_table(HYDRO / "nwm31_short_range.parquet").to_pylist()
    by_gage: dict[str, list[tuple[Any, float]]] = defaultdict(list)
    for r in obs:
        by_gage[r["series_key"]].append((r["value_time"], r["value"]))
    for rows in by_gage.values():
        rows.sort()
    gage_of = {r["right_key"]: r["left_key"] for r in xwalk}
    out = {}
    for f in fcst:
        gage = gage_of.get(f["series_key"])
        if gage is None:
            continue
        rows = by_gage[gage]
        k = bisect.bisect_right([t for t, _ in rows], f["reference_time"]) - 1
        if k < 0 or (max_age is not None and f["reference_time"] - rows[k][0] > max_age):
            continue
        out[(gage, f["reference_time"], f["value_time"])] = rows[k][1]
    return out


@pytest.mark.parametrize("max_age", [None, "PT20M"])
def test_persistence_values(units: UnitRegistry, max_age: str | None) -> None:
    spec = _spec(**({"max_age": max_age} if max_age else {}))
    res = engine.evaluate(spec, CATALOG, units)
    assert res.pairs is not None
    pairs = [
        r
        for r in res.pairs.to_pylist()
        if r["secondary_dataset"] == "persistence:nwm31_short_range"
    ]
    want = _expected(timedelta(minutes=20) if max_age else None)
    obs_at = {
        (r["series_key"], r["value_time"]): r["value"]
        for r in pq.read_table(HYDRO / "usgs_streamflow.parquet").to_pylist()
    }
    # Every forecast with an observation at its value time and a persistence value pairs.
    want = {k: v for k, v in want.items() if (k[0], k[2]) in obs_at}
    got = {
        (r["primary_series_key"], r["reference_time"], r["value_time"]): r["secondary_value"]
        for r in pairs
    }
    assert got == pytest.approx(want) and len(got) > 0
    assert all(r["secondary_series_key"] == r["primary_series_key"] for r in pairs)
    names = set(res.table.column("secondary_dataset").to_pylist())
    assert names == {"nwm31_short_range", "persistence:nwm31_short_range"}


def test_skill_against_persistence(units: UnitRegistry) -> None:
    spec = _spec(name="persistence")
    spec["aggregate"] = [
        {
            "group_by": ["primary_series_key", "secondary_dataset"],
            "metrics": ["root_mean_square_error", "mean_absolute_error"],
            "skill": {"reference": "persistence", "metrics": ["root_mean_square_error"]},
        }
    ]
    rows = engine.evaluate(spec, CATALOG, units).table.to_pylist()
    ref = [r for r in rows if r["secondary_dataset"] == "persistence"]
    assert ref and all(r["root_mean_square_error_skill"] == pytest.approx(0) for r in ref)
    nwm = [r for r in rows if r["secondary_dataset"] == "nwm31_short_range"]
    assert all(
        r["root_mean_square_error_skill"]
        == pytest.approx(1 - r["root_mean_square_error"] / p["root_mean_square_error"])
        for r, p in zip(
            sorted(nwm, key=lambda r: r["primary_series_key"]),
            sorted(ref, key=lambda r: r["primary_series_key"]),
            strict=True,
        )
    )


def test_generated_input_checks(units: UnitRegistry) -> None:
    spec = _spec()
    spec["inputs"]["secondary"][1]["like"] = "usgs_streamflow"
    with pytest.raises(PlanError, match="role"):
        engine.evaluate(spec, CATALOG, units)
    spec = _spec(name="nwm31_short_range")
    with pytest.raises(PlanError, match="distinct names"):
        engine.evaluate(spec, CATALOG, units)


# -- climatology --------------------------------------------------------------------------


def _expected_climatology(window_days: int) -> dict[tuple[str, int, int], float]:
    """Mean per series, day of year and hour over a circular window, computed directly."""
    obs = pq.read_table(HYDRO / "usgs_streamflow.parquet").to_pylist()
    half = window_days // 2
    hours: dict[tuple[str, datetime], list[float]] = defaultdict(list)
    for r in obs:  # hourly means first: the fixture is 5-minute data
        hours[(r["series_key"], r["value_time"].replace(minute=0))].append(r["value"])
    sums: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for (key, t), values in hours.items():
        sums[(key, t.timetuple().tm_yday, t.hour)].append(sum(values) / len(values))
    out = {}
    for series, doy, hour in {(s, d, h) for s, d, h in sums}:
        for day in range(doy - half, doy + half + 1):
            target = (day - 1) % 366 + 1
            vals = [
                v
                for d in range(target - half, target + half + 1)
                for v in sums.get((series, (d - 1) % 366 + 1, hour), [])
            ]
            if vals:
                out[(series, target, hour)] = sum(vals) / len(vals)
    return out


def test_generated_spec_checks() -> None:
    from pydantic import ValidationError

    from teehr_core.spec import load_spec

    base = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())
    for bad in (
        {"generate": "persistence", "from": "x"},
        {"generate": "climatology", "like": "x"},
        {"generate": "climatology", "from": "x", "max_age": "PT1H"},
    ):
        base["inputs"]["secondary"] = [{"dataset": "nwm31_short_range"}, bad]
        with pytest.raises(ValidationError):
            load_spec(base)


def test_persistence_for_one_long_series_is_fast(units: UnitRegistry) -> None:
    """One gage, a year of hourly short-range forecasts: DuckDB ran this ASOF join as a
    nested loop (it misjudges the left side as tiny), 70 s instead of 0.07 s."""
    import time
    from datetime import datetime

    import numpy as np
    import pyarrow as pa

    from teehr_core.engine.catalog import Catalog, Dataset

    start = np.datetime64(datetime(2024, 1, 1), "us")
    hours = 24 * 365
    obs_t = start + np.arange(hours * 4) * np.timedelta64(15, "m")
    ref = np.repeat(start + np.arange(hours) * np.timedelta64(1, "h"), 18)
    valid = ref + np.tile(np.arange(1, 19), hours) * np.timedelta64(1, "h")
    ts = pa.timestamp("us", tz="UTC")
    obs = pa.table(
        {
            "series_key": ["g"] * obs_t.size,
            "value_time": pa.array(obs_t, ts),
            "value": np.ones(obs_t.size),
        }
    )
    fcst = pa.table(
        {
            "series_key": ["r"] * ref.size,
            "reference_time": pa.array(ref, ts),
            "value_time": pa.array(valid, ts),
            "value": np.ones(ref.size),
        }
    )
    unit = "[cft_i]/s"
    catalog = Catalog(
        [
            Dataset("obs", "observed", obs, variable="streamflow", unit=unit, statistic="inst"),
            Dataset("fc", "forecast", fcst, variable="streamflow", unit=unit, statistic="inst"),
            Dataset(
                "xw",
                "auxiliary",
                pa.table({"left_key": ["g"], "right_key": ["r"]}),
                kind="crosswalk",
            ),
        ],
        units,
    )
    spec = {
        "version": 1,
        "name": "long",
        "inputs": {
            "primary": {"dataset": "obs"},
            "secondary": [
                {"dataset": "fc"},
                {"generate": "persistence", "like": "fc", "name": "persistence", "max_age": "PT2H"},
            ],
        },
        "pair": {"relation": "xw"},
        "aggregate": [
            {"group_by": ["primary_series_key", "secondary_dataset"], "metrics": ["count"]}
        ],
    }
    t0 = time.perf_counter()
    res = engine.evaluate(spec, catalog, units)
    assert time.perf_counter() - t0 < 15  # ~1 s; the nested loop took over a minute
    counts = dict(
        zip(res.table["secondary_dataset"].to_pylist(), res.table["count"].to_pylist(), strict=True)
    )
    paired = int((valid <= obs_t[-1]).sum())  # forecasts past the last observation don't pair
    assert counts == {"fc": paired, "persistence": paired}
