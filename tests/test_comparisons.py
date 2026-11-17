"""Comparing forecasts: common samples, ramps, the Diebold-Mariano test and the
persistence-climatology blend, on synthetic hourly load (interval means)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from teehr_core import engine
from teehr_core.engine import Catalog, Dataset
from teehr_core.metrics.significance import default_lag, diebold_mariano
from teehr_core.model.contract import TIMESTAMP
from teehr_core.model.periods import Period, PeriodRegistry

START = datetime(2026, 1, 1, tzinfo=UTC)
DAYS = 40
PHI = 0.9  # hourly autocorrelation of the anomaly
PERIODS = PeriodRegistry([Period("hour_starting", "duration", "PT1H", "start")])


def _base(t: datetime) -> float:
    return 1000 + 300 * math.sin(2 * math.pi * (t.hour - 6) / 24)


def _series(keys: list[str], times: list[datetime], values: list[float], **extra: Any) -> pa.Table:
    cols: dict[str, Any] = {"series_key": keys}
    cols.update({k: pa.array(v, TIMESTAMP) for k, v in extra.items()})
    cols["value_time"] = pa.array(times, TIMESTAMP)
    cols["value"] = pa.array(values, pa.float64())
    return pa.table(cols)


def _catalog(gappy_every: int = 5) -> tuple[Catalog, dict[datetime, float]]:
    rng = np.random.default_rng(7)
    times = [START + timedelta(hours=h) for h in range(DAYS * 24)]
    anomaly, obs = 0.0, {}
    for t in times:
        anomaly = PHI * anomaly + rng.normal(0, 30)
        obs[t] = _base(t) + anomaly
    series = pa.table({"series_key": ["zone-a"], "zone": ["a"]})
    meta = {"variable": "load", "unit": "MW", "statistic": "mean", "period": "hour_starting"}
    observed = Dataset(
        "load", "observed", _series(["zone-a"] * len(times), times, list(obs.values())),
        series=series, **meta,
    )  # fmt: skip
    clim = pa.table(
        {
            "series_key": ["zone-a"] * 366 * 24,
            "day_of_year": [d for d in range(1, 367) for _ in range(24)],
            "hour": [h for _ in range(366) for h in range(24)],
            "value": [_base(START.replace(hour=h)) for _ in range(366) for h in range(24)],
        }
    )
    climatology = Dataset(
        "load_clim", "auxiliary", clim, variable="load", unit="MW", kind="climatology"
    )

    def forecast(name: str, factor: float, skip: int | None) -> Dataset:
        keys, refs, vts, vals = [], [], [], []
        for d in range(1, DAYS - 1):
            if skip and d % skip == 0:
                continue
            ref = START + timedelta(days=d - 1, hours=6)  # issued 06:00 the day before
            for h in range(24):
                vt = START + timedelta(days=d, hours=h)
                keys.append("fc-a")
                refs.append(ref)
                vts.append(vt)
                vals.append(obs[vt] * factor)
        fseries = pa.table({"series_key": ["fc-a"], "zone": ["a"]})
        table = _series(keys, vts, vals, reference_time=refs)
        return Dataset(name, "forecast", table, series=fseries, **meta)

    datasets = [
        observed,
        climatology,
        forecast("fc", 1.02, None),
        forecast("fc_worse", 1.08, None),
        forecast("fc_gappy", 1.02, gappy_every),
    ]
    return Catalog(datasets, periods=PERIODS), obs


def _spec(predicted: list[Any], aggregate: list[Any], **more: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "name": "cmp",
        "inputs": {"primary": {"dataset": "load"}, "secondary": predicted},
        "pair": {"on_attribute": "zone", **more.pop("pair", {})},
        "aggregate": aggregate,
        **more,
    }


def _rows(result: Any) -> dict[str, dict[str, Any]]:
    return {r["secondary_dataset"]: r for r in result.table.to_pylist()}


def test_common_sample() -> None:
    catalog, _ = _catalog()
    predicted = [{"dataset": "fc"}, {"dataset": "fc_gappy"}]
    agg = [{"group_by": ["secondary_dataset"], "metrics": ["count", "root_mean_square_error"]}]
    each = _rows(engine.evaluate(_spec(predicted, agg), catalog))
    assert each["fc"]["count"] > each["fc_gappy"]["count"]
    common = _rows(engine.evaluate(_spec(predicted, agg, pair={"common_sample": True}), catalog))
    assert common["fc"]["count"] == common["fc_gappy"]["count"] == each["fc_gappy"]["count"]
    assert common["fc"]["root_mean_square_error"] == pytest.approx(
        common["fc_gappy"]["root_mean_square_error"]
    )


def test_change_field_and_ramps() -> None:
    catalog, obs = _catalog()
    spec = _spec(
        [{"dataset": "fc"}],
        [
            {
                "group_by": ["secondary_dataset"],
                "metrics": [
                    {
                        "name": "hits",
                        "threshold": {"value": 150, "unit": "MW"},
                        "values": {"primary": "dq_obs", "secondary": "dq_pred"},
                    },
                    {
                        "name": "misses",
                        "threshold": {"value": 150, "unit": "MW"},
                        "values": {"primary": "dq_obs", "secondary": "dq_pred"},
                    },
                    {
                        "name": "root_mean_square_error",
                        "values": {"primary": "dq_obs", "secondary": "dq_pred"},
                    },
                ],
            }
        ],
        fields=[
            {"name": "dq_obs", "fn": "change", "side": "primary", "window": "PT3H"},
            {"name": "dq_pred", "fn": "change", "side": "secondary", "window": "PT3H"},
        ],
        output={"include_pairs": True},
    )
    res = engine.evaluate(spec, catalog)
    pairs = res.pairs.to_pylist()  # type: ignore[union-attr]
    first = next(p for p in pairs if p["dq_obs"] is not None)
    vt = first["value_time"]
    assert first["dq_obs"] == pytest.approx(obs[vt] - obs[vt - timedelta(hours=3)])
    assert first["dq_pred"] == pytest.approx(1.02 * first["dq_obs"])
    # The first 3 hours of each forecast have no value 3 h earlier in that forecast.
    assert all(p["dq_pred"] is None for p in pairs if p["value_time"].hour < 3)
    row = res.table.to_pylist()[0]
    up = [p for p in pairs if p["dq_obs"] is not None and p["dq_pred"] is not None]
    assert row["hits_dq_obs"] == sum(p["dq_obs"] >= 150 and p["dq_pred"] >= 150 for p in up)
    assert row["misses_dq_obs"] == sum(p["dq_obs"] >= 150 and p["dq_pred"] < 150 for p in up)
    assert row["root_mean_square_error_dq_obs"] == pytest.approx(
        math.sqrt(sum((p["dq_pred"] - p["dq_obs"]) ** 2 for p in up) / len(up))
    )


def test_diebold_mariano_function() -> None:
    assert diebold_mariano([[0.0, 0.0, 0.0]]) is None  # identical forecasts
    assert diebold_mariano([[1.0]]) is None
    rng = np.random.default_rng(1)
    d = rng.normal(0, 1, 2000)
    # iid with no lags: the t-statistic of the mean (population variance).
    stat, p = diebold_mariano([d], lag=0)  # type: ignore[misc]
    assert stat == pytest.approx(d.mean() / (d.std() / math.sqrt(len(d))))
    assert 0 <= p <= 1
    stat, p = diebold_mariano([d - 0.2])  # type: ignore[misc]
    assert stat < -3 and p < 0.01
    # Autocorrelated differentials: the Newey-West variance is larger, the statistic smaller.
    ar = np.zeros(2000)
    for k in range(1, 2000):
        ar[k] = 0.8 * ar[k - 1] + rng.normal(0, 1)
    plain, _ = diebold_mariano([ar + 0.1], lag=0)  # type: ignore[misc]
    corrected, _ = diebold_mariano([ar + 0.1], lag=20)  # type: ignore[misc]
    assert abs(corrected) < abs(plain)
    assert default_lag(100) == 4 and default_lag(0) == 0


def test_diebold_mariano_in_a_spec() -> None:
    catalog, _ = _catalog()
    spec = _spec(
        [{"dataset": "fc"}, {"dataset": "fc_worse"}],
        [
            {
                "group_by": ["secondary_dataset"],
                "metrics": ["root_mean_square_error"],
                "skill": {
                    "reference": "fc_worse",
                    "metrics": ["root_mean_square_error"],
                    "test": "dm",
                },
            }
        ],
    )
    rows = _rows(engine.evaluate(spec, catalog))
    assert rows["fc"]["dm_statistic"] < -3 and rows["fc"]["dm_p_value"] < 0.01
    assert rows["fc_worse"]["dm_statistic"] is None
    absolute = spec | {
        "aggregate": [
            {**spec["aggregate"][0], "skill": {**spec["aggregate"][0]["skill"], "loss": "absolute"}}
        ]
    }
    assert _rows(engine.evaluate(absolute, catalog))["fc"]["dm_statistic"] < -3


def test_persistence_from_interval_means() -> None:
    """An hourly mean labelled 05:00 is complete at 06:00: usable for a 06:00 forecast."""
    catalog, obs = _catalog()
    spec = _spec(
        [{"dataset": "fc"}, {"generate": "persistence", "like": "fc", "name": "persistence"}],
        [{"group_by": ["secondary_dataset"], "metrics": ["count"]}],
        output={"include_pairs": True},
    )
    pairs = engine.evaluate(spec, catalog).pairs.to_pylist()  # type: ignore[union-attr]
    pers = [p for p in pairs if p["secondary_dataset"] == "persistence"]
    assert pers
    for p in pers[:30]:
        assert p["secondary_value"] == pytest.approx(obs[p["reference_time"] - timedelta(hours=1)])


def test_blend() -> None:
    catalog, obs = _catalog()
    predicted = [
        {"dataset": "fc"},
        {"generate": "persistence", "like": "fc", "name": "persistence"},
        {"generate": "climatology", "from": "load_clim", "name": "climatology"},
        {"generate": "blend", "like": "fc", "from": "load_clim", "name": "blend"},
    ]
    spec = _spec(
        predicted,
        [{"group_by": ["secondary_dataset"], "metrics": ["count", "root_mean_square_error"]}],
        output={"include_pairs": True},
    )
    res = engine.evaluate(spec, catalog)
    pairs = [p for p in res.pairs.to_pylist() if p["secondary_dataset"] == "blend"]  # type: ignore[union-attr]
    assert pairs
    # Expected: climatology plus the persisted anomaly damped by its correlation at the lag.
    anomaly = {t: v - _base(t) for t, v in obs.items()}
    times = sorted(anomaly)
    a = np.array([anomaly[t] for t in times])

    def gamma(lag_h: int) -> float:
        x, y = a[lag_h:], a[:-lag_h]
        return max(0.0, min(1.0, float(x @ y / math.sqrt((x @ x) * (y @ y)))))

    for p in pairs[:50]:
        tp = p["reference_time"] - timedelta(hours=1)
        lag = int((p["value_time"] - tp).total_seconds() // 3600)
        want = _base(p["value_time"]) + gamma(lag) * anomaly[tp]
        assert p["secondary_value"] == pytest.approx(want, rel=1e-9)
    rmse = {r["secondary_dataset"]: r["root_mean_square_error"] for r in res.table.to_pylist()}
    assert rmse["blend"] < rmse["persistence"] and rmse["blend"] <= rmse["climatology"] * 1.01


@pytest.mark.parametrize(
    "stage",
    [
        {
            "group_by": ["secondary_dataset"],
            "metrics": ["count", "root_mean_square_error", "kling_gupta_efficiency"],
            "skill": {"reference": "fc_worse", "metrics": ["root_mean_square_error"], "test": "dm"},
            "guards": [{"metrics": ["kling_gupta_efficiency"], "min_count": 500}],
        },
        {
            "group_by": ["secondary_dataset", "primary_series_key"],
            "metrics": ["root_mean_square_error"],
            "bootstrap": {"block": "PT6H", "reps": 20},
        },
        {"group_by": [], "metrics": ["count", "mean_error"]},
    ],
    ids=["skill-dm-guards", "bootstrap", "one-group"],
)
def test_aggregating_pairs_files_matches_in_memory(stage: dict[str, Any], tmp_path: Any) -> None:
    """A shuffle bucket's pairs, read in place from Parquet, aggregate as in memory."""
    import pyarrow.parquet as pq

    from teehr_core.engine.duckdb_backend import DuckDBBackend, PairsFiles
    from teehr_core.spec import load_spec

    catalog, _ = _catalog()
    predicted = [{"dataset": "fc"}, {"dataset": "fc_worse"}]
    spec = _spec(predicted, [stage], output={"include_pairs": True})
    pairs = engine.evaluate(spec, catalog).pairs
    assert pairs is not None
    pred = pairs["secondary_value"].to_pylist()
    pred[::97] = [None] * len(pred[::97])  # missing values count as missing either way
    pairs = pairs.set_column(
        pairs.schema.get_field_index("secondary_value"), "secondary_value", pa.array(pred)
    )
    for i, part in enumerate((pairs.slice(0, 1000), pairs.slice(1000))):  # two shards
        (tmp_path / f"shard-{i}" / "__bucket=3").mkdir(parents=True)
        pq.write_table(part, tmp_path / f"shard-{i}" / "__bucket=3" / "data.parquet")
    files = PairsFiles.open(f"{tmp_path}/*/__bucket=3/*.parquet", str(tmp_path))
    assert sorted(files.column_names) == sorted(pairs.column_names)

    parsed = load_spec(spec)
    want = DuckDBBackend().aggregate(parsed, pairs)[0].to_pylist()
    got = DuckDBBackend().aggregate(parsed, files, 1, "256MB")[0].to_pylist()
    assert len(got) == len(want) and got
    for a, b in zip(got, want, strict=True):
        assert a.keys() == b.keys()
        for k in a:
            if isinstance(a[k], float):
                assert a[k] == pytest.approx(b[k], rel=1e-9), k
            else:
                assert a[k] == b[k], k


def test_dm_lag_sums_merge_and_follow_time_steps() -> None:
    """Sums by lag split into parts add up to the whole's, and lags are time steps."""
    from teehr_core.metrics.significance import dm_from_sums, dm_lag_sums

    rng = np.random.default_rng(3)
    sizes = [300, 200, 50]
    d = np.concatenate([rng.normal(0.2, 1, n) for n in sizes])
    series = np.repeat(np.array(["a", "b", "c"]), sizes)
    hours = np.concatenate([np.arange(n) for n in sizes]) * 3600
    gid = np.zeros(d.size, np.int64)
    whole = dm_lag_sums(d, gid, series, hours, 3600, 12, 1)
    positional = diebold_mariano([d[series == s] for s in "abc"], lag=10)
    assert dm_from_sums(*(x[0] for x in whole), 10) == pytest.approx(positional, rel=1e-12)

    first = hours < 100 * 3600  # two parts, with lag pairs across the boundary
    parts = [dm_lag_sums(d, gid, series, hours, 3600, 12, 1, counted=c) for c in (first, ~first)]
    for w, a, b in zip(whole, *parts, strict=True):
        assert w == pytest.approx(a + b)

    gappy = rng.random(d.size) > 0.2  # missing hours: a lag partner must be k hours back
    sums = dm_lag_sums(d[gappy], gid[gappy], series[gappy], hours[gappy], 3600, 12, 1)
    dg, sg, tg = d[gappy], series[gappy], hours[gappy]
    for k in (0, 1, 5, 12):
        pairs = [(i, j) for i in range(dg.size) for j in range(dg.size)
                 if sg[i] == sg[j] and tg[j] == tg[i] - k * 3600]  # fmt: skip
        assert sums[0][0, k] == len(pairs)
        assert sums[3][0, k] == pytest.approx(sum(dg[i] * dg[j] for i, j in pairs))
    with pytest.raises(ValueError, match="needs 13 lags"):
        dm_from_sums(*(x[0] for x in sums), 13)
