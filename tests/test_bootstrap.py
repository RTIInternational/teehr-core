"""Shared components, weighted evaluation and bootstrap intervals."""

import numpy as np
import pytest

from teehr_core.metrics import METRICS, MetricRequest, Params, compute_metrics
from teehr_core.metrics.bootstrap import (
    BootstrapSettings,
    _Streams,
    bootstrap_intervals,
    group_seed,
    resample_counts,
)
from teehr_core.metrics.stats import GroupStats

RNG = np.random.default_rng(7)


def series(n: int, groups: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gid = np.repeat(np.arange(groups), n)
    obs = RNG.gamma(2.0, 50.0, size=gid.size) + 1
    pred = obs * RNG.normal(1.05, 0.2, size=gid.size)
    obs = np.round(obs)  # some ties
    return obs, pred, gid


def test_shared_components_are_computed_once() -> None:
    obs, pred, gid = series(50)
    stats = GroupStats(obs, pred, gid, 3)
    for name in (
        "root_mean_square_error",
        "nash_sutcliffe_efficiency",
        "kling_gupta_efficiency",
        "r_squared",
        "root_mean_standard_deviation_ratio",
        "kling_gupta_efficiency_mod1",
        "kling_gupta_efficiency_mod2",
    ):
        METRICS[name].finalize(stats, Params())
    assert len(stats.computed) == len(set(stats.computed))
    for shared in ("n", "ss_obs", "sum_sq_error", "pearson_r"):
        assert stats.computed.count(shared) == 1


def _params(name: str) -> Params:
    if "threshold" in METRICS[name].required:
        return Params(threshold_quantile=0.6)
    if name == "annual_peak_flow_bias":
        return Params(year_start_month=10)
    if name in ("normalized_mean_absolute_error", "normalized_root_mean_square_error"):
        return Params(normalize_by="range_primary")
    if name == "mean_absolute_scaled_error":
        return Params(scale_column="scale")
    return Params(side="pred") if "side" in METRICS[name].params else Params()


#: Interval metrics need a lower and an upper quantile per case (see test_probabilistic).
ROW_RESAMPLABLE = [n for n in METRICS if not n.startswith(("interval_", "winkler"))]


@pytest.mark.parametrize("name", ROW_RESAMPLABLE)
def test_weights_equal_explicit_resampling(name: str) -> None:
    """A weighted evaluation equals evaluating the resampled data, for every metric."""
    obs, pred, gid = series(40)
    # Daily times spanning two water years, so annual metrics have several years.
    times = (np.tile(np.arange(40), 3) * 86400.0 * 9) + 1.72e9
    counts = np.vstack([RNG.multinomial(40, np.full(40, 1 / 40)) for _ in range(3)])
    weights = np.zeros((2, obs.size))
    weights[0] = counts.ravel()  # replicate 0: resampled
    weights[1] = 1.0  # replicate 1: original data
    req = MetricRequest(name, params=_params(name))
    levels = np.tile(np.linspace(0.05, 0.95, 40), 3)
    cols = {"value_time": times, "quantile": levels, "scale": np.tile(np.linspace(0.5, 5, 40), 3)}
    got = compute_metrics([req], obs, pred, gid, 3, weights, cols)[req]
    idx = np.repeat(np.arange(obs.size), weights[0].astype(int))
    want = compute_metrics(
        [req], obs[idx], pred[idx], gid[idx], 3, columns={k: v[idx] for k, v in cols.items()}
    )[req][0]
    plain = compute_metrics([req], obs, pred, gid, 3, columns=cols)[req][0]
    np.testing.assert_allclose(got[0], want, rtol=1e-9, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(got[1], plain, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("method", ["iid", "circular_block", "stationary"])
def test_resample_counts(method: str) -> None:
    counts = resample_counts(_Streams(1), 100, 10, method, 50)  # type: ignore[arg-type]
    assert counts.shape == (50, 100) and np.all(counts.sum(axis=1) == 100)
    # Chunking replicates doesn't change the draws.
    s = _Streams(1)
    chunked = np.vstack(
        [resample_counts(s, 100, 10, method, 20) for _ in range(2)]
        + [resample_counts(s, 100, 10, method, 10)]
    )
    np.testing.assert_array_equal(counts, chunked)


def test_stationary_blocks_have_geometric_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    import teehr_core.metrics.bootstrap as bootstrap

    _, lengths, offsets = bootstrap._blocks(_Streams(5), 10_000, 25, "stationary", 40)
    assert np.all(np.add.reduceat(lengths, offsets[:-1]) == 10_000)
    full = np.ones(lengths.size, dtype=bool)
    full[offsets[1:] - 1] = False  # each resample's last block is cut
    assert 23 < lengths[full].mean() < 27
    # Resamples whose first draws fall short draw more blocks, the same however chunked.
    monkeypatch.setattr(bootstrap, "_stationary_draws", lambda n, block: 2)
    counts = resample_counts(_Streams(1), 100, 10, "stationary", 30)
    assert np.all(counts.sum(axis=1) == 100)
    s = _Streams(1)
    chunked = np.vstack([resample_counts(s, 100, 10, "stationary", 10) for _ in range(3)])
    np.testing.assert_array_equal(counts, chunked)


def test_circular_blocks_are_contiguous() -> None:
    streams = _Streams(3)
    counts = resample_counts(streams, 30, 30, "circular_block", 5)
    assert np.all(counts == 1)  # one block of the whole (circular) series


def settings(**kw: object) -> BootstrapSettings:
    base: dict[str, object] = {
        "method": "circular_block",
        "reps": 300,
        "seed": 42,
        "quantiles": (0.05, 0.95),
        "min_count": 30,
    }
    return BootstrapSettings(**{**base, **kw})  # type: ignore[arg-type]


def test_intervals_are_deterministic_and_partition_independent() -> None:
    obs, pred, gid = series(60)
    reqs = [
        MetricRequest("kling_gupta_efficiency"),
        MetricRequest("root_mean_square_error"),
        MetricRequest("spearman_correlation"),
    ]
    keys = [("a",), ("b",), ("c",)]
    blocks = np.full(3, 6)
    full = bootstrap_intervals(reqs, obs, pred, gid, keys, blocks, settings())
    again = bootstrap_intervals(reqs, obs, pred, gid, keys, blocks, settings())
    # Group "b" alone (as if in another shard) gets the same interval.
    sel = gid == 1
    alone = bootstrap_intervals(
        reqs,
        obs[sel],
        pred[sel],
        np.zeros(sel.sum(), dtype=np.int64),
        [("b",)],
        blocks[:1],
        settings(),
    )
    for r in reqs:
        np.testing.assert_array_equal(full[r], again[r])
        np.testing.assert_allclose(full[r][:, 1], alone[r][:, 0], rtol=1e-12)
        point = compute_metrics([r], obs, pred, gid, 3)[r][0]
        assert np.all(full[r][0] <= point) and np.all(point <= full[r][1])
    other = bootstrap_intervals(reqs, obs, pred, gid, keys, blocks, settings(seed=1))
    assert not np.array_equal(full[reqs[0]], other[reqs[0]])
    assert group_seed(42, ("a",)) != group_seed(42, ("b",))


def test_small_groups_and_undefined_replicates_are_null() -> None:
    obs, pred, gid = series(20)  # fewer than min_count
    req = MetricRequest("kling_gupta_efficiency")
    out = bootstrap_intervals([req], obs, pred, gid, [(1,), (2,), (3,)], np.full(3, 4), settings())
    assert np.all(np.isnan(out[req]))
    const = np.full(40, 5.0)  # constant obs: KGE undefined in every replicate
    out = bootstrap_intervals(
        [req],
        const,
        const + RNG.normal(size=40),
        np.zeros(40, dtype=np.int64),
        [()],
        np.array([4]),
        settings(min_count=10),
    )
    assert np.all(np.isnan(out[req]))


# -- water-year blocks -------------------------------------------------------------------


def test_gumboot_resamples_whole_years() -> None:
    """Recompute the intervals independently: draw whole years with the same streams."""
    years_per_group, days = 8, 30
    obs, pred, gid, years = [], [], [], []
    for g in range(2):
        for y in range(years_per_group):
            o = RNG.gamma(2.0, 50.0, size=days) + 1
            obs.append(o)
            pred.append(o * (1 + 0.1 * y) + RNG.normal(0, 5, size=days))  # error grows by year
            gid.append(np.full(days, g))
            years.append(np.full(days, 2000 + y))
    o, p, gi, yr = (np.concatenate(a) for a in (obs, pred, gid, years))
    req = MetricRequest("root_mean_square_error", None, Params())
    settings = BootstrapSettings("year_block", 200, 3, [0.05, 0.95], 30, min_years=5)
    keys = [("a",), ("b",)]
    got = bootstrap_intervals([req], o, p, gi, keys, np.ones(2, np.int64), settings, years=yr)[req]

    for g, key in enumerate(keys):
        streams = _Streams(group_seed(3, key))
        picks = streams.starts.integers(0, years_per_group, size=(200, years_per_group))
        sel = gi == g
        go, gp, gy = o[sel], p[sel], yr[sel] - 2000
        reps = [
            np.sqrt(np.mean(np.concatenate([(gp[gy == k] - go[gy == k]) ** 2 for k in row])))
            for row in picks
        ]
        assert got[:, g] == pytest.approx(np.quantile(reps, [0.05, 0.95]), rel=1e-12)


def test_gumboot_needs_enough_years() -> None:
    o = RNG.gamma(2.0, 50.0, size=120) + 1
    p = o * 1.1
    gid = np.zeros(120, np.int64)
    years = np.repeat([2001, 2002, 2003], 40)
    req = MetricRequest("root_mean_square_error", None, Params())
    settings = BootstrapSettings("year_block", 50, 1, [0.05, 0.95], 30, min_years=5)
    got = bootstrap_intervals([req], o, p, gid, [()], np.ones(1, np.int64), settings, years=years)
    assert np.all(np.isnan(got[req]))


@pytest.mark.skip(reason="needs the spec engine (teehr-core step 6)")
def test_water_years_and_gumboot_through_the_engine() -> None:
    from datetime import UTC, datetime, timedelta

    import pyarrow as pa
    from verirace import engine
    from verirace.contract import TIMESTAMP
    from verirace.engine import Catalog, Dataset
    from verirace.engine.duckdb_backend import _years

    t = pa.array(
        [datetime(2020, 9, 30, 23, tzinfo=UTC), datetime(2020, 10, 1, tzinfo=UTC)], TIMESTAMP
    )
    assert list(_years(pa.table({"value_time": t}), 10)) == [2020, 2021]
    assert list(_years(pa.table({"value_time": t}), 1)) == [2020, 2020]

    times = [datetime(2010, 1, 1, tzinfo=UTC) + timedelta(days=d) for d in range(0, 3650, 3)]
    obs = RNG.gamma(2.0, 50.0, size=len(times)) + 1
    common = {"variable": "v", "unit": "m", "statistic": "inst"}
    cat = Catalog(
        [
            Dataset(
                "obs",
                "observed",
                pa.table(
                    {
                        "series_key": ["o"] * len(times),
                        "value_time": pa.array(times, TIMESTAMP),
                        "value": obs,
                    }
                ),
                **common,
            ),  # type: ignore[arg-type]
            Dataset(
                "sim",
                "simulated",
                pa.table(
                    {
                        "series_key": ["s"] * len(times),
                        "value_time": pa.array(times, TIMESTAMP),
                        "value": obs * 1.2,
                    }
                ),
                **common,
            ),  # type: ignore[arg-type]
            Dataset(
                "xw",
                "auxiliary",
                pa.table({"left_key": ["o"], "right_key": ["s"]}),
                kind="crosswalk",
            ),
        ]
    )
    spec = {
        "version": 1,
        "name": "wy",
        "inputs": {"primary": {"dataset": "obs"}, "secondary": [{"dataset": "sim"}]},
        "pair": {"relation": "xw"},
        "aggregate": [
            {
                "group_by": ["primary_series_key"],
                "metrics": ["nash_sutcliffe_efficiency", "kling_gupta_efficiency"],
                "bootstrap": {"method": "year_block", "reps": 100, "seed": 5},
            }
        ],
    }
    row = engine.evaluate(spec, cat).table.to_pylist()[0]
    assert row["nash_sutcliffe_efficiency_q05"] <= row["nash_sutcliffe_efficiency_q95"]
    assert (
        row["kling_gupta_efficiency_q05"] is not None
        and row["kling_gupta_efficiency_q95"] is not None
    )


def test_gumboot_spec_checks() -> None:
    from pydantic import ValidationError

    from teehr_core.spec.models import Bootstrap

    with pytest.raises(ValidationError, match="no 'block'"):
        Bootstrap.model_validate({"method": "year_block", "block": "P7D"})
    assert Bootstrap.model_validate({"method": "year_block"}).year_start_month == 10


# -- block resamples as row ranges -------------------------------------------------------

#: Metrics bootstrapped from row ranges (SegmentStats), and two that need the weights.
SEGMENTED = [
    "count",
    "kling_gupta_efficiency",
    "kling_gupta_efficiency_mod1",
    "kling_gupta_efficiency_mod2",
    "mean_absolute_percentage_error",
    "mean_absolute_error",
    "mean_error",
    "nash_sutcliffe_efficiency_normalized",
    "nash_sutcliffe_efficiency",
    "pearson_correlation",
    "r_squared",
    "relative_bias",
    "root_mean_square_error",
    "symmetric_mean_absolute_percentage_error",
]
WEIGHTED = ["relative_median", "spearman_correlation"]


@pytest.mark.parametrize("method", ["circular_block", "iid", "stationary", "year_block"])
@pytest.mark.parametrize("transform", [None, "log"])
def test_row_ranges_match_weights(
    method: str, transform: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Block resamples evaluated from row ranges match evaluating them with weights."""
    import teehr_core.metrics.bootstrap as bootstrap

    obs, pred, gid = series(400, groups=4)
    obs[gid == 2] = 7.0  # a constant group: KGE, NSE and r undefined
    obs[5], pred[17], pred[30] = np.nan, np.inf, -3.0  # dropped (and by log only)
    keep = gid != 3
    obs, pred = (
        obs[keep | (np.arange(gid.size) % 20 == 0)],
        pred[keep | (np.arange(gid.size) % 20 == 0)],
    )
    gid = gid[keep | (np.arange(gid.size) % 20 == 0)]  # group 3: 20 pairs, below min_count
    years = 2000 + np.concatenate([np.arange(k) * 9 // k for k in np.bincount(gid)])
    reqs = [MetricRequest(name, transform) for name in [*SEGMENTED, *WEIGHTED]]  # type: ignore[arg-type]
    keys = [("a",), ("b",), ("c",), ("d",)]
    args = (reqs, obs, pred, gid, keys, np.array([37, 50, 40, 5]), settings(method=method))
    fast = bootstrap_intervals(*args, years=years)
    segmented = bootstrap._segment_replicates
    used: list[MetricRequest] = []
    monkeypatch.setattr(
        bootstrap,
        "_segment_replicates",
        lambda *a: used.extend(out := segmented(*a)) or out,  # type: ignore[func-returns-value]
    )
    bootstrap_intervals(*args, years=years)
    assert {r.name for r in used} == set(SEGMENTED)
    monkeypatch.setattr(bootstrap, "_pieces", lambda *a: None)
    dense = bootstrap_intervals(*args, years=years)
    for r in reqs:
        np.testing.assert_allclose(fast[r], dense[r], rtol=1e-9, atol=1e-12, equal_nan=True)
    assert np.all(np.isnan(fast[MetricRequest("kling_gupta_efficiency", transform)][:, [2, 3]]))  # type: ignore[arg-type]


def test_row_ranges_need_contiguous_units() -> None:
    """Cases interleaved in row order can't be ranges: those use the weights."""
    import teehr_core.metrics.bootstrap as bootstrap

    local = [np.array([0, 1, 0, 1])]
    s = settings(method="iid", min_count=1)
    assert bootstrap._pieces(local, np.array([0, 4]), np.ones(1, np.int64), [()], s, "iid") is None
    assert (
        bootstrap._pieces(local, np.array([0, 4]), np.ones(1, np.int64), [()], s, "stationary")
        is None
    )


@pytest.mark.parametrize(
    ("method", "unit", "block"),
    [
        ("circular_block", "day", 7),
        ("stationary", "day", 10),
        ("circular_block", "month", 2),
        ("stationary", "month", 1),
        ("year_block", "year", 1),
    ],
)
def test_state_per_time_unit_gives_the_pairs_intervals(method: str, unit: str, block: int) -> None:
    """Blocks of whole days, months or years drawn from state equal those drawn from pairs."""
    from teehr_core.metrics.bootstrap import state_intervals
    from teehr_core.metrics.registry import Transform, transformed
    from teehr_core.metrics.state import group_state

    rng = np.random.default_rng(3)
    parts = [np.sort(rng.choice(24 * 900, 6000, replace=False)) + g * 500 for g in range(3)]
    gid = np.repeat(np.arange(3), [p.size for p in parts])
    hours = np.concatenate(parts)  # gaps: some days and months have no pairs
    obs = np.exp(rng.normal(2, 1, gid.size))
    pred = obs * 1.2 + rng.normal(0, 2, gid.size)
    obs[::97] = np.nan  # pairs without a value: drawn, but counting for nothing
    per = {"day": hours // 24, "month": hours // (24 * 30), "year": hours // (24 * 365)}[unit]
    names = [
        "root_mean_square_error",
        "nash_sutcliffe_efficiency",
        "kling_gupta_efficiency",
        "pearson_correlation",
        "relative_bias",
        "maximum",
        "minimum",
    ]
    reqs = [MetricRequest(m) for m in names] + [
        MetricRequest("root_mean_square_error", Transform("log"))
    ]
    keys = [("a",), ("b",), ("c",)]
    settings = BootstrapSettings(method, 200, 5, [0.05, 0.5, 0.95], 30, 2)  # type: ignore[arg-type]
    by_year = method == "year_block"
    want = bootstrap_intervals(
        reqs, obs, pred, gid, keys, np.full(3, block), settings,
        years=per if by_year else None, periods=None if by_year else per,
    )  # fmt: skip

    cells, row = np.unique(np.stack([gid, per], 1), axis=0, return_inverse=True)
    states = {}
    for t in {r.transform for r in reqs}:
        o, p, keep = transformed(obs, pred, t)
        states[t] = group_state(
            np.where(keep, o, np.nan), np.where(keep, p, np.nan), row.ravel(), len(cells)
        )
    owner = cells[:, 0]
    first = np.array([cells[owner == g, 1].min() for g in range(3)])
    rank = np.concatenate([np.arange((owner == g).sum()) for g in range(3)])
    position = rank if by_year else cells[:, 1] - first[owner]
    cases = np.bincount(gid[np.isfinite(obs)], minlength=3)
    got = state_intervals(reqs, states, owner, position, keys, block, settings, cases)
    for r in reqs:
        np.testing.assert_allclose(got[r], want[r], rtol=1e-11, equal_nan=True)


def test_group_seed_encodes_keys_as_the_shared_standard() -> None:
    """blake2b of compact JSON; timestamps ISO 8601 UTC with Z, whole floats with .0."""
    import hashlib
    from datetime import UTC, datetime, timedelta, timezone

    from teehr_core.metrics.bootstrap import group_seed

    def expected(text: str) -> int:
        return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")

    at = datetime(2024, 1, 1, 5, tzinfo=timezone(timedelta(hours=5)))  # midnight UTC
    assert group_seed(7, ["usgs-1", at, 1.0, None]) == expected(
        '[7,"usgs-1","2024-01-01T00:00:00Z",1.0,null]'
    )
    fraction = datetime(2024, 1, 1, 0, 0, 0, 500_000, tzinfo=UTC)
    assert group_seed(0, [fraction]) == expected('[0,"2024-01-01T00:00:00.5Z"]')
