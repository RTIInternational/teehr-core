"""Mergeable state: metrics from parts merged equal metrics from all the pairs at once."""

from __future__ import annotations

import duckdb
import numpy as np
import pyarrow as pa
import pytest

from teehr_core.metrics.registry import METRICS, MetricRequest, Transform, compute_metrics
from teehr_core.metrics.state import (
    SKETCH,
    SKETCH_TYPE,
    STATE_COLUMNS,
    additive_columns,
    compute_from_state,
    group_state,
    merge_sql,
    merge_state,
    mergeable,
    sketches_from,
)
from teehr_core.spec.models import MetricSpec

#: The platform evaluations' metrics (deploy/flows/evaluations): incremental runs need them.
PLATFORM = [
    "count",
    "mean_error",
    "relative_bias",
    "root_mean_square_error",
    "mean_absolute_error",
    "nash_sutcliffe_efficiency",
    "kling_gupta_efficiency",
]


def _pairs(seed: int = 7) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    sizes = [50, 7, 1, 30, 12, 20]  # group 2 has one pair
    gid = np.repeat(np.arange(len(sizes)), sizes)
    obs = rng.gamma(2.0, 50.0, gid.size) + 1e3 * gid  # large means: tests the centering
    pred = obs * rng.normal(1.0, 0.2, gid.size) + rng.normal(0, 5, gid.size)
    obs[gid == 4] = 42.0  # constant observations
    pred[gid == 5] = np.nan  # no finite pairs at all
    obs[3] = -1.0  # a negative value (log drops it)
    vt = np.concatenate([np.arange(n) * 3600.0 for n in sizes])
    return obs, pred, gid, vt


def _ensemble(seed: int = 11) -> tuple[np.ndarray, ...]:
    """Forecast cases of three members (or quantiles 0.05, 0.5, 0.95) each, per group.

    Members of a case share its value time (so a case never splits between parts), and
    each pair has a threshold of its own (``thr``) and a naive scale (``scale``).
    """
    rng = np.random.default_rng(seed)
    cases = [40, 9, 1, 25]
    case_gid = np.repeat(np.arange(len(cases)), cases)
    n_cases = case_gid.size
    case_obs = rng.gamma(2.0, 50.0, n_cases)
    members = 3
    gid = np.repeat(case_gid, members)
    obs = np.repeat(case_obs, members)
    pred = obs * rng.normal(1.0, 0.3, obs.size)
    pred[7] = obs[7]  # a member equal to the observation (rank ties)
    vt = np.repeat(np.concatenate([np.arange(n) * 3600.0 for n in cases]), members)
    columns = {
        "value_time": vt,
        "_case": np.repeat(np.arange(n_cases), members).astype(np.float64),
        "quantile": np.tile([0.05, 0.5, 0.95], n_cases),
        "thr": rng.uniform(50, 150, obs.size),
        "scale": rng.uniform(5, 20, obs.size),
    }
    return obs, pred, gid, columns


def _merged(
    requests: list[MetricRequest],
    parts: int,
    seed: int = 3,
    data: tuple[np.ndarray, ...] | None = None,
    engine: str = "sql",
) -> dict[MetricRequest, np.ndarray]:
    """Split each group's pairs by time into ``parts`` and merge their states.

    ``engine``: ``sql`` (merge_sql in DuckDB), ``numpy`` (merge_state), or ``twice``
    (merge_state on two halves of the parts, then on the two results).
    """
    if data is None:
        obs, pred, gid, vt = _pairs()
        columns: dict[str, np.ndarray] = {}
    else:
        obs, pred, gid, columns = data  # type: ignore[assignment]
        vt = columns["value_time"]
        columns = {k: v for k, v in columns.items() if k != "value_time"}
    n_groups = int(gid.max()) + 1
    additive = additive_columns(requests)
    cuts = np.sort(np.random.default_rng(seed).uniform(0, vt.max(), parts - 1))
    part = np.searchsorted(cuts, vt)
    states = {}
    for transform in {r.transform for r in requests}:
        rows = []
        for p in range(parts):
            m = part == p
            reqs = [r for r in requests if r.transform == transform]
            cols = {k: v[m] for k, v in columns.items()}
            st = group_state(obs[m], pred[m], gid[m], n_groups, vt[m], transform, reqs, cols)
            arrays = {
                c: pa.array(st[c], SKETCH_TYPE)
                if c.startswith(SKETCH)
                else st.get(c, np.zeros(n_groups))
                for c in [*STATE_COLUMNS, *additive]
            }
            rows.append(
                pa.table({"g": np.arange(n_groups), "part": np.full(n_groups, p), **arrays})
            )
        if engine == "sql":
            states[transform] = _merge_in_sql(rows, additive)
        else:
            if engine == "twice":
                halves = [rows[: parts // 2], rows[parts // 2 :]]
                rows = [
                    pa.table({"g": np.arange(n_groups), **_as_columns(m, additive)})
                    for m in (_merge_in_numpy(h, additive, n_groups) for h in halves if h)
                ]
            merged = _merge_in_numpy(rows, additive, n_groups)
            states[transform] = {
                c: sketches_from(v) if c.startswith(SKETCH) else v for c, v in merged.items()
            }
    return compute_from_state(requests, states)


def _as_columns(state: dict, additive: list[str]) -> dict[str, object]:
    return {
        c: pa.array(state[c], SKETCH_TYPE) if c.startswith(SKETCH) else state[c]
        for c in [*STATE_COLUMNS, *additive]
    }


def _merge_in_sql(rows: list[pa.Table], additive: list[str]) -> dict:
    con = duckdb.connect()
    con.register("t", pa.concat_tables(rows))
    sql = merge_sql(["g"], "t", additive)
    merged = con.sql(f"SELECT * FROM ({sql}) ORDER BY g").to_arrow_table()
    # Nulls (groups without pairs) as NaN, as group_state gives them.
    return {
        c: sketches_from(merged[c])
        if c.startswith(SKETCH)
        else np.asarray(merged[c].to_numpy(zero_copy_only=False), np.float64)
        for c in [*STATE_COLUMNS, *additive]
    }


def _merge_in_numpy(rows: list[pa.Table], additive: list[str], n_groups: int) -> dict:
    table = pa.concat_tables(rows)
    state = {
        c: table[c].to_pylist() if c.startswith(SKETCH) else table[c].to_numpy(zero_copy_only=False)
        for c in [*STATE_COLUMNS, *additive]
    }
    return merge_state(state, table["g"].to_numpy(), n_groups, additive)


def _direct(
    requests: list[MetricRequest], data: tuple[np.ndarray, ...] | None = None
) -> dict[MetricRequest, np.ndarray]:
    if data is None:
        obs, pred, gid, vt = _pairs()
        columns = {"value_time": vt}
    else:
        obs, pred, gid, columns = data  # type: ignore[assignment]
    return compute_metrics(requests, obs, pred, gid, int(gid.max()) + 1, columns=columns)


def _simple() -> list[MetricRequest]:
    return [MetricRequest(name) for name, m in METRICS.items() if not m.required]


def test_platform_metrics_are_mergeable_and_pair_metrics_are_not() -> None:
    assert mergeable([MetricRequest(n) for n in [*PLATFORM, "pearson_correlation", "r_squared"]])
    for name in ("median", "spearman_correlation", "rank_histogram_reliability"):
        if name in METRICS:
            assert not mergeable([MetricRequest(name)]), name


ENGINES = ["sql", "numpy", "twice"]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("parts", [1, 2, 5])
def test_merged_state_gives_the_same_metrics(parts: int, engine: str) -> None:
    # On plain pairs: metrics reading forecast cases or quantiles are tested on _ensemble.
    requests = [r for r in _simple() if mergeable([r]) and METRICS[r.name].needs <= {"value_time"}]
    assert len(requests) >= 20  # most error and summary metrics
    got, want = _merged(requests, parts, engine=engine), _direct(requests)
    for r in requests:
        np.testing.assert_allclose(got[r], want[r], rtol=1e-9, atol=1e-9, err_msg=r.name)


@pytest.mark.parametrize("engine", ENGINES)
def test_transformed_metrics_merge_too(engine: str) -> None:
    requests = [
        MetricRequest("root_mean_square_error", Transform.LOG),
        MetricRequest("nash_sutcliffe_efficiency", Transform.SQRT),
    ]
    assert mergeable(requests)
    got, want = _merged(requests, 4, engine=engine), _direct(requests)
    for r in requests:
        np.testing.assert_allclose(got[r], want[r], rtol=1e-9, err_msg=str(r))


def _request(**spec: object) -> MetricRequest:
    m = MetricSpec(**spec)  # type: ignore[arg-type]
    return MetricRequest(m.name, m.transform, m.to_params())


#: Metrics kept in state as additive sums of their own: counts at fixed thresholds and
#: per-case scores (ADR: docs/design/state-bootstrap.md, "Additive sums").
ADDITIVE_METRICS = [
    _request(name="hits", threshold={"value": 100.0, "unit": "m"}),
    _request(name="critical_success_index", threshold={"value": 100.0, "unit": "m"}),
    _request(name="probability_of_detection", threshold={"column": "thr"}),
    _request(name="false_alarm_ratio", threshold={"column": "thr"}),
    _request(name="mean_crps_ensemble"),
    _request(name="mean_crps_ensemble", estimator="fair"),
    _request(name="mean_brier_score", threshold={"value": 100.0, "unit": "m"}),
    _request(name="mean_pinball_loss"),
    _request(name="interval_coverage", interval=[0.05, 0.95]),
    _request(name="mean_interval_width", interval=[0.05, 0.95]),
    _request(name="mean_interval_score", interval=[0.05, 0.95]),
    _request(name="ensemble_spread"),
    _request(name="ensemble_mean_root_mean_square_error"),
    _request(name="spread_skill_ratio"),
    _request(name="rank_frequency", rank=1),
    _request(name="rank_outlier_fraction"),
    _request(name="mean_absolute_scaled_error", scale={"column": "scale"}),
    _request(name="root_mean_square_error"),
    # Quantile sketches.
    _request(name="relative_median"),
    _request(name="flow_duration_curve_slope"),
    _request(name="flow_duration_curve_slope", side="secondary", exceedance=[0.1, 0.9]),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("parts", [1, 3, 7])
def test_additive_sums_merge_exactly(parts: int, engine: str) -> None:
    assert all(mergeable([r]) for r in ADDITIVE_METRICS)
    data = _ensemble()
    got = _merged(ADDITIVE_METRICS, parts, data=data, engine=engine)
    want = _direct(ADDITIVE_METRICS, data)
    for r in ADDITIVE_METRICS:
        np.testing.assert_allclose(got[r], want[r], rtol=1e-9, atol=1e-12, err_msg=r.name)


def test_thresholds_from_the_groups_observations_need_the_pairs() -> None:
    for name in ("hits", "mean_brier_score"):
        r = _request(name=name, threshold={"primary_quantile": 0.9})
        assert r.params.threshold_quantile == 0.9
        assert not mergeable([r]) and additive_columns([r]) == []


def test_additive_columns_name_their_parameters() -> None:
    assert additive_columns(
        [_request(name="mean_crps_ensemble"), _request(name="mean_crps_ensemble", estimator="fair")]
    ) == [
        "+crps|fair|cases",
        "+crps|fair|sum",
        "+crps|standard|cases",
        "+crps|standard|sum",
    ]
    assert (
        additive_columns(
            [_request(name="root_mean_square_error"), _request(name="spearman_correlation")]
        )
        == []
    )
    assert additive_columns([_request(name="relative_median")]) == ["~obs", "~pred"]
    assert additive_columns([_request(name="flow_duration_curve_slope")]) == ["~obs"]


def test_quantile_metrics_are_within_the_sketch_accuracy() -> None:
    """Sketch quantiles are within 0.1% of the exact ones (here: a few times over)."""
    from teehr_core.metrics.sketch import ALPHA

    obs, pred, gid, _ = _pairs()
    keep = np.isfinite(pred) & (gid != 4)  # varying, finite groups
    obs, pred, gid = obs[keep], pred[keep], np.unique(gid[keep], return_inverse=True)[1]
    r = _request(name="relative_median")
    got = compute_metrics([r], obs, pred, gid, int(gid.max()) + 1)[r][0]
    exact = [
        np.quantile(pred[gid == g], 0.5, method="averaged_inverted_cdf")
        / np.quantile(obs[gid == g], 0.5, method="averaged_inverted_cdf")
        for g in range(int(gid.max()) + 1)
    ]
    np.testing.assert_allclose(got, exact, rtol=2.5 * ALPHA)
