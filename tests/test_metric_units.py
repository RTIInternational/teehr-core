"""Metric units: each metric's kind and converting results.

Ported from VeriRace (``tests/unit/test_metric_units.py`` at 668ab7d); its tests that run
evaluations through the API come with the engine (step 6).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from teehr_core.metrics.registry import METRICS, MetricRequest, Params, compute_metrics
from teehr_core.metrics.units import (
    KINDS,
    NotConvertible,
    UnitKind,
    convert_results,
    kind_factor,
    metric_kind,
    result_factors,
)
from teehr_core.model.units import Conversion
from teehr_core.spec import load_spec
from teehr_core.spec.models import MetricSpec

CFS_TO_CMS = Conversion(0.028316846592, 0.0)
F_TO_C = Conversion(5 / 9, -32 * 5 / 9)
#: Metrics that need ensemble or quantile columns (their kinds are checked by formula).
_COLUMNS = {
    "interval_coverage",
    "mean_interval_width",
    "mean_pinball_loss",
    "mean_interval_score",
    "mean_absolute_scaled_error",
}
_THRESHOLD = {
    "hits", "misses", "false_alarms", "correct_negatives", "probability_of_detection",
    "probability_of_false_detection", "false_alarm_ratio", "critical_success_index",
    "success_ratio", "frequency_bias_index", "mean_brier_score",
}  # fmt: skip


def test_every_metric_has_a_kind() -> None:
    assert set(KINDS) == set(METRICS)


def _compute(name: str, obs: Any, pred: Any, threshold: float, times: Any) -> float:
    params = Params(threshold=threshold) if name in _THRESHOLD else Params()
    req = MetricRequest(name, None, params)
    columns = {c: times for c in req.columns() if c == "value_time"}
    out = compute_metrics([req], obs, pred, np.zeros(len(obs), np.int64), 1, columns=columns)
    return float(out[req][0][0])


@pytest.mark.parametrize("conv", [CFS_TO_CMS, F_TO_C], ids=["scale", "offset"])
def test_converting_a_result_matches_computing_in_the_unit(conv: Conversion) -> None:
    """A converted result equals the metric computed on converted data."""
    rng = np.random.default_rng(3)
    n = 1500
    obs = 50 + 30 * np.sin(np.arange(n) / 40) + rng.gamma(2, 5, n)
    pred = obs * 1.08 + rng.normal(0, 6, n) + 2
    times = np.arange(n) * 3600.0 + 1.7e9
    checked = 0
    for name in sorted(set(METRICS) - _COLUMNS):
        try:
            before = _compute(name, obs, pred, 80.0, times)
        except Exception:  # ensemble metrics need member columns
            continue
        if math.isnan(before):
            continue
        after = _compute(
            name, obs * conv.scale + conv.offset, pred * conv.scale + conv.offset,
            80.0 * conv.scale + conv.offset, times,
        )  # fmt: skip
        factor = kind_factor(KINDS[name], None, conv)
        if factor is None:
            assert KINDS[name] in (UnitKind.RATIO, UnitKind.SUM) and conv.offset != 0
            assert not math.isclose(after, before, rel_tol=1e-6), name  # really changes
            continue
        s, t = factor
        assert math.isclose(before * s + t, after, rel_tol=1e-9, abs_tol=1e-9), name
        checked += 1
    assert checked >= 30


def test_options_and_transforms_change_the_kind() -> None:
    assert metric_kind(MetricSpec(name="normalized_mean_absolute_error")) is UnitKind.RATIO
    assert (
        metric_kind(MetricSpec(name="normalized_mean_absolute_error", normalize_by="range_primary"))
        is UnitKind.INVARIANT
    )
    spec = load_spec(
        {
            "version": 1,
            "name": "u",
            "inputs": {"primary": {"dataset": "o"}, "secondary": {"dataset": "p"}},
            "pair": {"on_attribute": "series_key"},
            "aggregate": [
                {
                    "group_by": ["primary_series_key", "secondary_dataset"],
                    "metrics": [
                        "root_mean_square_error",
                        "mean_square_error",
                        "kling_gupta_efficiency",
                        {"name": "root_mean_square_error", "transform": "log"},
                        {"name": "root_mean_square_error", "transform": "sqrt"},
                        {"name": "average"},
                    ],
                    "bootstrap": {"method": "iid", "reps": 10},
                    "skill": {"reference": "p", "metrics": ["root_mean_square_error"]},
                },
                {
                    "group_by": ["secondary_dataset"],
                    "metrics": [
                        {"name": "median", "of": "root_mean_square_error"},
                        {"name": "std", "of": "average_primary"},
                        {"name": "sum", "of": "average_primary"},
                        {"name": "mean", "of": "root_mean_square_error_skill"},
                    ],
                },
            ],
        }
    )
    a = CFS_TO_CMS.scale
    first = result_factors(spec.model_copy(update={"aggregate": spec.aggregate[:1]}), CFS_TO_CMS)
    assert first["root_mean_square_error"] == first["root_mean_square_error_q05"] == (a, 0.0)
    assert first["mean_square_error"] == (a * a, 0.0)
    assert first["root_mean_square_error_log"] == (1.0, 0.0) and first[
        "kling_gupta_efficiency"
    ] == (1.0, 0.0)
    assert first["root_mean_square_error_sqrt"] == pytest.approx((math.sqrt(a), 0.0))
    assert first["root_mean_square_error_skill"] == (1.0, 0.0)
    last = result_factors(spec, F_TO_C)
    assert last["median_root_mean_square_error"] == (5 / 9, 0.0)
    assert last["std_average_primary"] == (5 / 9, 0.0)
    assert last["sum_average_primary"] is None  # a sum of levels with an offset
    assert last["mean_root_mean_square_error_skill"] == (1.0, 0.0)


def test_convert_results_refuses_ratios_with_an_offset() -> None:
    import pyarrow as pa

    spec = load_spec(
        {
            "version": 1,
            "name": "u",
            "inputs": {"primary": {"dataset": "o"}, "secondary": {"dataset": "p"}},
            "pair": {"on_attribute": "series_key"},
            "aggregate": [
                {
                    "group_by": ["primary_series_key"],
                    "metrics": ["root_mean_square_error", "kling_gupta_efficiency", "count"],
                }
            ],
        }
    )
    table = pa.table(
        {
            "primary_series_key": ["a"],
            "root_mean_square_error": [9.0],
            "kling_gupta_efficiency": [0.8],
            "count": [10],
        }
    )
    out = convert_results(table, spec, CFS_TO_CMS).to_pylist()[0]
    assert (
        out["root_mean_square_error"] == pytest.approx(9 * CFS_TO_CMS.scale)
        and out["kling_gupta_efficiency"] == 0.8
    )
    assert out["count"] == 10
    with pytest.raises(NotConvertible, match="kling_gupta_efficiency"):
        convert_results(table, spec, F_TO_C)
    no_ratio = table.drop_columns(["kling_gupta_efficiency"])
    spec_rmse = spec.model_copy(
        update={
            "aggregate": [
                spec.aggregate[0].model_copy(update={"metrics": spec.aggregate[0].metrics[::2]})
            ]
        }
    )
    assert (
        convert_results(no_ratio, spec_rmse, F_TO_C).to_pylist()[0]["root_mean_square_error"] == 5.0
    )
