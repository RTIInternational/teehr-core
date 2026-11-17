import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from teehr_core import metrics
from teehr_core.metrics import compute_grouped
from tests.golden_utils import assert_matches
from tests.spec_shim import output_name

#: Metrics callable as f(obs, sim): no required options, no times, two-sided.
METRIC_NAMES = [
    name
    for name, m in metrics.METRICS.items()
    if not m.required and not m.needs_time and m.family not in ("signature", "probabilistic")
]


def vector_cases(golden_dir: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = json.loads((golden_dir / "metric_vectors.json").read_text())[
        "cases"
    ]
    return cases


def test_registry_matches_golden_definitions(golden_dir: Path) -> None:
    defs = json.loads((golden_dir / "metric_vectors.json").read_text())["metric_definitions"]
    assert set(metrics.metric_names()) == set(defs)


@pytest.mark.parametrize("metric", METRIC_NAMES)
def test_matches_exact_vectors(golden_dir: Path, metric: str) -> None:
    for case in vector_cases(golden_dir):
        expected = case["expected"][metric]
        got = getattr(metrics, metric)(case["obs"], case["sim"])
        if expected is None:
            assert got is None, case["name"]
        else:
            assert got == pytest.approx(expected, rel=1e-12, abs=1e-12), case["name"]


def test_variants_match_exact_vectors(golden_dir: Path) -> None:
    """Parameterized metrics (thresholds, normalizers, sides, times) via evaluate()."""
    data = json.loads((golden_dir / "metric_vectors.json").read_text())
    variants = [s for s in data["metric_specs"] if isinstance(s, dict)]
    for case in data["cases"]:
        for spec in variants:
            output = output_name(spec)
            options = {k: v for k, v in spec.items() if k not in ("name", "as")}
            got = metrics.evaluate(
                case["obs"], case["sim"], [spec["name"]], times=case["times"], **options
            )[spec["name"]]
            assert_matches(got, case["expected"][output], (case["name"], output))


def test_threshold_forms_agree() -> None:
    obs, sim = [1.0, 5, 2, 8, 3, 9], [2.0, 6, 7, 1, 3, 9]
    fixed = metrics.csi(obs, sim, threshold=8)
    assert metrics.csi(obs, sim, threshold=[8] * 6) == fixed
    assert (
        metrics.evaluate(
            obs, sim, ["critical_success_index"], threshold={"primary_quantile": 0.75}
        )["critical_success_index"]
        == fixed
    )
    with pytest.raises(ValueError, match="needs a threshold"):
        metrics.evaluate(obs, sim, ["probability_of_detection"])
    with pytest.raises(ValueError, match="needs times"):
        metrics.evaluate(obs, sim, ["max_value_time_delta"])


def test_non_finite_pairs_dropped() -> None:
    obs = [1.0, 2.0, float("nan"), 4.0, 5.0]
    sim = [1.5, 1.5, 9.0, 3.0, float("inf")]
    assert metrics.count(obs, sim) == 3
    assert metrics.rmse(obs, sim) == metrics.rmse([1, 2, 4], [1.5, 1.5, 3])


def test_transform_log_excludes_non_positive() -> None:
    obs = [1.0, 10.0, 100.0, 0.0]
    sim = [1.0, 10.0, 100.0, 5.0]
    assert metrics.nse(obs, sim, transform="log") == 1.0
    assert metrics.count(obs, sim, transform="log") == 3


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="equal length"):
        metrics.nse([1, 2, 3], [1, 2])


def test_grouped_equals_per_group() -> None:
    rng = np.random.default_rng(0)
    obs = rng.gamma(2.0, 50.0, 300)
    sim = obs * rng.normal(1.05, 0.2, 300)
    gid = rng.integers(0, 7, 300)
    for name in METRIC_NAMES:
        grouped = compute_grouped(name, obs, sim, gid, 7)
        for g in range(7):
            single = getattr(metrics, name)(obs[gid == g], sim[gid == g])
            assert grouped[g] == pytest.approx(single, rel=1e-12), (name, g)


def test_empty_group_is_undefined_except_count() -> None:
    obs = np.array([1.0, 2.0, 3.0])
    sim = np.array([1.0, 2.5, 2.0])
    gid = np.zeros(3, dtype=np.int64)
    assert compute_grouped("count", obs, sim, gid, 2)[1] == 0
    for name in METRIC_NAMES:
        if name != "count":
            assert math.isnan(compute_grouped(name, obs, sim, gid, 2)[1]), name


def test_near_constant_input_is_undefined() -> None:
    # Two values one rounding step apart: any correlation computed from them is noise
    # (the exact answer here is -1; naive two-pass arithmetic gives 0). Found by Hypothesis.
    a = 0.01
    obs = np.array([a, np.nextafter(a, 1)])
    sim = obs[::-1].copy()
    for name in ["nash_sutcliffe_efficiency", "kling_gupta_efficiency", "pearson_correlation"]:
        assert getattr(metrics, name)(obs, sim) is None, name
    assert metrics.rmse(obs, sim) is not None  # error metrics stay defined


def test_small_but_real_variation_is_defined() -> None:
    obs = np.array([1000.0, 1000.001, 1000.002])
    assert metrics.pearson_r(obs, obs) == pytest.approx(1.0)


def test_shared_transforms() -> None:
    """The shared transform list: log (natural), sqrt, square, cube, exp, reciprocal (inv), abs."""
    obs, pred = [1.0, 2.0, 3.0, -1.0], [1.5, 1.0, 3.5, -2.0]
    for name, fn in (
        ("cube", lambda x: x**3),
        ("exp", math.exp),
        ("abs", abs),
        ("reciprocal", lambda x: 1 / x),
    ):
        expect = sum(fn(p) - fn(o) for o, p in zip(obs, pred, strict=True)) / len(obs)
        assert metrics.mean_error(obs, pred, transform=name) == pytest.approx(expect)
    # log leaves out the pair it's undefined for (-1, -2).
    assert metrics.count(obs, pred, transform="log") == 3
    assert metrics.count([1.0, 1000.0], [1.0, 1.0], transform="exp") == 1  # exp(1000) overflows
    # In a spec, transform "inv" is an alias of "reciprocal": tested with the spec models.


def test_kge_weights() -> None:
    """sr, sa, sb weight KGE's three terms; with sr = 0 only variability and bias remain."""
    obs = [1.0, 2.0, 3.0, 4.0, 6.0]
    pred = [1.5, 1.5, 3.5, 3.0, 7.0]
    o, p = np.array(obs), np.array(pred)
    alpha, beta = p.std() / o.std(), p.mean() / o.mean()
    got = metrics.evaluate(obs, pred, ["kge"], kge_weights=(0.0, 2.0, 1.0))
    expect = 1 - math.sqrt(2 * (alpha - 1) ** 2 + (beta - 1) ** 2)
    assert got["kling_gupta_efficiency"] == pytest.approx(expect)
    default = metrics.evaluate(obs, pred, ["kge"])["kling_gupta_efficiency"]
    assert (
        default
        == metrics.evaluate(obs, pred, ["kge"], kge_weights=(1, 1, 1))["kling_gupta_efficiency"]
    )
