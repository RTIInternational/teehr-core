"""Property-based checks of metric invariants."""

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from teehr_core import metrics
from teehr_core.metrics.stats import CONSTANT_RTOL

values = st.floats(min_value=0.01, max_value=1e5, allow_nan=False, allow_infinity=False)
series = arrays(np.float64, st.integers(min_value=2, max_value=60), elements=values)


def varies(x: np.ndarray) -> bool:
    """The engine's definition: spread larger than floating-point noise."""
    return bool(x.max() - x.min() > CONSTANT_RTOL * max(abs(x.max()), abs(x.min())))


def well_conditioned(x: np.ndarray) -> bool:
    """Spread well above rounding error, so invariance holds to tight tolerances."""
    return bool(x.max() - x.min() > 1e-6 * max(abs(x.max()), abs(x.min())))


@given(series)
@settings(max_examples=200, deadline=None)
def test_perfect_prediction(obs: np.ndarray) -> None:
    assert metrics.mean_error(obs, obs) == 0
    assert metrics.rmse(obs, obs) == 0
    assert metrics.relative_bias(obs, obs) == 0
    if varies(obs):
        assert metrics.nse(obs, obs) == 1
        assert metrics.pearson_r(obs, obs) is not None
        assert abs(metrics.pearson_r(obs, obs) - 1) < 1e-12  # type: ignore[operator]
        assert abs(metrics.kge(obs, obs) - 1) < 1e-12  # type: ignore[operator]
    else:
        assert metrics.nse(obs, obs) is None
        assert metrics.kge(obs, obs) is None


@given(series, st.randoms(use_true_random=False))
@settings(max_examples=100, deadline=None)
def test_row_order_does_not_matter(obs: np.ndarray, rnd: object) -> None:
    assume(well_conditioned(obs))
    sim = obs * 1.1 + 3.0
    perm = np.random.default_rng(abs(hash(rnd)) % 2**32).permutation(obs.size)
    for name in [
        "mean_error",
        "relative_bias",
        "root_mean_square_error",
        "nash_sutcliffe_efficiency",
        "kling_gupta_efficiency",
        "pearson_correlation",
    ]:
        a = getattr(metrics, name)(obs, sim)
        b = getattr(metrics, name)(obs[perm], sim[perm])
        assert (a is None and b is None) or abs(a - b) <= 1e-9 * max(1.0, abs(a)), name


@given(series, st.floats(min_value=0.1, max_value=10))
@settings(max_examples=100, deadline=None)
def test_correlation_is_scale_invariant(obs: np.ndarray, k: float) -> None:
    assume(well_conditioned(obs))
    sim = obs[::-1].copy()
    a = metrics.pearson_r(obs, sim)
    b = metrics.pearson_r(obs, sim * k)
    assert (a is None and b is None) or abs(a - b) < 1e-9  # type: ignore[operator]


@given(series)
@settings(max_examples=100, deadline=None)
def test_bounds(obs: np.ndarray) -> None:
    sim = obs[::-1].copy()
    nse, kge, r = metrics.nse(obs, sim), metrics.kge(obs, sim), metrics.pearson_r(obs, sim)
    assert nse is None or nse <= 1 + 1e-12
    assert kge is None or kge <= 1 + 1e-12
    assert r is None or -1 - 1e-12 <= r <= 1 + 1e-12
