"""Metric registry: each metric is a formula over shared components (``stats``).

Ported from VeriRace (``verirace/metrics/registry.py`` at 668ab7d).

A stage's metrics are evaluated together on one :class:`~teehr_core.metrics.stats.GroupStats`
per transform, so components they share (sums, centered sums of squares, sorted values)
are computed once. The same formulas evaluate bootstrap replicates (weighted pairs).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from teehr_core.metrics.stats import FloatArray, GroupStats, IntArray
from teehr_core.metrics.timing import timing

__all__ = [
    "ALIASES",
    "METRICS",
    "TRANSFORM_ALIASES",
    "Metric",
    "MetricRequest",
    "OffsetTransform",
    "Params",
    "Transform",
    "TransformKey",
    "aliases_of",
    "base_transform",
    "canonical_name",
    "compute_grouped",
    "compute_metrics",
    "metric_names",
    "parse_transform_label",
    "transform_label",
    "transformed",
]


class Transform(StrEnum):
    """Transform applied to obs and pred before a metric is computed (the shared list).

    Pairs where the transform is undefined or not finite (``log`` of a non-positive
    value, ``exp`` overflowing) are excluded from that metric only.
    """

    LOG = "log"  # natural
    SQRT = "sqrt"
    SQUARE = "square"
    CUBE = "cube"
    EXP = "exp"
    RECIPROCAL = "reciprocal"
    ABS = "abs"


#: Other names a spec may give a transform.
TRANSFORM_ALIASES: dict[str, Transform] = {"inv": Transform.RECIPROCAL}


@dataclass(frozen=True)
class OffsetTransform:
    """A transform of the values shifted first: ``transform(x + offset)``.

    ``offset`` is in the evaluation unit, e.g. ``log(Q + 0.01)`` keeps zero flows a
    plain ``log`` would leave out (shared spec: a transform's ``offset``).
    """

    transform: Transform
    offset: float


#: What a :class:`MetricRequest` is transformed by: a transform, with or without an offset.
TransformKey = Transform | OffsetTransform


def transform_label(key: TransformKey | None) -> str:
    """Return ``key`` as text (``''``, ``log``, ``log+0.01``), e.g. for state rows."""
    if key is None:
        return ""
    if isinstance(key, OffsetTransform):
        return f"{key.transform.value}+{key.offset!r}"
    return key.value


def parse_transform_label(label: str) -> TransformKey | None:
    """Return the transform key :func:`transform_label` gave ``label`` for."""
    if not label:
        return None
    name, plus, offset = label.partition("+")
    return OffsetTransform(Transform(name), float(offset)) if plus else Transform(name)


def base_transform(key: TransformKey | None) -> Transform | None:
    """Return the transform of ``key`` without its offset."""
    return key.transform if isinstance(key, OffsetTransform) else key


_TRANSFORMS: dict[Transform, Callable[[FloatArray], FloatArray]] = {
    Transform.LOG: np.log,
    Transform.SQRT: np.sqrt,
    Transform.SQUARE: np.square,
    Transform.CUBE: lambda x: np.power(x, 3),
    Transform.EXP: np.exp,
    Transform.RECIPROCAL: np.reciprocal,
    Transform.ABS: np.abs,
}


@dataclass(frozen=True)
class Params:
    """A metric's parameters (from the spec), e.g. a threshold or a normalizer."""

    threshold: float | None = None
    threshold_column: str | None = None
    threshold_quantile: float | None = None
    #: ``mean_obs`` | ``range_obs`` | ``value`` | ``column``
    normalize_by: str = "mean_obs"
    normalize_value: float | None = None
    normalize_column: str | None = None
    side: str = "obs"
    #: The month a year starts in; None = the metric's default (1, or 10 for timing).
    year_start_month: int | None = None
    #: Timing signatures: leave out a year with fewer than (1 - this) of its days.
    missing_day_threshold: float = 0.1
    exceedance: tuple[float, float] = (0.33, 0.66)
    #: flow_duration_curve_slope: the log form (True) or the linear slope.
    log: bool = True
    #: CRPS estimator: ``standard`` or ``fair``.
    estimator: str = "standard"
    #: Central prediction interval as two quantile levels.
    interval: tuple[float, float] = (0.05, 0.95)
    #: rank_frequency: the observation rank among ensemble members (0 = below all).
    rank: int = 0
    #: mase: the pair column holding each pair's naive scale.
    scale_column: str | None = None
    #: KGE family: weights on the correlation, variability and bias terms (sr, sa, sb).
    kge_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def columns(self) -> set[str]:
        """Pair columns these parameters read."""
        return {c for c in (self.threshold_column, self.normalize_column, self.scale_column) if c}


#: Spec names (pairing sides: primary, secondary) for the metrics' internal ones.
SPEC_SIDES: dict[str, str] = {"primary": "obs", "secondary": "pred"}
SPEC_NORMALIZERS: dict[str, str] = {"mean_primary": "mean_obs", "range_primary": "range_obs"}

Finalizer = Callable[[GroupStats, Params], FloatArray]


@dataclass(frozen=True)
class Metric:
    """A registered metric: a formula over components, and its definition.

    ``params`` names the spec options the metric accepts (``required`` must be given);
    ``needs`` names the pair columns it reads besides obs and pred (``value_time``,
    ``quantile``, or ``_case``, the forecast-case key the engine supplies); ``output``
    is the result type (``float``, ``int`` or ``timestamp``); ``skill`` marks error
    scores (lower is better, 0 is perfect) that skill scores can be computed from.
    """

    name: str
    finalize: Finalizer
    description: str
    family: str
    output: str = "float"
    params: frozenset[str] = frozenset()
    required: frozenset[str] = frozenset()
    needs: frozenset[str] = frozenset()
    transforms: bool = True
    skill: bool = False

    @property
    def needs_time(self) -> bool:
        """Whether the metric reads each pair's ``value_time``."""
        return "value_time" in self.needs

    @property
    def integer(self) -> bool:
        """Whether results are integer counts."""
        return self.output == "int"


def _div(num: FloatArray, den: FloatArray) -> FloatArray:
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den == 0, np.nan, out)


def _undefined(values: FloatArray, mask: npt.NDArray[np.bool_]) -> FloatArray:
    return np.where(mask, np.nan, values)


def _varies(s: GroupStats, side: str) -> npt.NDArray[np.bool_]:
    flag = s["obs_constant"] if side == "obs" else s["pred_constant"]
    result: npt.NDArray[np.bool_] = flag == 0
    return result


def _mae(s: GroupStats) -> FloatArray:
    return _div(s["sum_abs_error"], s["n"])


def _mse(s: GroupStats) -> FloatArray:
    return _div(s["sum_sq_error"], s["n"])


def _rmse(s: GroupStats) -> FloatArray:
    return np.sqrt(_mse(s))


def _nse(s: GroupStats) -> FloatArray:
    return _undefined(1.0 - _div(s["sum_sq_error"], s["ss_obs"]), ~_varies(s, "obs"))


def _both_vary(s: GroupStats) -> npt.NDArray[np.bool_]:
    return _varies(s, "obs") & _varies(s, "pred")


def _kge(s: GroupStats, p: Params, alpha: FloatArray, bias_term: FloatArray) -> FloatArray:
    """``1 - sqrt(sr (r-1)^2 + sa (alpha-1)^2 + sb bias_term)``; null if a side is constant."""
    sr, sa, sb = p.kge_weights
    r = s["pearson_r"]
    with np.errstate(invalid="ignore"):
        values = 1.0 - np.sqrt(sr * (r - 1) ** 2 + sa * (alpha - 1) ** 2 + sb * bias_term)
    return _undefined(values, ~_both_vary(s))


def _kge_2009(s: GroupStats, p: Params) -> FloatArray:
    alpha = _div(s["std_pred"], s["std_obs"])
    beta = _div(s["mean_pred"], s["mean_obs"])
    return _kge(s, p, alpha, (beta - 1) ** 2)


def _cv(s: GroupStats, side: str) -> FloatArray:
    return _div(s[f"std_{side}"], s[f"mean_{side}"])


def _kge_2012(s: GroupStats, p: Params) -> FloatArray:
    gamma = _div(_cv(s, "pred"), _cv(s, "obs"))
    beta = _div(s["mean_pred"], s["mean_obs"])
    return _kge(s, p, gamma, (beta - 1) ** 2)


def _kge_2021(s: GroupStats, p: Params) -> FloatArray:
    alpha = _div(s["std_pred"], s["std_obs"])
    bias = _div(s["mean_pred"] - s["mean_obs"], s["std_obs"])
    return _kge(s, p, alpha, bias**2)


_KGE_WEIGHTS = frozenset({"sr", "sa", "sb"})
_WEIGHTS_NOTE = " Weights sr, sa, sb (default 1) multiply the three terms."


def _m(
    name: str,
    finalize: Callable[[GroupStats], FloatArray],
    description: str,
    family: str,
    integer: bool = False,
) -> Metric:
    """Register a metric without parameters."""
    return Metric(
        name, lambda s, _p: finalize(s), description, family, "int" if integer else "float"
    )


# -- parameterized helpers ------------------------------------------------------------
def _threshold(s: GroupStats, p: Params) -> FloatArray | float:
    if p.threshold is not None:
        return p.threshold
    if p.threshold_column is not None:
        return s.column(p.threshold_column)
    q = p.threshold_quantile
    assert q is not None
    return s.per_pair(s.cached(f"quantile_obs|{q}", lambda: s.quantile("obs", q)))


_CELLS = ("hits", "false_alarms", "misses", "correct_negatives")


def _contingency(s: GroupStats, p: Params) -> dict[str, FloatArray]:
    key = f"contingency|{p.threshold}|{p.threshold_column}|{p.threshold_quantile}"

    def compute() -> FloatArray:
        thr = _threshold(s, p)
        obs_event, pred_event = s.obs >= thr, s.pred >= thr
        cells = [obs_event & pred_event, ~obs_event & pred_event, obs_event & ~pred_event]
        cells.append(~obs_event & ~pred_event)
        return np.stack([s.gsum(c.astype(np.float64)) for c in cells])

    if p.threshold_quantile is None:  # counts at fixed thresholds add up (state)
        hits, false_alarms, misses, correct_negatives = s.additive(
            tuple(f"{key}|{c}" for c in _CELLS), lambda: tuple(compute())
        )
    else:  # a quantile of the group's observations: needs every pair
        hits, false_alarms, misses, correct_negatives = s.cached(key, compute)
    return {
        "hits": hits,
        "false_alarms": false_alarms,
        "misses": misses,
        "correct_negatives": correct_negatives,
    }


def _cell(name: str) -> Finalizer:
    return lambda s, p: _contingency(s, p)[name]


def _score(formula: Callable[[dict[str, FloatArray]], FloatArray]) -> Finalizer:
    return lambda s, p: formula(_contingency(s, p))


def _normalizer(s: GroupStats, p: Params) -> FloatArray:
    if p.normalize_by == "range_obs":
        return s["max_obs"] - s["min_obs"]
    if p.normalize_by == "value":
        return np.full_like(s["n"], p.normalize_value or 0.0)
    if p.normalize_by == "column":
        assert p.normalize_column is not None
        column = p.normalize_column
        (total,) = s.additive((f"sum|{column}",), lambda: (s.gsum(s.column(column)),))
        return _div(total, s["n"])
    return s["mean_obs"]


def _year_index(seconds: FloatArray, start_month: int) -> npt.NDArray[np.int64]:
    months = seconds.astype("datetime64[s]").astype("datetime64[M]").astype(np.int64)
    return (months - (start_month - 1)) // 12


def _annual_peak_relative_bias(s: GroupStats, p: Params) -> FloatArray:
    def compute() -> FloatArray:
        years = _year_index(s.column("value_time"), p.year_start_month or 1)
        key = s.gid * 100_000 + (years - years.min() if years.size else years)
        uniq, sub = np.unique(key, return_inverse=True)
        parent = (uniq // 100_000).astype(np.int64)
        fine = s.regroup(sub.astype(np.int64), uniq.size)
        peaks_obs, peaks_pred = fine.gmax(fine.obs), fine.gmax(fine.pred)
        return _div(s.to_parent(peaks_pred - peaks_obs, parent), s.to_parent(peaks_obs, parent))

    return s.cached(f"annual_peak_relative_bias|{p.year_start_month or 1}", compute)


def _fdc_slope(s: GroupStats, p: Params) -> FloatArray:
    lo, hi = p.exceedance
    q_lo, q_hi = s.sketch_quantile(p.side, 1 - lo), s.sketch_quantile(p.side, 1 - hi)
    if not p.log:
        return np.asarray((q_lo - q_hi) / (hi - lo), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (np.log(q_lo) - np.log(q_hi)) / (hi - lo)
    return np.where((q_lo > 0) & (q_hi > 0), slope, np.nan)


_THRESHOLD = frozenset({"threshold"})
_TIME = frozenset({"value_time"})
_SIDE = frozenset({"side"})


def _contingency_metric(name: str, finalize: Finalizer, description: str, integer: bool) -> Metric:
    return Metric(
        name,
        finalize,
        description + " An event is a value >= threshold.",
        "contingency",
        "int" if integer else "float",
        params=_THRESHOLD,
        required=_THRESHOLD,
        transforms=False,
    )


def _signature(name: str, finalize: Finalizer, description: str, **kw: object) -> Metric:
    return Metric(
        name,
        finalize,
        description + " Of obs or pred (side, default obs); column <name>_<side>.",
        "signature",
        **kw,  # type: ignore[arg-type]
    )


def _brier(s: GroupStats, p: Params) -> FloatArray:
    key = f"brier|{p.threshold}|{p.threshold_column}|{p.threshold_quantile}"
    if p.threshold_quantile is None:
        return s.case_mean(key, lambda: _brier_cases(s, p))
    # A quantile of the group's observations: needs every pair.
    return s.cached(key, lambda: s.mean_over_cases(_brier_cases(s, p)))


def _brier_cases(s: GroupStats, p: Params) -> FloatArray:
    def compute() -> FloatArray:
        ids, sizes = s.cases()
        thr = _threshold(s, p)  # per replicate for an obs quantile
        shape = s.weights.shape
        pred_event = np.broadcast_to(s.pred >= thr, shape).astype(np.float64)
        obs_event = np.broadcast_to(s.obs >= thr, shape).astype(np.float64)

        def per_case(x: FloatArray) -> FloatArray:
            rows = [np.bincount(ids, weights=r, minlength=sizes.size) for r in x]
            return np.vstack(rows) / sizes

        return (per_case(pred_event) - (per_case(obs_event) > 0.5)) ** 2

    return s.cached(f"brier|{p.threshold}|{p.threshold_column}|{p.threshold_quantile}", compute)


def _interval_cases(s: GroupStats, p: Params) -> dict[str, FloatArray]:
    lo, hi = p.interval

    def compute() -> FloatArray:
        ids, sizes = s.cases()
        level = s.column("quantile")
        n_cases = sizes.size
        at_lo, at_hi = np.isclose(level, lo), np.isclose(level, hi)
        low = np.bincount(ids, weights=np.where(at_lo, s.pred, 0.0), minlength=n_cases)
        high = np.bincount(ids, weights=np.where(at_hi, s.pred, 0.0), minlength=n_cases)
        valid = (np.bincount(ids, weights=at_lo, minlength=n_cases) == 1) & (
            np.bincount(ids, weights=at_hi, minlength=n_cases) == 1
        )
        y = s.case_obs()
        alpha = 1.0 - (hi - lo)
        covered = ((low <= y) & (y <= high)).astype(np.float64)
        width = high - low
        winkler = (
            width + (2 / alpha) * np.maximum(low - y, 0) + (2 / alpha) * np.maximum(y - high, 0)
        )
        out = np.stack([covered, width, winkler])
        out[:, ~valid] = np.nan
        return out

    covered, width, winkler = s.cached(f"interval|{lo}|{hi}", compute)
    return {"coverage": covered, "width": width, "winkler": winkler}


def _interval_mean(s: GroupStats, p: Params, part: str) -> FloatArray:
    lo, hi = p.interval
    return s.case_mean(f"interval|{lo}|{hi}|{part}", lambda: _interval_cases(s, p)[part])


def _pinball(s: GroupStats) -> FloatArray:
    def sums() -> tuple[FloatArray, FloatArray]:
        q = s.column("quantile")
        err = s.obs - s.pred
        loss = np.maximum(q * err, (q - 1) * err)
        valid = np.isfinite(loss)
        return s.gsum(np.where(valid, loss, 0.0)), s.gsum(valid.astype(np.float64))

    return _div(*s.additive(("pinball|sum", "pinball|n"), sums))


_CASE = frozenset({"_case"})
_QUANTILE_CASE = frozenset({"quantile", "_case"})

_PROBABILISTIC = (
    Metric(
        "mean_crps_ensemble",
        lambda s, p: s.case_mean(
            f"crps|{p.estimator}", lambda: s.crps_cases(p.estimator == "fair")
        ),
        "Continuous ranked probability score of ensemble forecasts, averaged over "
        "forecast cases: mean|x - y| - sum|x_i - x_j| / (2 m^2) over a case's m members "
        "(estimator: standard, or fair with 2 m (m - 1)). Equals MAE for one member.",
        "probabilistic",
        params=frozenset({"estimator"}),
        needs=_CASE,
        skill=True,
    ),
    Metric(
        "mean_brier_score",
        _brier,
        "Brier score of the event value >= threshold: mean over cases of (p - o)^2, p = "
        "fraction of members predicting the event, o = 1 if observed.",
        "probabilistic",
        params=_THRESHOLD,
        required=_THRESHOLD,
        needs=_CASE,
        transforms=False,
        skill=True,
    ),
    Metric(
        "mean_pinball_loss",
        lambda s, p: _pinball(s),
        "Mean pinball (quantile) loss over quantile forecasts: q(y - x) if y >= x, else "
        "(1 - q)(x - y), q = the pair's quantile level.",
        "probabilistic",
        needs=frozenset({"quantile"}),
        skill=True,
    ),
    Metric(
        "interval_coverage",
        lambda s, p: _interval_mean(s, p, "coverage"),
        "Fraction of cases whose observation falls within the central interval (interval "
        "[lo, hi] quantile levels, default [0.05, 0.95]).",
        "probabilistic",
        params=frozenset({"interval"}),
        needs=_QUANTILE_CASE,
    ),
    Metric(
        "mean_interval_width",
        lambda s, p: _interval_mean(s, p, "width"),
        "Mean width of the central interval (hi - lo quantile values).",
        "probabilistic",
        params=frozenset({"interval"}),
        needs=_QUANTILE_CASE,
    ),
    Metric(
        "mean_interval_score",
        lambda s, p: _interval_mean(s, p, "winkler"),
        "Interval (Winkler) score: width + (2/alpha)(lo - y) if y < lo, + (2/alpha)(y - hi) "
        "if y > hi, alpha = 1 - (hi - lo); mean over cases.",
        "probabilistic",
        params=frozenset({"interval"}),
        needs=_QUANTILE_CASE,
        skill=True,
    ),
    Metric(
        "ensemble_spread",
        lambda s, p: np.sqrt(s.case_mean("ensemble_variance", lambda: s.case_moments()[1])),
        "Ensemble spread: sqrt(mean over cases of the members' variance), m - 1 "
        "denominator; cases with one member are left out.",
        "probabilistic",
        needs=_CASE,
        transforms=False,
    ),
    Metric(
        "ensemble_mean_root_mean_square_error",
        lambda s, p: _ensemble_mean_rmse(s),
        "RMSE of the ensemble mean: sqrt(mean over cases of (mean of members - obs)^2).",
        "probabilistic",
        needs=_CASE,
        transforms=False,
        skill=True,
    ),
    Metric(
        "spread_skill_ratio",
        lambda s, p: _spread_skill(s),
        "Spread-skill ratio: sqrt(mean over cases of (m + 1) / m x member variance) / RMSE "
        "of the ensemble mean (Fortin et al. 2014); about 1 for a reliable ensemble, "
        "below 1 under-dispersed.",
        "probabilistic",
        needs=_CASE,
        transforms=False,
    ),
    Metric(
        "rank_frequency",
        lambda s, p: s.case_mean(
            f"rank|{p.rank}", lambda: (s.case_ranks()[0] == p.rank).astype(float)
        ),
        "Share of cases whose observation has rank `rank` among the members (0: below "
        "all, m: above all; ties count half, rounded down). One per rank gives the rank "
        "histogram.",
        "probabilistic",
        params=frozenset({"rank"}),
        required=frozenset({"rank"}),
        needs=_CASE,
        transforms=False,
    ),
    Metric(
        "rank_outlier_fraction",
        lambda s, p: _rank_outliers(s),
        "Share of cases whose observation is outside the ensemble's range (rank 0 or m); "
        "2 / (m + 1) for a reliable ensemble.",
        "probabilistic",
        needs=_CASE,
        transforms=False,
    ),
    Metric(
        "rank_histogram_reliability",
        lambda s, p: _rank_reliability(s),
        "Reliability index of the rank histogram: sum over ranks of |frequency - 1 / "
        "(m + 1)| (Delle Monache et al. 2006); 0 is flat. Null when a group's cases have "
        "different ensemble sizes.",
        "probabilistic",
        needs=_CASE,
        transforms=False,
    ),
)


def _mase(s: GroupStats, p: Params) -> FloatArray:
    """Mean of |pred - obs| / scale over pairs with a positive scale."""
    assert p.scale_column is not None
    column = p.scale_column

    def sums() -> tuple[FloatArray, FloatArray]:
        scale = s.column(column)
        valid = np.isfinite(scale) & (scale > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(valid, np.abs(s.pred - s.obs) / np.where(valid, scale, 1.0), 0.0)
        return s.gsum(ratio), s.gsum(valid.astype(float))

    return _div(*s.additive((f"mase|{column}|sum", f"mase|{column}|n"), sums))


def _ensemble_mean_rmse(s: GroupStats) -> FloatArray:
    def squared_error() -> FloatArray:
        mean, _, _ = s.case_moments()
        return (mean - s.case_obs()) ** 2

    return np.sqrt(s.case_mean("ensemble_mean_sq_error", squared_error))


def _spread_skill(s: GroupStats) -> FloatArray:
    def adjusted_variance() -> FloatArray:
        _, var, sizes = s.case_moments()
        m = sizes.astype(float)
        out: FloatArray = (m + 1) / m * var
        return out

    spread = np.sqrt(s.case_mean("ensemble_variance_adjusted", adjusted_variance))
    return _div(spread, _ensemble_mean_rmse(s))


def _rank_outliers(s: GroupStats) -> FloatArray:
    def outside() -> FloatArray:
        rank, sizes = s.case_ranks()
        out: FloatArray = ((rank == 0) | (rank == sizes)).astype(float)
        return out

    return s.case_mean("rank_outlier", outside)


def _rank_reliability(s: GroupStats) -> FloatArray:
    # Its ranks run to the largest ensemble in the data: not a fixed set of sums.
    rank, sizes = s.case_ranks()
    m = sizes.astype(float)
    mean_m, mean_m2 = s.mean_over_cases(m), s.mean_over_cases(m * m)
    uniform = np.isclose(mean_m2, mean_m**2)
    total = np.zeros_like(mean_m)
    for k in range(int(sizes.max(initial=0)) + 1):
        freq = s.mean_over_cases((rank == k).astype(float))
        with np.errstate(invalid="ignore"):
            total += np.where(k <= mean_m, np.abs(freq - 1 / (mean_m + 1)), 0.0)
    return np.where(uniform, total, np.nan)


_PARAMETERIZED = (
    _contingency_metric("hits", _cell("hits"), "Pairs where obs and pred are events.", True),
    _contingency_metric(
        "misses", _cell("misses"), "Pairs where obs is an event and pred isn't.", True
    ),
    _contingency_metric(
        "false_alarms", _cell("false_alarms"), "Pairs where pred is an event and obs isn't.", True
    ),
    _contingency_metric(
        "correct_negatives", _cell("correct_negatives"), "Pairs where neither is an event.", True
    ),
    _contingency_metric(
        "probability_of_detection",
        _score(lambda t: _div(t["hits"], t["hits"] + t["misses"])),
        "Probability of detection: hits / (hits + misses); null without observed events.",
        False,
    ),
    _contingency_metric(
        "probability_of_false_detection",
        _score(lambda t: _div(t["false_alarms"], t["false_alarms"] + t["correct_negatives"])),
        "Probability of false detection: false_alarms / (false_alarms + correct_negatives).",
        False,
    ),
    _contingency_metric(
        "false_alarm_ratio",
        _score(lambda t: _div(t["false_alarms"], t["hits"] + t["false_alarms"])),
        "False alarm ratio: false_alarms / (hits + false_alarms); null without predicted events.",
        False,
    ),
    _contingency_metric(
        "critical_success_index",
        _score(lambda t: _div(t["hits"], t["hits"] + t["misses"] + t["false_alarms"])),
        "Critical success index (threat score): hits / (hits + misses + false_alarms).",
        False,
    ),
    _contingency_metric(
        "success_ratio",
        _score(lambda t: _div(t["hits"], t["hits"] + t["false_alarms"])),
        "Success ratio: hits / (hits + false_alarms) = 1 - FAR.",
        False,
    ),
    _contingency_metric(
        "proportion_correct",
        _score(
            lambda t: _div(
                t["hits"] + t["correct_negatives"],
                t["hits"] + t["false_alarms"] + t["misses"] + t["correct_negatives"],
            )
        ),
        "Proportion correct (accuracy): (hits + correct_negatives) / count.",
        False,
    ),
    _contingency_metric(
        "frequency_bias_index",
        _score(lambda t: _div(t["hits"] + t["false_alarms"], t["hits"] + t["misses"])),
        "Frequency bias: (hits + false_alarms) / (hits + misses); 1 = as many predicted "
        "as observed events.",
        False,
    ),
    Metric(
        "max_value_time_delta",
        lambda s, p: s.time_of_max("pred") - s.time_of_max("obs"),
        "Seconds from the time of max(obs) to the time of max(pred) (earliest time of each "
        "maximum); positive means the predicted peak is later.",
        "timing",
        needs=_TIME,
    ),
    Metric(
        "annual_peak_flow_bias",
        _annual_peak_relative_bias,
        "sum over years of (annual max(pred) - annual max(obs)) / sum of annual max(obs). "
        "Years start in year_start_month (default 1, calendar years; 10 for water years).",
        "timing",
        params=frozenset({"year_start_month"}),
        needs=_TIME,
    ),
    _signature("minimum", lambda s, p: s[f"min_{p.side}"], "Minimum.", params=_SIDE),
    _signature("maximum", lambda s, p: s[f"max_{p.side}"], "Maximum.", params=_SIDE),
    _signature("average", lambda s, p: s[f"mean_{p.side}"], "Mean.", params=_SIDE),
    _signature("sum", lambda s, p: s[f"sum_{p.side}"], "Sum.", params=_SIDE),
    _signature(
        "variance",
        lambda s, p: _div(s[f"ss_{p.side}"], s["n"]),
        "Population variance.",
        params=_SIDE,
    ),
    _signature(
        "max_value_time",
        lambda s, p: s.time_of_max(p.side),
        "Time of the maximum (earliest if repeated).",
        params=_SIDE,
        needs=_TIME,
        output="timestamp",
    ),
    _signature(
        "center_of_timing",
        lambda s, p: timing(
            s, p.side, p.year_start_month or 10, p.missing_day_threshold, spread=False
        ),
        "Center of timing: the flow-weighted mean day of the year (day 1 = the first day "
        "of year_start_month, default 10), from daily means, averaged over the years; a "
        "year with fewer than 1 - missing_day_threshold (default 0.1) of its days, and "
        "days without flow, are left out. Needs the pairs.",
        params=frozenset({"side", "year_start_month", "missing_day_threshold"}),
        needs=_TIME,
    ),
    _signature(
        "standard_deviation_of_timing",
        lambda s, p: timing(
            s, p.side, p.year_start_month or 10, p.missing_day_threshold, spread=True
        ),
        "Standard deviation of timing: the flow-weighted spread of the days around the "
        "center of timing, sqrt(sum q (d - CT)^2 / (sum q (n - 1) / n)) over the n days "
        "with flow, per year, averaged over the years (as center_of_timing).",
        params=frozenset({"side", "year_start_month", "missing_day_threshold"}),
        needs=_TIME,
    ),
    _signature(
        "flow_duration_curve_slope",
        _fdc_slope,
        "Flow duration curve slope: (ln Q(e1) - ln Q(e2)) / (e2 - e1), Q(e) = value "
        "exceeded a fraction e of the time, exceedance [e1, e2] default [0.33, 0.66]; "
        "null if either value <= 0. With log false, the linear slope (Q(e1) - Q(e2)) / "
        "(e2 - e1). Quantiles come from quantile sketches (within 0.1%), so the metric "
        "merges from state.",
        params=frozenset({"side", "exceedance", "log"}),
    ),
)


METRICS: dict[str, Metric] = {
    m.name: m
    for m in (
        _m("count", lambda s: s["n"], "Number of pairs.", "count", integer=True),
        # -- errors ------------------------------------------------------------------
        _m(
            "mean_error",
            lambda s: _div(s["sum_error"], s["n"]),
            "mean(pred - obs).",
            "error",
        ),
        _m("mean_absolute_error", _mae, "mean(|pred - obs|).", "error"),
        _m("mean_square_error", _mse, "mean((pred - obs)^2).", "error"),
        _m(
            "root_mean_square_error",
            _rmse,
            "Root mean square error: sqrt(mean((pred - obs)^2)).",
            "error",
        ),
        _m(
            "max_value_delta",
            lambda s: s["max_pred"] - s["max_obs"],
            "max(pred) - max(obs).",
            "error",
        ),
        # -- relative ----------------------------------------------------------------
        _m(
            "relative_bias",
            lambda s: _div(s["sum_error"], s["sum_obs"]),
            "sum(pred - obs) / sum(obs); null if sum(obs) == 0.",
            "relative",
        ),
        _m(
            "mean_absolute_relative_error",
            lambda s: _div(s["sum_abs_error"], s["sum_obs"]),
            "sum(|pred - obs|) / sum(obs); null if sum(obs) == 0.",
            "relative",
        ),
        _m(
            "multiplicative_bias",
            lambda s: _div(s["mean_pred"], s["mean_obs"]),
            "mean(pred) / mean(obs); null if mean(obs) == 0.",
            "relative",
        ),
        _m(
            "relative_median",
            lambda s: _div(s.sketch_quantile("pred", 0.5), s.sketch_quantile("obs", 0.5)),
            "median(pred) / median(obs); null if median(obs) == 0. Medians come from "
            "quantile sketches (within 0.1%), so the metric merges from state.",
            "relative",
        ),
        _m(
            "relative_minimum",
            lambda s: _div(s["min_pred"], s["min_obs"]),
            "min(pred) / min(obs); null if min(obs) == 0.",
            "relative",
        ),
        _m(
            "relative_maximum",
            lambda s: _div(s["max_pred"], s["max_obs"]),
            "max(pred) / max(obs); null if max(obs) == 0.",
            "relative",
        ),
        _m(
            "relative_standard_deviation",
            lambda s: _undefined(_div(s["std_pred"], s["std_obs"]), ~_varies(s, "obs")),
            "std(pred) / std(obs) (population std); null if obs is constant.",
            "relative",
        ),
        _m(
            "mean_absolute_percentage_error",
            lambda s: 100 * _div(s["sum_abs_pct_error"], s["n_obs_nonzero"]),
            "Mean absolute percentage error: 100 * mean(|pred - obs| / |obs|) over pairs "
            "with obs != 0; null if there are none.",
            "relative",
        ),
        _m(
            "symmetric_mean_absolute_percentage_error",
            lambda s: 100 * _div(s["sum_sym_pct_error"], s["n"]),
            "Symmetric MAPE: 100 * mean(2|pred - obs| / (|obs| + |pred|)); a pair where "
            "both are 0 counts as 0 error. Ranges 0 to 200.",
            "relative",
        ),
        Metric(
            "normalized_mean_absolute_error",
            lambda s, p: _div(_mae(s), _normalizer(s, p)),
            "MAE / normalizer: mean(obs) by default; normalize_by range_obs (max - min), "
            "a number (e.g. installed capacity) or a column. Null if the normalizer is 0.",
            "relative",
            params=frozenset({"normalize_by"}),
        ),
        Metric(
            "normalized_root_mean_square_error",
            lambda s, p: _div(_rmse(s), _normalizer(s, p)),
            "RMSE / normalizer (as nmae).",
            "relative",
            params=frozenset({"normalize_by"}),
        ),
        Metric(
            "mean_absolute_scaled_error",
            _mase,
            "Mean absolute scaled error: mean of |pred - obs| / scale, scale being each "
            "pair's in-sample naive error (a naive_scale field: its series' mean "
            "|y(t) - y(t - lag)| over a training period); < 1 beats that naive forecast's "
            "typical error. Pairs without a positive scale are left out.",
            "relative",
            params=frozenset({"scale"}),
            required=frozenset({"scale"}),
            skill=True,
        ),
        # -- correlation and variability -------------------------------------------------
        _m(
            "pearson_correlation",
            lambda s: s["pearson_r"],
            "Pearson correlation; null if obs or pred is constant.",
            "correlation",
        ),
        _m(
            "spearman_correlation",
            lambda s: s["spearman_r"],
            "Spearman rank correlation (ties get average ranks); null if obs or pred is constant.",
            "correlation",
        ),
        _m(
            "r_squared",
            lambda s: s["pearson_r"] ** 2,
            "Coefficient of determination as the squared Pearson correlation.",
            "correlation",
        ),
        _m(
            "coefficient_of_variation_ratio",
            lambda s: _undefined(_div(_cv(s, "pred"), _cv(s, "obs")), ~_varies(s, "obs")),
            "Ratio of coefficients of variation: (std(pred)/mean(pred)) / "
            "(std(obs)/mean(obs)); null if obs is constant or a mean is 0.",
            "correlation",
        ),
        _m(
            "root_mean_standard_deviation_ratio",
            lambda s: _undefined(_div(_rmse(s), s["std_obs"]), ~_varies(s, "obs")),
            "RMSE-observations standard deviation ratio: RMSE / std(obs); null if obs is constant.",
            "correlation",
        ),
        # -- efficiency ----------------------------------------------------------------
        _m(
            "nash_sutcliffe_efficiency",
            _nse,
            "Nash-Sutcliffe efficiency: 1 - sum((pred - obs)^2) / sum((obs - mean(obs))^2); "
            "null if obs is constant (within rounding).",
            "efficiency",
        ),
        _m(
            "nash_sutcliffe_efficiency_normalized",
            lambda s: _div(np.ones_like(s["n"]), 2.0 - _nse(s)),
            "Normalized NSE: 1 / (2 - NSE), in (0, 1]; null if obs is constant.",
            "efficiency",
        ),
        Metric(
            "kling_gupta_efficiency",
            _kge_2009,
            "Kling-Gupta efficiency (Gupta et al. 2009): 1 - sqrt((r-1)^2 + (alpha-1)^2 + "
            "(beta-1)^2), alpha = std(pred)/std(obs), beta = mean(pred)/mean(obs); null if "
            "obs or pred is constant or mean(obs) == 0." + _WEIGHTS_NOTE,
            "efficiency",
            params=_KGE_WEIGHTS,
        ),
        Metric(
            "kling_gupta_efficiency_mod1",
            _kge_2012,
            "KGE' (Kling et al. 2012): as KGE with alpha replaced by the ratio of "
            "coefficients of variation (pred/obs)." + _WEIGHTS_NOTE,
            "efficiency",
            params=_KGE_WEIGHTS,
        ),
        Metric(
            "kling_gupta_efficiency_mod2",
            _kge_2021,
            "KGE'' (Tang et al. 2021): 1 - sqrt((r-1)^2 + (alpha-1)^2 + beta^2), alpha = "
            "std(pred)/std(obs), beta = (mean(pred) - mean(obs)) / std(obs)." + _WEIGHTS_NOTE,
            "efficiency",
            params=_KGE_WEIGHTS,
        ),
    )
}
METRICS.update({m.name: m for m in (*_PARAMETERIZED, *_PROBABILISTIC)})
for _name in (
    "mean_absolute_error",
    "mean_square_error",
    "root_mean_square_error",
    "mean_absolute_relative_error",
    "mean_absolute_percentage_error",
    "symmetric_mean_absolute_percentage_error",
    "normalized_mean_absolute_error",
    "normalized_root_mean_square_error",
):
    METRICS[_name] = replace(METRICS[_name], skill=True)


#: Frozen short names, each for exactly one metric (with its default parameters).
#: An alias is accepted wherever a spec names a metric; results carry the canonical
#: name. Never repurpose or remove one (the shared metric standard's rule).
ALIASES: dict[str, str] = {
    "annual_peak_relative_bias": "annual_peak_flow_bias",
    "mean_value": "average",
    "csi": "critical_success_index",
    "accuracy": "proportion_correct",
    "ensemble_mean_rmse": "ensemble_mean_root_mean_square_error",
    "far": "false_alarm_ratio",
    "fdc_slope": "flow_duration_curve_slope",
    "frequency_bias": "frequency_bias_index",
    "kge": "kling_gupta_efficiency",
    "kge_2012": "kling_gupta_efficiency_mod1",
    "kge_mod1": "kling_gupta_efficiency_mod1",
    "kge_2021": "kling_gupta_efficiency_mod2",
    "kge_mod2": "kling_gupta_efficiency_mod2",
    "max_value": "maximum",
    "mae": "mean_absolute_error",
    "mape": "mean_absolute_percentage_error",
    "mase": "mean_absolute_scaled_error",
    "brier_score": "mean_brier_score",
    "crps": "mean_crps_ensemble",
    "me": "mean_error",
    "winkler_score": "mean_interval_score",
    "interval_width": "mean_interval_width",
    "pinball_loss": "mean_pinball_loss",
    "mean_squared_error": "mean_square_error",
    "mse": "mean_square_error",
    "min_value": "minimum",
    "multbias": "multiplicative_bias",
    "relative_mean": "multiplicative_bias",
    "relmean": "multiplicative_bias",
    "nse": "nash_sutcliffe_efficiency",
    "nnse": "nash_sutcliffe_efficiency_normalized",
    "nmae": "normalized_mean_absolute_error",
    "nrmse": "normalized_root_mean_square_error",
    "pearson_r": "pearson_correlation",
    "pod": "probability_of_detection",
    "pofd": "probability_of_false_detection",
    "relbias": "relative_bias",
    "relative_max": "relative_maximum",
    "relative_min": "relative_minimum",
    "relative_std": "relative_standard_deviation",
    "relstd": "relative_standard_deviation",
    "variability_ratio": "relative_standard_deviation",
    "vr": "relative_standard_deviation",
    "rmse": "root_mean_square_error",
    "rsr": "root_mean_standard_deviation_ratio",
    "spearman_r": "spearman_correlation",
    "sum_value": "sum",
    "smape": "symmetric_mean_absolute_percentage_error",
}
assert not set(ALIASES) & set(METRICS), "an alias can't also be a metric's name"
assert set(ALIASES.values()) <= set(METRICS), "an alias must name a registered metric"


def canonical_name(name: str) -> str:
    """Return a metric's canonical name, given it or an alias (unknown names unchanged)."""
    return ALIASES.get(name, name)


def aliases_of(name: str) -> list[str]:
    """Return a metric's aliases."""
    return sorted(a for a, c in ALIASES.items() if c == name)


def metric_names() -> list[str]:
    """Return registered metric names (canonical)."""
    return list(METRICS)


@dataclass(frozen=True)
class MetricRequest:
    """One metric to compute in a stage, optionally on transformed values."""

    name: str
    transform: TransformKey | None = None
    params: Params = Params()

    def columns(self) -> set[str]:
        """Pair columns (besides obs and pred) this metric reads."""
        return set(METRICS[self.name].needs) | self.params.columns()


def transformed(
    obs: FloatArray, pred: FloatArray, transform: TransformKey | None
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.bool_]]:
    """Return ``obs`` and ``pred`` after ``transform``, and which pairs stay (both finite)."""
    keep: npt.NDArray[np.bool_] = np.isfinite(obs) & np.isfinite(pred)
    if transform is not None:
        if isinstance(transform, OffsetTransform):
            obs, pred = obs + transform.offset, pred + transform.offset
        fn = _TRANSFORMS[base_transform(transform)]  # type: ignore[index]
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            obs, pred = fn(obs), fn(pred)
        keep &= np.isfinite(obs) & np.isfinite(pred)
    return obs, pred, keep


def _stats(
    obs: FloatArray,
    pred: FloatArray,
    gid: IntArray,
    n_groups: int,
    transform: TransformKey | None,
    weights: FloatArray | None,
    columns: dict[str, FloatArray] | None,
) -> GroupStats:
    obs, pred, keep = transformed(obs, pred, transform)
    if not keep.all():
        obs, pred, gid = obs[keep], pred[keep], gid[keep]
        weights = None if weights is None else weights[:, keep]
        columns = None if columns is None else {k: v[keep] for k, v in columns.items()}
    return GroupStats(obs, pred, gid, n_groups, weights, columns)


def compute_metrics(
    requests: Sequence[MetricRequest],
    obs: FloatArray,
    pred: FloatArray,
    gid: IntArray,
    n_groups: int,
    weights: FloatArray | None = None,
    columns: dict[str, FloatArray] | None = None,
) -> dict[MetricRequest, FloatArray]:
    """Compute several metrics per group, sharing components between them.

    One :class:`GroupStats` is built per transform; every metric on it reuses the
    components already computed. Pairs with a non-finite obs or pred are dropped (per
    transform, so a ``log`` metric drops non-positive values only for itself).

    ``columns`` holds other per-pair values metrics need (:meth:`MetricRequest.columns`),
    e.g. ``value_time`` as epoch seconds.

    Returns ``(replicates, groups)`` arrays (replicates is 1 without ``weights``); NaN
    marks undefined results. Timestamps are epoch seconds.
    """
    out: dict[MetricRequest, FloatArray] = {}
    by_transform: dict[TransformKey | None, list[MetricRequest]] = {}
    for req in requests:
        by_transform.setdefault(req.transform, []).append(req)
    for transform, reqs in by_transform.items():
        stats = _stats(obs, pred, gid, n_groups, transform, weights, columns)
        for req in reqs:
            out[req] = METRICS[req.name].finalize(stats, req.params)
    return out


def compute_grouped(
    name: str,
    obs: FloatArray,
    sim: FloatArray,
    gid: IntArray,
    n_groups: int,
    transform: TransformKey | None = None,
) -> FloatArray | IntArray:
    """Compute metric ``name`` for each group (one value per group; NaN = undefined)."""
    req = MetricRequest(name, transform)
    values: FloatArray = compute_metrics([req], obs, sim, gid, n_groups)[req][0]
    if METRICS[name].integer:
        counts: IntArray = values.astype(np.int64)
        return counts
    return values
