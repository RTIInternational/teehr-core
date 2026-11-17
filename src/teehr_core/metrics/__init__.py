"""Metrics on paired observed and predicted values, with no service needed.

Ported from VeriRace (``verirace/metrics/__init__.py`` at 668ab7d).

Each function takes two equal-length arrays (lists, NumPy or Arrow) and returns one
number; :func:`evaluate` computes several at once, sharing the work between them. They
are the same formulas the VeriRace engine runs. Pairs where either value is not finite
are dropped first. A metric that is undefined for the input returns ``None``.

Contingency metrics take a ``threshold`` (an event is a value >= threshold), timing
metrics take ``times`` (datetimes, ``datetime64`` or epoch seconds), and signatures take
one series.

Examples
--------
>>> from teehr_core import metrics
>>> metrics.nse([1, 2, 3, 4, 5], [1.5, 1.5, 3.5, 3, 6])
0.725
>>> result = metrics.evaluate([1, 2, 3, 4, 5], [1.5, 1.5, 3.5, 3, 6], ["rmse", "kge"])
>>> result["root_mean_square_error"]  # results carry canonical names; rmse is an alias
0.7416198487095663
>>> metrics.csi([1, 5, 2, 8], [2, 6, 7, 1], threshold=5)
0.3333333333333333
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt

from teehr_core.metrics.registry import (
    METRICS,
    SPEC_NORMALIZERS,
    SPEC_SIDES,
    Metric,
    MetricRequest,
    Params,
    Transform,
    canonical_name,
    compute_grouped,
    compute_metrics,
    metric_names,
)

ArrayLike = Sequence[float] | npt.NDArray[np.floating] | npt.NDArray[np.integer]
TimesLike = Sequence[Any] | npt.NDArray[Any]
TransformLike = Transform | str | None
Value = float | int | datetime | None


def _arrays(
    obs: ArrayLike, pred: ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    o = np.asarray(obs, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if o.shape != p.shape or o.ndim != 1:
        raise ValueError(
            f"obs and pred must be 1-D arrays of equal length; got {o.shape}, {p.shape}"
        )
    return o, p, np.zeros(o.size, dtype=np.int64)


def _seconds(times: TimesLike) -> npt.NDArray[np.float64]:
    arr = np.asarray(times)
    if arr.dtype.kind == "U":  # ISO strings, possibly with an offset
        arr = np.array([datetime.fromisoformat(x) for x in arr], dtype=object)
    if arr.dtype.kind in "OM":
        if arr.dtype.kind == "O":
            arr = np.array(
                [t.astimezone(UTC).replace(tzinfo=None) if t.tzinfo else t for t in arr],
                dtype="datetime64[us]",
            )
        micros = arr.astype("datetime64[us]").astype(np.int64)
        return micros.astype(np.float64) / 1e6
    return arr.astype(np.float64)


def _value(name: str, value: float) -> Value:
    if np.isnan(value):
        return None
    output = METRICS[name].output
    if output == "int":
        return int(value)
    if output == "timestamp":
        return datetime.fromtimestamp(value, UTC)
    return float(value)


def evaluate(
    obs: ArrayLike,
    pred: ArrayLike,
    names: Iterable[str],
    *,
    transform: TransformLike = None,
    times: TimesLike | None = None,
    threshold: float | ArrayLike | dict[str, float] | None = None,
    normalize_by: str | float | ArrayLike | None = None,
    side: str | None = None,
    year_start_month: int | None = None,
    exceedance: tuple[float, float] | None = None,
    estimator: str | None = None,
    interval: tuple[float, float] | None = None,
    cases: ArrayLike | None = None,
    quantiles: float | ArrayLike | None = None,
    scale: float | ArrayLike | None = None,
    kge_weights: tuple[float, float, float] | None = None,
) -> dict[str, Value]:
    """Compute several metrics at once, sharing intermediate results between them.

    Options (``threshold``, ``normalize_by``, ...) apply to every metric that takes them;
    an array ``threshold`` or ``normalize_by`` gives one value per pair, and
    ``threshold={"primary_quantile": 0.9}`` uses a quantile of the observations.
    Probabilistic metrics read long-format data: ``cases`` gives each pair's forecast
    case (pairs of one ensemble or quantile forecast share it) and ``quantiles`` each
    pair's quantile level. ``scale`` is MASE's in-sample naive scale: one number, or one
    per pair. ``kge_weights`` are the KGE family's ``(sr, sa, sb)``.
    """
    o, p, gid = _arrays(obs, pred)
    t = Transform(transform) if transform is not None else None
    columns: dict[str, npt.NDArray[np.float64]] = {}
    params: dict[str, Any] = {}
    if times is not None:
        columns["value_time"] = _seconds(times)
    if isinstance(threshold, dict):
        params["threshold_quantile"] = float(threshold["primary_quantile"])
    elif threshold is not None:
        if np.ndim(threshold) == 0:
            params["threshold"] = float(threshold)  # type: ignore[arg-type]
        else:
            columns["_threshold"] = np.asarray(threshold, dtype=np.float64)
            params["threshold_column"] = "_threshold"
    if normalize_by is not None:
        if isinstance(normalize_by, str):
            params["normalize_by"] = SPEC_NORMALIZERS.get(normalize_by, normalize_by)
        elif np.ndim(normalize_by) == 0:
            params["normalize_by"], params["normalize_value"] = "value", float(normalize_by)  # type: ignore[arg-type]
        else:
            columns["_normalizer"] = np.asarray(normalize_by, dtype=np.float64)
            params["normalize_by"], params["normalize_column"] = "column", "_normalizer"
    if side is not None:
        params["side"] = SPEC_SIDES.get(side, side)
    if year_start_month is not None:
        params["year_start_month"] = year_start_month
    if exceedance is not None:
        params["exceedance"] = tuple(exceedance)
    if estimator is not None:
        params["estimator"] = estimator
    if interval is not None:
        params["interval"] = tuple(interval)
    if cases is not None:
        columns["_case"] = np.asarray(cases, dtype=np.float64)
    if quantiles is not None:
        columns["quantile"] = np.broadcast_to(np.asarray(quantiles, dtype=np.float64), o.shape)
    if kge_weights is not None:
        params["kge_weights"] = tuple(float(w) for w in kge_weights)
    if scale is not None:
        columns["_scale"] = np.broadcast_to(np.asarray(scale, dtype=np.float64), o.shape)
        params["scale_column"] = "_scale"
    requests = []
    for given in names:
        name = canonical_name(given)
        if name not in METRICS:
            raise KeyError(f"unknown metric {name!r}; available: {', '.join(METRICS)}")
        metric = METRICS[name]
        if "threshold" in metric.required and threshold is None:
            raise ValueError(f"{name} needs a threshold")
        if "scale" in metric.required and scale is None:
            raise ValueError(f"{name} needs a scale")
        if metric.needs_time and times is None:
            raise ValueError(f"{name} needs times")
        if "quantile" in metric.needs and quantiles is None:
            raise ValueError(f"{name} needs quantiles")
        if t is not None and not metric.transforms:
            raise ValueError(f"{name} doesn't take a transform")
        requests.append(MetricRequest(name, t, Params(**params)))
    values = compute_metrics(requests, o, p, gid, 1, columns=columns)
    return {r.name: _value(r.name, float(values[r][0, 0])) for r in requests}


def _one(name: str, obs: ArrayLike, pred: ArrayLike, **kw: Any) -> Any:
    return evaluate(obs, pred, [name], **kw)[canonical_name(name)]


def count(obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None) -> int:
    """Count pairs with finite obs and pred."""
    return int(_one("count", obs, pred, transform=transform) or 0)


def mean_error(obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None) -> float | None:
    """Compute mean error.

    mean(pred - obs).
    """
    value: float | None = _one("mean_error", obs, pred, transform=transform)
    return value


def mean_absolute_error(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute mean absolute error.

    mean(|pred - obs|).
    """
    value: float | None = _one("mean_absolute_error", obs, pred, transform=transform)
    return value


def mean_square_error(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute mean squared error.

    mean((pred - obs)^2).
    """
    value: float | None = _one("mean_squared_error", obs, pred, transform=transform)
    return value


def root_mean_square_error(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute rmse.

    Root mean square error: sqrt(mean((pred - obs)^2)).
    """
    value: float | None = _one("rmse", obs, pred, transform=transform)
    return value


def max_value_delta(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute max value delta.

    max(pred) - max(obs).
    """
    value: float | None = _one("max_value_delta", obs, pred, transform=transform)
    return value


def relative_bias(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute relative bias.

    sum(pred - obs) / sum(obs); null if sum(obs) == 0.
    """
    value: float | None = _one("relative_bias", obs, pred, transform=transform)
    return value


def mean_absolute_relative_error(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute mean absolute relative error.

    sum(|pred - obs|) / sum(obs); null if sum(obs) == 0.
    """
    value: float | None = _one("mean_absolute_relative_error", obs, pred, transform=transform)
    return value


def multiplicative_bias(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute multiplicative bias.

    mean(pred) / mean(obs); null if mean(obs) == 0.
    """
    value: float | None = _one("multiplicative_bias", obs, pred, transform=transform)
    return value


def relative_median(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute relative median.

    median(pred) / median(obs); null if median(obs) == 0.
    """
    value: float | None = _one("relative_median", obs, pred, transform=transform)
    return value


def relative_minimum(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute relative min.

    min(pred) / min(obs); null if min(obs) == 0.
    """
    value: float | None = _one("relative_min", obs, pred, transform=transform)
    return value


def relative_maximum(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute relative max.

    max(pred) / max(obs); null if max(obs) == 0.
    """
    value: float | None = _one("relative_max", obs, pred, transform=transform)
    return value


def relative_standard_deviation(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute relative std.

    std(pred) / std(obs) (population std); null if obs is constant.
    """
    value: float | None = _one("relative_std", obs, pred, transform=transform)
    return value


def mean_absolute_percentage_error(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute mape.

    Mean absolute percentage error: 100 * mean(|pred - obs| / |obs|) over pairs with obs
    != 0; null if there are none.
    """
    value: float | None = _one("mape", obs, pred, transform=transform)
    return value


def symmetric_mean_absolute_percentage_error(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute smape.

    Symmetric MAPE: 100 * mean(2|pred - obs| / (|obs| + |pred|)); a pair where both are
    0 counts as 0 error. Ranges 0 to 200.
    """
    value: float | None = _one("smape", obs, pred, transform=transform)
    return value


def normalized_mean_absolute_error(
    obs: ArrayLike,
    pred: ArrayLike,
    *,
    normalize_by: str | float | ArrayLike = "mean_primary",
    transform: TransformLike = None,
) -> float | None:
    """Compute nmae.

    MAE / normalizer: mean(obs) by default; normalize_by range_primary (max - min), a number
    (e.g. installed capacity) or a column. Null if the normalizer is 0.
    """
    value: float | None = _one("nmae", obs, pred, normalize_by=normalize_by, transform=transform)
    return value


def normalized_root_mean_square_error(
    obs: ArrayLike,
    pred: ArrayLike,
    *,
    normalize_by: str | float | ArrayLike = "mean_primary",
    transform: TransformLike = None,
) -> float | None:
    """Compute nrmse.

    RMSE / normalizer (as nmae).
    """
    value: float | None = _one("nrmse", obs, pred, normalize_by=normalize_by, transform=transform)
    return value


def mean_absolute_scaled_error(
    obs: ArrayLike, pred: ArrayLike, scale: float | ArrayLike
) -> float | None:
    """Compute the mean absolute scaled error: mean(|pred - obs| / scale).

    ``scale`` is the in-sample naive error (e.g. mean |y(t) - y(t - lag)| over a
    training period): one number, or one per pair. < 1 beats that naive forecast.
    """
    value: float | None = _one("mase", obs, pred, scale=scale)
    return value


def pearson_correlation(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute pearson r.

    Pearson correlation; null if obs or pred is constant.
    """
    value: float | None = _one("pearson_r", obs, pred, transform=transform)
    return value


def spearman_correlation(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute spearman r.

    Spearman rank correlation (ties get average ranks); null if obs or pred is constant.
    """
    value: float | None = _one("spearman_r", obs, pred, transform=transform)
    return value


def r_squared(obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None) -> float | None:
    """Compute r squared.

    Coefficient of determination as the squared Pearson correlation.
    """
    value: float | None = _one("r_squared", obs, pred, transform=transform)
    return value


def coefficient_of_variation_ratio(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute coefficient of variation ratio.

    Ratio of coefficients of variation: (std(pred)/mean(pred)) / (std(obs)/mean(obs));
    null if obs is constant or a mean is 0.
    """
    value: float | None = _one("coefficient_of_variation_ratio", obs, pred, transform=transform)
    return value


def root_mean_standard_deviation_ratio(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute rsr.

    RMSE-observations standard deviation ratio: RMSE / std(obs); null if obs is
    constant.
    """
    value: float | None = _one("rsr", obs, pred, transform=transform)
    return value


def nash_sutcliffe_efficiency(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute nse.

    Nash-Sutcliffe efficiency: 1 - sum((pred - obs)^2) / sum((obs - mean(obs))^2); null
    if obs is constant (within rounding).
    """
    value: float | None = _one("nse", obs, pred, transform=transform)
    return value


def nash_sutcliffe_efficiency_normalized(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute nnse.

    Normalized NSE: 1 / (2 - NSE), in (0, 1]; null if obs is constant.
    """
    value: float | None = _one("nnse", obs, pred, transform=transform)
    return value


def kling_gupta_efficiency(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute kge.

    Kling-Gupta efficiency (Gupta et al. 2009): 1 - sqrt((r-1)^2 + (alpha-1)^2 +
    (beta-1)^2), alpha = std(pred)/std(obs), beta = mean(pred)/mean(obs); null if obs or
    pred is constant or mean(obs) == 0.
    """
    value: float | None = _one("kge", obs, pred, transform=transform)
    return value


def kling_gupta_efficiency_mod1(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute kge 2012.

    KGE' (Kling et al. 2012): as KGE with alpha replaced by the ratio of coefficients of
    variation (pred/obs).
    """
    value: float | None = _one("kge_2012", obs, pred, transform=transform)
    return value


def kling_gupta_efficiency_mod2(
    obs: ArrayLike, pred: ArrayLike, *, transform: TransformLike = None
) -> float | None:
    """Compute kge 2021.

    KGE'' (Tang et al. 2021): 1 - sqrt((r-1)^2 + (alpha-1)^2 + beta^2), alpha =
    std(pred)/std(obs), beta = (mean(pred) - mean(obs)) / std(obs).
    """
    value: float | None = _one("kge_2021", obs, pred, transform=transform)
    return value


def hits(obs: ArrayLike, pred: ArrayLike, *, threshold: float | ArrayLike) -> int:
    """Compute hits.

    Pairs where obs and pred are events. An event is a value >= threshold.
    """
    value = _one("hits", obs, pred, threshold=threshold)
    return int(value or 0)


def misses(obs: ArrayLike, pred: ArrayLike, *, threshold: float | ArrayLike) -> int:
    """Compute misses.

    Pairs where obs is an event and pred isn't. An event is a value >= threshold.
    """
    value = _one("misses", obs, pred, threshold=threshold)
    return int(value or 0)


def false_alarms(obs: ArrayLike, pred: ArrayLike, *, threshold: float | ArrayLike) -> int:
    """Compute false alarms.

    Pairs where pred is an event and obs isn't. An event is a value >= threshold.
    """
    value = _one("false_alarms", obs, pred, threshold=threshold)
    return int(value or 0)


def correct_negatives(obs: ArrayLike, pred: ArrayLike, *, threshold: float | ArrayLike) -> int:
    """Compute correct negatives.

    Pairs where neither is an event. An event is a value >= threshold.
    """
    value = _one("correct_negatives", obs, pred, threshold=threshold)
    return int(value or 0)


def probability_of_detection(
    obs: ArrayLike, pred: ArrayLike, *, threshold: float | ArrayLike
) -> float | None:
    """Compute pod.

    Probability of detection: hits / (hits + misses); null without observed events. An
    event is a value >= threshold.
    """
    value: float | None = _one("pod", obs, pred, threshold=threshold)
    return value


def probability_of_false_detection(
    obs: ArrayLike, pred: ArrayLike, *, threshold: float | ArrayLike
) -> float | None:
    """Compute pofd.

    Probability of false detection: false_alarms / (false_alarms + correct_negatives).
    An event is a value >= threshold.
    """
    value: float | None = _one("pofd", obs, pred, threshold=threshold)
    return value


def false_alarm_ratio(
    obs: ArrayLike, pred: ArrayLike, *, threshold: float | ArrayLike
) -> float | None:
    """Compute far.

    False alarm ratio: false_alarms / (hits + false_alarms); null without predicted
    events. An event is a value >= threshold.
    """
    value: float | None = _one("far", obs, pred, threshold=threshold)
    return value


def critical_success_index(
    obs: ArrayLike, pred: ArrayLike, *, threshold: float | ArrayLike
) -> float | None:
    """Compute csi.

    Critical success index (threat score): hits / (hits + misses + false_alarms). An
    event is a value >= threshold.
    """
    value: float | None = _one("csi", obs, pred, threshold=threshold)
    return value


def success_ratio(obs: ArrayLike, pred: ArrayLike, *, threshold: float | ArrayLike) -> float | None:
    """Compute success ratio.

    Success ratio: hits / (hits + false_alarms) = 1 - FAR. An event is a value >=
    threshold.
    """
    value: float | None = _one("success_ratio", obs, pred, threshold=threshold)
    return value


def frequency_bias_index(
    obs: ArrayLike, pred: ArrayLike, *, threshold: float | ArrayLike
) -> float | None:
    """Compute frequency bias.

    Frequency bias: (hits + false_alarms) / (hits + misses); 1 = as many predicted as
    observed events. An event is a value >= threshold.
    """
    value: float | None = _one("frequency_bias", obs, pred, threshold=threshold)
    return value


def max_value_time_delta(obs: ArrayLike, pred: ArrayLike, times: TimesLike) -> float | None:
    """Compute max value time delta.

    Seconds from the time of max(obs) to the time of max(pred) (earliest time of each
    maximum); positive means the predicted peak is later.
    """
    value: float | None = _one("max_value_time_delta", obs, pred, times=times)
    return value


def annual_peak_flow_bias(
    obs: ArrayLike, pred: ArrayLike, times: TimesLike, *, year_start_month: int = 1
) -> float | None:
    """Compute annual peak relative bias.

    sum over years of (annual max(pred) - annual max(obs)) / sum of annual max(obs).
    Years start in year_start_month (default 1, calendar years; 10 for water years).
    """
    value: float | None = _one(
        "annual_peak_relative_bias", obs, pred, times=times, year_start_month=year_start_month
    )
    return value


def minimum(values: ArrayLike) -> float | None:
    """Compute the min value of one series.

    Minimum. Of obs or pred (side, default obs); column <name>_<side>.
    """
    value: float | None = _one("min_value", values, values)
    return value


def maximum(values: ArrayLike) -> float | None:
    """Compute the max value of one series.

    Maximum. Of obs or pred (side, default obs); column <name>_<side>.
    """
    value: float | None = _one("max_value", values, values)
    return value


def average(values: ArrayLike) -> float | None:
    """Compute the mean value of one series.

    Mean. Of obs or pred (side, default obs); column <name>_<side>.
    """
    value: float | None = _one("mean_value", values, values)
    return value


def sum_value(values: ArrayLike) -> float | None:
    """Compute the sum value of one series.

    Sum. Of obs or pred (side, default obs); column <name>_<side>.
    """
    value: float | None = _one("sum_value", values, values)
    return value


def variance(values: ArrayLike) -> float | None:
    """Compute the variance of one series.

    Population variance. Of obs or pred (side, default obs); column <name>_<side>.
    """
    value: float | None = _one("variance", values, values)
    return value


def max_value_time(values: ArrayLike, times: TimesLike) -> datetime | None:
    """Compute the max value time of one series.

    Time of the maximum (earliest if repeated). Of obs or pred (side, default obs);
    column <name>_<side>.
    """
    value: datetime | None = _one("max_value_time", values, values, times=times)
    return value


def flow_duration_curve_slope(
    values: ArrayLike, *, exceedance: tuple[float, float] = (0.33, 0.66)
) -> float | None:
    """Compute the fdc slope of one series.

    Flow duration curve slope: (ln Q(e1) - ln Q(e2)) / (e2 - e1), Q(e) = value exceeded
    a fraction e of the time, exceedance [e1, e2] default [0.33, 0.66]; null if either
    value <= 0. Of obs or pred (side, default obs); column <name>_<side>.
    """
    value: float | None = _one("fdc_slope", values, values, exceedance=exceedance)
    return value


# -- probabilistic forecasts -------------------------------------------------------------
def _ensemble_long(
    obs: ArrayLike, ensemble: Sequence[Sequence[float]] | npt.NDArray[np.floating]
) -> tuple[list[float], list[float], list[int]]:
    """Long format: one (obs, member, case) row per member; members may vary in number."""
    obs_l, pred_l, case_l = [], [], []
    for c, (y, members) in enumerate(zip(np.asarray(obs, dtype=np.float64), ensemble, strict=True)):
        for x in np.asarray(members, dtype=np.float64).ravel():
            obs_l.append(float(y))
            pred_l.append(float(x))
            case_l.append(c)
    return obs_l, pred_l, case_l


def mean_crps_ensemble(
    obs: ArrayLike,
    ensemble: Sequence[Sequence[float]] | npt.NDArray[np.floating],
    *,
    estimator: str = "standard",
) -> float | None:
    """Compute the CRPS of ensemble forecasts, averaged over forecasts.

    ``ensemble`` has one row of members per observation (rows may differ in length).
    ``estimator`` is ``standard`` or ``fair``.
    """
    o, x, c = _ensemble_long(obs, ensemble)
    value: float | None = _one("crps", o, x, cases=c, estimator=estimator)
    return value


def mean_brier_score(
    obs: ArrayLike,
    ensemble: Sequence[Sequence[float]] | npt.NDArray[np.floating],
    *,
    threshold: float,
) -> float | None:
    """Compute the Brier score of the event value >= threshold from ensemble forecasts."""
    o, x, c = _ensemble_long(obs, ensemble)
    value: float | None = _one("brier_score", o, x, cases=c, threshold=threshold)
    return value


Ensemble = Sequence[Sequence[float]] | npt.NDArray[np.floating]


def ensemble_spread(obs: ArrayLike, ensemble: Ensemble) -> float | None:
    """Compute the ensemble spread: sqrt(mean of each forecast's member variance)."""
    o, x, c = _ensemble_long(obs, ensemble)
    value: float | None = _one("ensemble_spread", o, x, cases=c)
    return value


def ensemble_mean_root_mean_square_error(obs: ArrayLike, ensemble: Ensemble) -> float | None:
    """Compute the RMSE of the ensemble mean."""
    o, x, c = _ensemble_long(obs, ensemble)
    value: float | None = _one("ensemble_mean_rmse", o, x, cases=c)
    return value


def spread_skill_ratio(obs: ArrayLike, ensemble: Ensemble) -> float | None:
    """Compute the spread-skill ratio (about 1 for a reliable ensemble)."""
    o, x, c = _ensemble_long(obs, ensemble)
    value: float | None = _one("spread_skill_ratio", o, x, cases=c)
    return value


def rank_histogram(obs: ArrayLike, ensemble: Ensemble) -> list[int]:
    """Count forecasts by the observation's rank among their members (0 = below all).

    One count per rank ``0 .. m`` for the largest ensemble size ``m``; ties count half,
    rounded down.
    """
    y = np.asarray(obs, dtype=np.float64)
    members = [np.asarray(m, dtype=np.float64).ravel() for m in ensemble]
    counts = [0] * (max((len(m) for m in members), default=0) + 1)
    for value, xs in zip(y, members, strict=True):
        counts[int(np.sum(xs < value) + np.sum(xs == value) // 2)] += 1
    return counts


def rank_histogram_reliability(obs: ArrayLike, ensemble: Ensemble) -> float | None:
    """Compute the rank histogram's reliability index (0 = flat; needs equal sizes)."""
    o, x, c = _ensemble_long(obs, ensemble)
    value: float | None = _one("rank_histogram_reliability", o, x, cases=c)
    return value


def mean_pinball_loss(obs: ArrayLike, pred: ArrayLike, quantile: float | ArrayLike) -> float | None:
    """Compute the mean pinball (quantile) loss of quantile forecasts.

    ``quantile`` is the level of ``pred``: one number, or one per pair.
    """
    value: float | None = _one("pinball_loss", obs, pred, quantiles=quantile)
    return value


def _interval(
    name: str, obs: ArrayLike, lower: ArrayLike, upper: ArrayLike, interval: tuple[float, float]
) -> float | None:
    y = np.asarray(obs, dtype=np.float64)
    o = np.concatenate([y, y])
    x = np.concatenate([np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)])
    levels = np.concatenate([np.full(y.size, interval[0]), np.full(y.size, interval[1])])
    c = np.concatenate([np.arange(y.size), np.arange(y.size)])
    value: float | None = _one(name, o, x, cases=c, quantiles=levels, interval=interval)
    return value


def interval_coverage(
    obs: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    interval: tuple[float, float] = (0.05, 0.95),
) -> float | None:
    """Compute the fraction of observations within [lower, upper]."""
    return _interval("interval_coverage", obs, lower, upper, interval)


def mean_interval_width(
    obs: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    interval: tuple[float, float] = (0.05, 0.95),
) -> float | None:
    """Compute the mean width of [lower, upper] (obs only sets which pairs count)."""
    return _interval("interval_width", obs, lower, upper, interval)


def mean_interval_score(
    obs: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    interval: tuple[float, float] = (0.05, 0.95),
) -> float | None:
    """Compute the interval (Winkler) score; ``interval`` gives the levels (alpha)."""
    return _interval("winkler_score", obs, lower, upper, interval)


# Short names (the metric aliases) for the functions above.
mean_squared_error = mean_square_error
rmse = root_mean_square_error
relative_min = relative_minimum
relative_max = relative_maximum
relative_std = relative_standard_deviation
mape = mean_absolute_percentage_error
smape = symmetric_mean_absolute_percentage_error
nmae = normalized_mean_absolute_error
nrmse = normalized_root_mean_square_error
mase = mean_absolute_scaled_error
pearson_r = pearson_correlation
spearman_r = spearman_correlation
rsr = root_mean_standard_deviation_ratio
nse = nash_sutcliffe_efficiency
nnse = nash_sutcliffe_efficiency_normalized
kge = kling_gupta_efficiency
kge_2012 = kling_gupta_efficiency_mod1
kge_2021 = kling_gupta_efficiency_mod2
pod = probability_of_detection
pofd = probability_of_false_detection
far = false_alarm_ratio
csi = critical_success_index
frequency_bias = frequency_bias_index
annual_peak_relative_bias = annual_peak_flow_bias
min_value = minimum
max_value = maximum
mean_value = average
fdc_slope = flow_duration_curve_slope
crps = mean_crps_ensemble
brier_score = mean_brier_score
pinball_loss = mean_pinball_loss
interval_width = mean_interval_width
winkler_score = mean_interval_score
ensemble_mean_rmse = ensemble_mean_root_mean_square_error
variability_ratio = relative_standard_deviation
relative_mean = multiplicative_bias


__all__ = [
    "METRICS",
    "Metric",
    "MetricRequest",
    "Params",
    "Transform",
    "annual_peak_flow_bias",
    "annual_peak_relative_bias",
    "average",
    "brier_score",
    "coefficient_of_variation_ratio",
    "compute_grouped",
    "compute_metrics",
    "correct_negatives",
    "count",
    "critical_success_index",
    "crps",
    "csi",
    "ensemble_mean_rmse",
    "ensemble_mean_root_mean_square_error",
    "ensemble_spread",
    "evaluate",
    "false_alarm_ratio",
    "false_alarms",
    "far",
    "fdc_slope",
    "flow_duration_curve_slope",
    "frequency_bias",
    "frequency_bias_index",
    "hits",
    "interval_coverage",
    "interval_width",
    "kge",
    "kge_2012",
    "kge_2021",
    "kling_gupta_efficiency",
    "kling_gupta_efficiency_mod1",
    "kling_gupta_efficiency_mod2",
    "mape",
    "mase",
    "max_value",
    "max_value_delta",
    "max_value_time",
    "max_value_time_delta",
    "maximum",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_absolute_relative_error",
    "mean_absolute_scaled_error",
    "mean_brier_score",
    "mean_crps_ensemble",
    "mean_error",
    "mean_interval_score",
    "mean_interval_width",
    "mean_pinball_loss",
    "mean_square_error",
    "mean_squared_error",
    "mean_value",
    "metric_names",
    "min_value",
    "minimum",
    "misses",
    "multiplicative_bias",
    "nash_sutcliffe_efficiency",
    "nash_sutcliffe_efficiency_normalized",
    "nmae",
    "nnse",
    "normalized_mean_absolute_error",
    "normalized_root_mean_square_error",
    "nrmse",
    "nse",
    "pearson_correlation",
    "pearson_r",
    "pinball_loss",
    "pod",
    "pofd",
    "probability_of_detection",
    "probability_of_false_detection",
    "r_squared",
    "rank_histogram",
    "rank_histogram_reliability",
    "relative_bias",
    "relative_max",
    "relative_maximum",
    "relative_mean",
    "relative_median",
    "relative_min",
    "relative_minimum",
    "relative_standard_deviation",
    "relative_std",
    "rmse",
    "root_mean_square_error",
    "root_mean_standard_deviation_ratio",
    "rsr",
    "smape",
    "spearman_correlation",
    "spearman_r",
    "spread_skill_ratio",
    "success_ratio",
    "sum_value",
    "symmetric_mean_absolute_percentage_error",
    "variability_ratio",
    "variance",
    "winkler_score",
]
