"""How each metric's value changes when the data's unit changes.

Ported from VeriRace (``verirace/metrics/units.py`` at 668ab7d).

Metrics are computed in one unit (the observed dataset's, or the spec's ``output.unit``).
A conversion to another unit of the same dimension is ``x' = a * x + b`` (``b`` is nonzero
only between temperature scales). Each metric has a :class:`UnitKind` saying how its
result follows: a level of the variable converts like a value, an error by ``a`` only, a
squared error by ``a**2``; dimensionless metrics don't change. Ratios of values are
dimensionless for units without an offset but change with one, so they can't be
converted then (evaluate in the target unit instead). See docs/design/metric-units.md.

:func:`result_factors` maps every column of a spec's results to the linear map that
converts it, and :func:`convert_results` applies them.
"""

from __future__ import annotations

import math
from enum import StrEnum

import pyarrow as pa
import pyarrow.compute as pc

from teehr_core.metrics.registry import Transform
from teehr_core.model.units import Conversion
from teehr_core.spec.models import REASON_COLUMN, ChangeField, EvaluationSpec, MetricSpec

__all__ = [
    "KINDS",
    "NotConvertible",
    "UnitKind",
    "convert_results",
    "kind_factor",
    "metric_kind",
    "result_factors",
]


class UnitKind(StrEnum):
    """How a metric's value follows a change of unit (``x' = a * x + b``)."""

    #: A level of the variable (min, max, mean): ``a * x + b``.
    VALUE = "value"
    #: A sum of values: ``a * x + n * b`` (so only ``a * x`` without an offset).
    SUM = "sum"
    #: A difference or spread: ``a * x``.
    DIFFERENCE = "difference"
    #: A squared difference or a variance: ``a**2 * x``.
    DIFFERENCE2 = "difference2"
    #: Dimensionless whatever the units, offsets included.
    INVARIANT = "invariant"
    #: A ratio of values: unchanged without an offset; not convertible with one.
    RATIO = "ratio"
    #: A time or duration.
    TIME = "time"


_V, _S, _D, _D2, _I, _R, _T = (
    UnitKind.VALUE,
    UnitKind.SUM,
    UnitKind.DIFFERENCE,
    UnitKind.DIFFERENCE2,
    UnitKind.INVARIANT,
    UnitKind.RATIO,
    UnitKind.TIME,
)

#: The unit kind of every registered metric.
KINDS: dict[str, UnitKind] = {
    "count": _I,
    "mean_error": _D,
    "mean_absolute_error": _D,
    "mean_square_error": _D2,
    "root_mean_square_error": _D,
    "max_value_delta": _D,
    "relative_bias": _R,
    "mean_absolute_relative_error": _R,
    "multiplicative_bias": _R,
    "relative_median": _R,
    "relative_minimum": _R,
    "relative_maximum": _R,
    "relative_standard_deviation": _I,
    "mean_absolute_percentage_error": _R,
    "symmetric_mean_absolute_percentage_error": _R,
    # Invariant with normalize_by: range_primary (see metric_kind).
    "normalized_mean_absolute_error": _R,
    "normalized_root_mean_square_error": _R,
    "mean_absolute_scaled_error": _I,
    "pearson_correlation": _I,
    "spearman_correlation": _I,
    "r_squared": _I,
    "coefficient_of_variation_ratio": _R,
    "root_mean_standard_deviation_ratio": _I,
    "nash_sutcliffe_efficiency": _I,
    "nash_sutcliffe_efficiency_normalized": _I,
    "kling_gupta_efficiency": _R,
    "kling_gupta_efficiency_mod1": _R,
    "kling_gupta_efficiency_mod2": _I,
    "hits": _I,
    "misses": _I,
    "false_alarms": _I,
    "correct_negatives": _I,
    "probability_of_detection": _I,
    "probability_of_false_detection": _I,
    "false_alarm_ratio": _I,
    "critical_success_index": _I,
    "success_ratio": _I,
    "frequency_bias_index": _I,
    "proportion_correct": _I,
    "max_value_time_delta": _T,
    "annual_peak_flow_bias": _R,
    "minimum": _V,
    "maximum": _V,
    "average": _V,
    "sum": _S,
    "variance": _D2,
    "max_value_time": _T,
    "flow_duration_curve_slope": _R,  # linear (log: false): a difference (see metric_kind)
    "center_of_timing": _I,  # days of the year
    "standard_deviation_of_timing": _I,
    "mean_crps_ensemble": _D,
    "mean_brier_score": _I,
    "mean_pinball_loss": _D,
    "interval_coverage": _I,
    "mean_interval_width": _D,
    "mean_interval_score": _D,
    "ensemble_spread": _D,
    "ensemble_mean_root_mean_square_error": _D,
    "spread_skill_ratio": _I,
    "rank_frequency": _I,
    "rank_outlier_fraction": _I,
    "rank_histogram_reliability": _I,
}

#: A linear map ``(s, t)``: ``x' = s * x + t``; None when the column can't be converted.
Factor = tuple[float, float] | None
_KEEP: Factor = (1.0, 0.0)


class NotConvertible(ValueError):
    """Some result columns can't be converted between these units (an offset)."""

    def __init__(self, columns: list[str], conversion: Conversion) -> None:
        self.columns = columns
        super().__init__(
            f"{', '.join(columns)} can't be converted between units with different zero "
            "points (e.g. degF and Cel): they divide by levels of the variable. Evaluate "
            "in the target unit instead (output.unit in the spec)."
        )


def metric_kind(m: MetricSpec) -> UnitKind:
    """Return the unit kind of a first-stage metric, given its options."""
    kind = KINDS[m.name]
    if (
        m.name in ("normalized_mean_absolute_error", "normalized_root_mean_square_error")
        and m.normalize_by == "range_primary"
    ):
        return _I
    if m.name == "flow_duration_curve_slope" and m.log is False:
        return _D
    return kind


def kind_factor(kind: UnitKind, transform: Transform | None, conv: Conversion) -> Factor:
    """Return how a metric of ``kind`` converts: ``(s, t)`` for ``s * x + t``, or None."""
    a, b = conv.scale, conv.offset
    if kind in (_I, _T):
        return _KEEP
    if transform is not None:
        if b != 0:
            return None
        return _transformed(kind, transform, a)
    if kind is _V:
        return (a, b)
    if kind is _S:
        return (a, 0.0) if b == 0 else None
    if kind is _D:
        return (a, 0.0)
    if kind is _D2:
        return (a * a, 0.0)
    return _KEEP if b == 0 else None  # ratio


def _transformed(kind: UnitKind, transform: Transform, a: float) -> Factor:
    """Factor for a metric of transformed values, without an offset."""
    if transform is Transform.LOG:
        # log(a x) = log a + log x: levels shift, differences don't change.
        if kind in (_V, _S):
            return (1.0, math.log(a)) if kind is _V else None
        return _KEEP if kind in (_D, _D2) else None
    if transform is Transform.EXP:  # exp(a x) isn't a scaling of exp(x)
        return _KEEP if a == 1 else None
    power = {
        Transform.SQRT: 0.5,
        Transform.SQUARE: 2.0,
        Transform.CUBE: 3.0,
        Transform.RECIPROCAL: -1.0,
        Transform.ABS: 1.0,  # |a x| = a |x| for a > 0
    }[transform]
    if kind is _R:
        return _KEEP
    return (a ** (power * (2 if kind is _D2 else 1)), 0.0)


def _summary(name: str, base: Factor) -> Factor:
    """Factor of a summary (later stage) of a column converting by ``base``."""
    if name == "count":
        return _KEEP
    if base is None:
        return None
    s, t = base
    if name in ("mean", "median", "min", "max"):
        return (s, t)
    if name == "std":
        return (abs(s), 0.0)
    # sum: sum(s x + t) = s sum(x) + n t
    return (s, 0.0) if t == 0 else None


def result_factors(spec: EvaluationSpec, conv: Conversion) -> dict[str, Factor]:
    """Map each column of the spec's results (its last stage) to its conversion.

    Group-by columns and the guards' reason column aren't listed (they pass through).
    """
    stages = spec.aggregate
    first = stages[0]
    changes = {f.name for f in spec.fields if isinstance(f, ChangeField)}
    cols: dict[str, Factor] = {}
    for m in first.metrics:
        kind = metric_kind(m)
        if m.values is not None and kind not in (_I, _T):
            # Changes over a window are differences of values; other columns: unknown.
            if {m.values.primary, m.values.secondary} <= changes:
                kind = {_V: _D, _S: _D}.get(kind, kind)
            else:
                cols[m.output_name()] = None
                continue
        cols[m.output_name()] = kind_factor(kind, m.transform_name, conv)
    if first.bootstrap is not None:
        for m in first.bootstrapped():
            for c in first.bootstrap.columns(m.output_name()):
                cols[c] = cols[m.output_name()]
    if first.skill is not None:
        for c in first.skill.metrics:
            cols[first.skill.column(c)] = _KEEP
        for c in first.skill.test_columns():
            cols[c] = _KEEP
    for stage in stages[1:]:
        cols = {
            m.output_name(): _summary(m.name, cols.get(m.of or "", _KEEP)) for m in stage.metrics
        }
    return cols


def convert_results(table: pa.Table, spec: EvaluationSpec, conv: Conversion) -> pa.Table:
    """Convert a results table to another unit (raises :class:`NotConvertible`)."""
    factors = result_factors(spec, conv)
    blocked = [c for c in table.column_names if c in factors and factors[c] is None]
    if blocked:
        raise NotConvertible(blocked, conv)
    for i, name in enumerate(table.column_names):
        factor = factors.get(name)
        if name == REASON_COLUMN or factor is None or factor == _KEEP:
            continue
        col = table.column(i)
        if not pa.types.is_floating(col.type) and not pa.types.is_integer(col.type):
            continue
        s, t = factor
        table = table.set_column(i, name, pc.add(pc.multiply(pc.cast(col, pa.float64()), s), t))
    return table
