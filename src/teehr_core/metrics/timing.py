"""Timing signatures: when in the (water) year a series' volume arrives.

Ported from teehr (``teehr/metrics/signature_funcs.py``: ``center_of_timing``,
``standard_deviation_of_timing``) to teehr-core's group statistics.

Both work on daily values (the mean of a day's values), year by year, and average over
the years: a year with fewer than ``1 - missing_day_threshold`` of its days present is
left out, as are days with no flow. Days are counted from 1 at the start of the year
(``year_start_month``, default 10: US water years). Replicate weights (bootstrap
resample counts) weight each value, as if it were repeated.

These need the pairs, so they don't merge from state.
"""

from __future__ import annotations

import numpy as np

from teehr_core.metrics.stats import FloatArray, GroupStats

_DAY = 86_400


def _year_starts(days: np.ndarray, start_month: int) -> np.ndarray:
    """Return the first day (days since epoch) of the year each day belongs to."""
    months = (days * _DAY).astype("datetime64[s]").astype("datetime64[M]").astype(np.int64)
    year = (months - (start_month - 1)) // 12
    first = (year * 12 + (start_month - 1)).astype("datetime64[M]")
    return np.asarray(first.astype("datetime64[D]").astype(np.int64))


def _year_length(start: int, start_month: int) -> int:
    first = np.datetime64(start, "D").astype("datetime64[M]")
    nxt = (first + np.timedelta64(12, "M")).astype("datetime64[D]").astype(np.int64)
    return int(nxt - start)


def _one_series(
    values: np.ndarray,
    seconds: np.ndarray,
    weights: np.ndarray,
    start_month: int,
    missing: float,
    spread: bool,
) -> float:
    keep = weights > 0
    values, seconds, weights = values[keep], seconds[keep], weights[keep]
    if values.size == 0:
        return float("nan")
    days = np.floor(seconds / _DAY).astype(np.int64)
    uniq, inverse = np.unique(days, return_inverse=True)
    daily = np.bincount(inverse, weights=weights * values) / np.bincount(inverse, weights=weights)
    starts = _year_starts(uniq, start_month)
    results = []
    for start in np.unique(starts):
        in_year = starts == start
        length = _year_length(int(start), start_month)
        if in_year.sum() < (1 - missing) * length:
            continue
        q = daily[in_year]
        d = (uniq[in_year] - start + 1).astype(np.float64)
        nonzero = q > 0
        q, d = q[nonzero], d[nonzero]
        if q.size == 0 or (spread and q.size <= 1):
            continue
        total = q.sum()
        if total <= 0:
            continue
        ct = float(np.sum(q * d) / total)
        if not spread:
            results.append(ct)
            continue
        ct = ((ct - 1) % length) + 1
        n = q.size
        denominator = total * (n - 1) / n
        if denominator == 0:
            continue
        results.append(float(np.sqrt(np.sum(q * (d - ct) ** 2) / denominator)))
    return float(np.mean(results)) if results else float("nan")


def timing(s: GroupStats, side: str, start_month: int, missing: float, spread: bool) -> FloatArray:
    """Center of timing (``spread`` false) or its standard deviation per replicate, group."""
    values = s.obs if side == "obs" else s.pred
    seconds = s.column("value_time")
    out = np.full((s.replicates, s.n_groups), np.nan)
    bounds = np.searchsorted(s.gid, np.arange(s.n_groups + 1))
    for g in range(s.n_groups):
        lo, hi = bounds[g], bounds[g + 1]
        if lo == hi:
            continue
        for r in range(s.replicates):
            out[r, g] = _one_series(
                values[lo:hi], seconds[lo:hi], np.asarray(s.weights[r, lo:hi]),
                start_month, missing, spread,
            )  # fmt: skip
    return out
