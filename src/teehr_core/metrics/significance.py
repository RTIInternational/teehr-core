"""Significance of the difference between two forecasts: the Diebold-Mariano test.

Ported from VeriRace (``verirace/metrics/significance.py`` at 668ab7d).

Given each matched observation's loss differential ``d = loss(forecast) - loss(reference)``
(squared or absolute errors), the test statistic is ``mean(d) / sqrt(LRV / n)``, where the
long-run variance LRV accounts for autocorrelated errors with a Newey-West (Bartlett)
estimate. Under no difference it is approximately standard normal; negative means the
forecast's losses are smaller. Diebold & Mariano (1995); Newey & West (1987).

Differentials come per series: autocovariances are summed within series (series are
treated as independent of each other). Lag k pairs a differential with the one k time
steps earlier in its series (none when that time has no differential), so the test
merges across parts of the data: :func:`dm_lag_sums` reduces any part to sums per lag
that add, and :func:`dm_from_sums` finishes the test from them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "DM_SUMS",
    "default_lag",
    "diebold_mariano",
    "dm_from_sums",
    "dm_lag_sums",
    "time_step",
]

#: The sums per lag k (0..K) a group's test needs: pairs (t, t - k steps) of one series,
#: their count, the sums of the later and earlier differentials, and of their products.
DM_SUMS = ("dm_n", "dm_a", "dm_b", "dm_s")


def default_lag(n: int) -> int:
    """Newey-West's rule of thumb for the number of lags: floor(4 (n / 100)^(2/9))."""
    return math.floor(4 * (n / 100) ** (2 / 9)) if n > 0 else 0


def diebold_mariano(
    differentials: Sequence[npt.ArrayLike], lag: int | None = None
) -> tuple[float, float] | None:
    """Return ``(statistic, two-sided p-value)``, or None if it can't be computed.

    ``differentials`` holds one array of loss differentials per series, one per time step
    in time order (a NaN is a missing step). None when there are fewer than 2 values or
    the variance is not positive (e.g. the two forecasts are identical).
    """
    series = [np.asarray(s, dtype=np.float64) for s in differentials]
    if not series:
        return None
    d = np.concatenate(series)
    ids = np.repeat(np.arange(len(series)), [s.size for s in series])
    steps = np.concatenate([np.arange(s.size) for s in series])
    n = int(np.isfinite(d).sum())
    max_lag = default_lag(n) if lag is None else lag
    n_k, a, b, s = dm_lag_sums(d, np.zeros(d.size, np.int64), ids, steps, 1, max_lag, 1)
    return dm_from_sums(n_k[0], a[0], b[0], s[0], lag)


def dm_lag_sums(
    d: npt.ArrayLike,
    gid: npt.ArrayLike,
    series: npt.ArrayLike,
    time: npt.ArrayLike,
    step: int,
    max_lag: int,
    n_groups: int,
    counted: npt.NDArray[np.bool_] | None = None,
    into: npt.NDArray[np.int64] | None = None,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Return the :data:`DM_SUMS` per group and lag: four ``(n_groups, max_lag + 1)`` arrays.

    ``d`` are loss differentials, ``gid`` their group, ``series`` their series (integer
    or string ids) and ``time`` their time as integers (e.g. epoch seconds) in units of
    which ``step`` is one time step. A pair (t, t - k steps) counts when both have a finite
    differential in the same group and series, and only when ``counted`` holds at t: a
    part of the data (a month) counts the pairs whose later end is in it, so parts' sums
    add up to the whole's when each is given the ``max_lag`` steps before it. ``into``
    (default ``gid``) is where each pair's sums go, by its later end: e.g. a group and
    month, with ``n_groups`` of them.
    """
    values = np.asarray(d, dtype=np.float64)
    keep = np.isfinite(values)
    later = keep if counted is None else keep & counted
    out = np.zeros((4, n_groups, max_lag + 1))
    if not later.any():
        return tuple(out)
    groups = np.asarray(gid)[keep]
    # One run per (group, series), times sorted within: a lag partner is found on
    # (run, time) as one integer key.
    _, run = np.unique(np.rec.fromarrays([groups, np.asarray(series)[keep]]), return_inverse=True)
    t = np.asarray(time, dtype=np.int64)[keep]
    span = int(t.max() - t.min()) + (max_lag + 1) * step + 1
    key = run.astype(np.int64) * span + (t - t.min())
    order = np.argsort(key, kind="stable")
    key, values, at = key[order], values[keep][order], later[keep][order]
    target = (groups if into is None else np.asarray(into)[keep])[order]
    for k in range(max_lag + 1):
        # Without gaps the partner is k places back (slices, no copies); the rest, and
        # the first k rows (a gap can put a partner closer), are looked up by time.
        head = min(k, key.size)
        later_key = key[head:]
        hit = at[head:] & (key[: key.size - head] == later_key - k * step)
        partner = np.where(hit, values[: key.size - head], 0.0)
        miss = np.flatnonzero(at[head:] & ~hit)
        if miss.size:
            found = np.minimum(np.searchsorted(key, later_key[miss] - k * step), key.size - 1)
            ok = key[found] == later_key[miss] - k * step
            hit[miss[ok]] = True
            partner[miss[ok]] = values[found[ok]]
        dt = np.where(hit, values[head:], 0.0)
        gk = target[head:]
        sums = (hit, dt, partner, dt * partner)
        for j, weights in enumerate(sums):
            out[j, :, k] = np.bincount(gk, weights=weights, minlength=n_groups)
        first = np.flatnonzero(at[:head])
        if first.size:
            found = np.searchsorted(key, key[first] - k * step)
            ok = (found < first) & (key[np.minimum(found, key.size - 1)] == key[first] - k * step)
            rows, partners = first[ok], found[ok]
            extra = (np.ones(rows.size), values[rows], values[partners])
            for j, weights in enumerate((*extra, values[rows] * values[partners])):
                out[j, :, k] += np.bincount(target[rows], weights=weights, minlength=n_groups)
    return tuple(out)


def time_step(series: npt.ArrayLike, time: npt.ArrayLike) -> int | None:
    """Return the smallest positive spacing of ``time`` within a series (None: none)."""
    t = np.asarray(time, dtype=np.int64)
    if t.size < 2:
        return None
    _, ids = np.unique(np.asarray(series), return_inverse=True)
    order = np.lexsort((t, ids))
    gaps = np.diff(t[order])[np.diff(ids[order]) == 0]
    gaps = gaps[gaps > 0]
    return int(gaps.min()) if gaps.size else None


def dm_from_sums(
    n: npt.ArrayLike, a: npt.ArrayLike, b: npt.ArrayLike, s: npt.ArrayLike, lag: int | None = None
) -> tuple[float, float] | None:
    """Return ``(statistic, two-sided p-value)`` from one group's :data:`DM_SUMS` by lag.

    The mean is ``a[0] / n[0]``; the lag-k autocovariance is ``(s - m (a + b) + n m^2)``
    over ``n[0]`` (each pair's centred product, summed). None as :func:`diebold_mariano`.
    """
    nk, ak, bk, sk = (np.asarray(x, dtype=np.float64) for x in (n, a, b, s))
    total = float(nk[0])
    if total < 2:
        return None
    lags = default_lag(int(total)) if lag is None else lag
    if lags >= nk.size:
        raise ValueError(f"the test needs {lags} lags; the sums have {nk.size - 1}")
    mean = float(ak[0]) / total
    gamma = (sk - mean * (ak + bk) + nk * mean**2) / total
    weights = 1 - np.arange(1, lags + 1) / (lags + 1)
    variance = float(gamma[0] + 2 * (weights @ gamma[1 : lags + 1]))
    # From uncentred sums, a zero variance (identical forecasts) can come out as rounding
    # noise: anything below that relative to the mean square counts as zero.
    if not variance > 1e-12 * float(sk[0]) / total:
        return None
    statistic = mean / math.sqrt(variance / total)
    return statistic, math.erfc(abs(statistic) / math.sqrt(2))
