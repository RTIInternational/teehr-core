"""Quantile sketches: order statistics that merge, at a fixed relative accuracy.

Ported from VeriRace (``verirace/metrics/sketch.py`` at 668ab7d).

A value's **bucket** is ``ceil(log_gamma |x|)`` with ``gamma = (1 + ALPHA) / (1 - ALPHA)``
(as in DDSketch), stored with its sign; its **representative** is ``2 gamma^k / (gamma + 1)``,
within ``ALPHA`` (0.1%) of every value in the bucket. A group's sketch is its weighted
count per bucket. Counts add, so sketches of parts of a group's pairs (months, shards,
bootstrap units) merge exactly, whatever the order or split.

Quantile metrics (``relative_median``, ``fdc_slope``) always use them: on the pairs they
take quantiles of values snapped to their representatives (:func:`snap`), which are the
same order statistics a merged sketch gives (:func:`sketch_quantiles`). So every path
(pairs, state, a bootstrap from state) gives the same result, within 0.1% of the exact
quantile of the values.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

__all__ = [
    "ALPHA",
    "bucket_ids",
    "representatives",
    "sketch",
    "sketch_quantiles",
    "snap",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Relative accuracy of a sketch's values.
ALPHA = 0.001
_GAMMA = (1 + ALPHA) / (1 - ALPHA)
_LOG_GAMMA = math.log(_GAMMA)
#: Magnitudes below this count as zero (bucket 0).
_MIN = 1e-12
#: Bucket ids are ``sign * (_OFFSET + k)``, so ids sort as their values do.
_OFFSET = 1 << 20


def bucket_ids(x: FloatArray) -> IntArray:
    """Return each (finite) value's bucket id; ids are ordered as the values are."""
    a = np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.ceil(np.log(np.where(a < _MIN, 1.0, a)) / _LOG_GAMMA)
    ids = (np.sign(x) * (_OFFSET + k)).astype(np.int64)
    return np.where(a < _MIN, 0, ids)


def representatives(ids: IntArray) -> FloatArray:
    """Return each bucket's representative value (0 for the zero bucket)."""
    k = np.abs(ids) - _OFFSET
    with np.errstate(over="ignore"):
        value = 2 * np.power(_GAMMA, k.astype(np.float64)) / (_GAMMA + 1)
    return np.where(ids == 0, 0.0, np.sign(ids) * value)


def snap(x: FloatArray) -> FloatArray:
    """Return each value's bucket representative (within ALPHA of it)."""
    return representatives(bucket_ids(x))


def sketch(x: FloatArray, gid: IntArray, n_groups: int) -> list[tuple[IntArray, FloatArray]]:
    """Return each group's sketch: its bucket ids (sorted) and the count in each."""
    ids = bucket_ids(x)
    out: list[tuple[IntArray, FloatArray]] = []
    order = np.lexsort((ids, gid))
    g, b = gid[order], ids[order]
    bounds = np.searchsorted(g, np.arange(n_groups + 1))
    for i in range(n_groups):
        part = b[bounds[i] : bounds[i + 1]]
        uniq, counts = np.unique(part, return_counts=True)
        out.append((uniq.astype(np.int64), counts.astype(np.float64)))
    return out


def sketch_quantiles(ids: IntArray, weights: FloatArray, n: FloatArray, q: float) -> FloatArray:
    """Return quantile ``q`` per replicate of a group's sketch.

    ``ids`` are its buckets (sorted), ``weights`` their counts ``(replicates, buckets)``
    and ``n`` the replicates' totals. As GroupStats.quantile on the snapped values: the
    mean of the order statistics at ``q * n`` (the first bucket whose cumulative count
    reaches it, and the first that exceeds it); NaN without values.
    """
    reps = weights.shape[0]
    out = np.full(reps, np.nan)
    if ids.size == 0:
        return out
    values = representatives(ids)
    cum = np.cumsum(weights, axis=1)
    target = np.round(q * n, 9)
    for r in range(reps):
        if not n[r] > 0:
            continue
        lo = min(int(np.searchsorted(cum[r], target[r], side="left")), ids.size - 1)
        hi = min(int(np.searchsorted(cum[r], target[r], side="right")), ids.size - 1)
        out[r] = (values[lo] + values[hi]) / 2
    return out
