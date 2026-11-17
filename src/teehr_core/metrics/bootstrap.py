"""Bootstrap confidence intervals, computed with the same formulas as point values.

Ported from VeriRace (``verirace/metrics/bootstrap.py`` at 668ab7d).

Each group is resampled ``reps`` times. A resample is expressed as **weights**: how
often each pair was drawn. All metrics of a stage then evaluate many replicates at once
through :func:`teehr_core.metrics.registry.compute_metrics`, sharing components exactly as
the point values do.

Block resamples are whole row ranges, so metrics built from sums and extremes are
evaluated from them directly (:class:`SegmentStats`): prefix sums and range queries cost
each replicate its blocks, not its pairs. Other metrics use the weights.

Determinism: each group's random streams are seeded from the stage seed and the group's
key values, and draws are consumed in a fixed order, so results don't depend on how
groups are split across shards or processes, nor on how replicates are chunked.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from teehr_core.metrics.registry import (
    METRICS,
    MetricRequest,
    Transform,
    compute_metrics,
    transformed,
)
from teehr_core.metrics.state import (
    _EXTREMES,
    _SUMS,
    _TIMES,
    ADDITIVE,
    SKETCH,
    STATE_COLUMNS,
    compute_from_state,
)
from teehr_core.metrics.stats import COMPONENTS, FloatArray, IntArray

__all__ = [
    "BootstrapSettings",
    "NotSegmentable",
    "SegmentStats",
    "bootstrap_intervals",
    "group_seed",
    "resample_counts",
    "state_intervals",
]

Method = Literal["iid", "circular_block", "stationary", "year_block"]

#: Upper bound on weights held at once (replicates x pairs); replicates are chunked.
_WEIGHT_BUDGET = 20_000_000


@dataclass(frozen=True)
class BootstrapSettings:
    """How to resample (see the spec's ``bootstrap`` block)."""

    method: Method
    reps: int
    seed: int
    quantiles: Sequence[float]
    min_count: int
    #: year_block: groups with fewer distinct years get NaN intervals.
    min_years: int = 5


def _key_json(value: Any) -> Any:
    """Encode a key value the way the shared bootstrap standard does (beyond plain JSON)."""
    if isinstance(value, datetime):
        utc = value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        text = utc.strftime("%Y-%m-%dT%H:%M:%S")
        if utc.microsecond:
            text += f".{utc.microsecond:06d}".rstrip("0")
        return text + "Z"
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def group_seed(seed: int, key: Sequence[Any]) -> int:
    """Stable seed for one group, from the stage seed and the group's key values.

    ``blake2b`` (8 bytes, little-endian) of the compact JSON ``[seed, *key]``: timestamps
    as ISO 8601 UTC with ``Z`` (fractional seconds only when non-zero), whole floats with
    ``.0``, missing values as ``null`` (``standards/bootstrap.md`` in the shared spec).
    """
    text = json.dumps([seed, *key], default=_key_json, separators=(",", ":"))
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


class _Streams:
    """A group's random streams: block starts, block lengths, and extra blocks.

    The extra blocks are for the rare stationary resample whose first draws don't cover
    the series.
    """

    def __init__(self, seed: int) -> None:
        starts, breaks, more = np.random.SeedSequence(seed).spawn(3)
        self.starts = np.random.default_rng(starts)
        self.breaks = np.random.default_rng(breaks)
        self.more = np.random.default_rng(more)


def _stationary_draws(n: int, block: int) -> int:
    """Blocks drawn per stationary resample: enough to cover ``n`` all but very rarely."""
    mean = n / min(block, n)
    return min(n, int(np.ceil(mean + 6 * np.sqrt(mean) + 6)))


def _blocks(
    streams: _Streams, n: int, block: int, method: Method, reps: int
) -> tuple[IntArray, IntArray, IntArray]:
    """Each resample's blocks of ``n`` units: starts, lengths, and where resamples begin.

    Starts and lengths are flat; replicate ``r``'s blocks are ``offsets[r]:offsets[r + 1]``.

    ``iid`` draws ``n`` blocks of one; ``circular_block`` blocks of ``block``;
    ``stationary`` (Politis & Romano 1994) geometric lengths with mean ``block``. The
    last block is cut to fill ``n``, and blocks wrap around the series. Each resample
    takes a fixed number of draws (stationary ones that fall short then draw more, in
    order, from their own stream), so chunking replicates doesn't change them.
    """
    if method != "stationary":
        b = 1 if method == "iid" else min(block, n)
        k = -(-n // b)
        lengths = np.full(k, b, dtype=np.int64)
        lengths[-1] = n - (k - 1) * b
        starts = streams.starts.integers(0, n, size=(reps, k))
        return starts.ravel(), np.tile(lengths, reps), np.arange(reps + 1) * k
    p = 1.0 / min(block, n)
    m = _stationary_draws(n, block)
    lengths = streams.breaks.geometric(p, size=(reps, m)).astype(np.int64)
    starts = streams.starts.integers(0, n, size=(reps, m))
    begun = np.cumsum(lengths, axis=1) - lengths  # units before each block
    used = begun < n
    lengths = np.minimum(lengths, n - begun)  # the last block is cut to fill n
    short = set(np.flatnonzero(begun[:, -1] + lengths[:, -1] < n).tolist())
    if not short:
        offsets = np.concatenate([[0], np.cumsum(used.sum(axis=1))])
        return starts[used], lengths[used], offsets
    parts_s, parts_l = [], []  # rare: append blocks one at a time until covered
    for r in range(reps):
        s, length = list(starts[r, used[r]]), list(lengths[r, used[r]])
        if r in short:
            total = sum(length)
            while total < n:
                length.append(min(int(streams.more.geometric(p)), n - total))
                s.append(int(streams.more.integers(0, n)))
                total += length[-1]
        parts_s.append(s)
        parts_l.append(length)
    offsets = np.concatenate([[0], np.cumsum([len(s) for s in parts_s])])
    flat = np.concatenate(parts_s).astype(np.int64), np.concatenate(parts_l).astype(np.int64)
    return flat[0], flat[1], offsets


def resample_counts(streams: _Streams, n: int, block: int, method: Method, reps: int) -> FloatArray:
    """How often each of ``n`` ordered pairs is drawn in each of ``reps`` resamples."""
    if n == 0:
        return np.zeros((reps, 0))
    starts, lengths, offsets = _blocks(streams, n, block, method, reps)
    # Mark +1 where each block (or its wrapped part) begins and -1 where it ends.
    row = np.repeat(np.arange(reps) * (n + 1), np.diff(offsets))
    end = starts + lengths
    wrap = end > n
    at = np.concatenate(
        [row + starts, row + np.minimum(end, n), row[wrap], row[wrap] + end[wrap] - n]
    )
    step = np.concatenate(
        [np.ones(starts.size), -np.ones(starts.size), np.ones(wrap.sum()), -np.ones(wrap.sum())]
    )
    marks = np.bincount(at, weights=step, minlength=reps * (n + 1)).reshape(reps, n + 1)
    return np.cumsum(marks, axis=1)[:, :n]


def bootstrap_intervals(
    requests: Sequence[MetricRequest],
    obs: FloatArray,
    pred: FloatArray,
    gid: IntArray,
    group_keys: Sequence[Sequence[Any]],
    block_rows: IntArray,
    settings: BootstrapSettings,
    columns: dict[str, FloatArray] | None = None,
    cases: IntArray | None = None,
    years: IntArray | None = None,
    periods: IntArray | None = None,
) -> dict[MetricRequest, FloatArray]:
    """Bootstrap quantiles of each metric per group: ``(len(quantiles), groups)``.

    Rows must be sorted by group, and within a group in the order blocks follow (e.g.
    by time). ``cases`` assigns rows to forecast cases (e.g. the members of one ensemble
    forecast): whole cases are resampled, in order of first appearance, and
    ``block_rows`` counts cases. Without it every row is a case. Groups with fewer
    than ``min_count`` pairs, and replicates where a metric is undefined, are left out;
    a group with no defined replicates gets NaN.

    With ``year_block``, ``years`` labels each row's (water) year: each replicate draws
    as many whole years as the group spans, with replacement, so every year keeps its
    seasonality; groups spanning fewer than ``min_years`` years get NaN.

    With ``periods`` (each row's calendar day or month, as a number), blocks are made of
    ``block_rows`` consecutive periods, counted from the group's first to its last,
    empty ones included, so a block spans the same time however dense the pairs.
    Resamples of state per period (:func:`teehr_core.metrics.state.state_intervals`) take
    the same draws, so they give the same intervals.
    """
    n_groups = len(group_keys)
    sizes = np.bincount(gid, minlength=n_groups)
    if np.any(np.diff(gid) < 0):
        raise ValueError("rows must be sorted by group")
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    case_of_row = np.arange(obs.size) if cases is None else cases
    by_year = settings.method == "year_block"
    if by_year and years is None:
        raise ValueError("year_block needs each row's year")
    by_period = periods is not None and not by_year
    units = years if by_year and years is not None else case_of_row
    # Per group: each row's unit (case, or year) as a position 0..k-1 in order of first
    # appearance; or its period's offset from the group's first.
    local: list[IntArray] = []
    for g in range(n_groups):
        if by_period:
            assert periods is not None
            part = periods[bounds[g] : bounds[g + 1]]
            local.append(part - part.min() if part.size else part)
            continue
        part = units[bounds[g] : bounds[g + 1]]
        uniq, first, inverse = np.unique(part, return_index=True, return_inverse=True)
        rank = np.empty(uniq.size, dtype=np.int64)
        rank[np.argsort(first)] = np.arange(uniq.size)
        local.append(rank[inverse])
    method: Method = "iid" if by_year else settings.method  # whole years, iid
    pieces = _pieces(local, bounds, block_rows, group_keys, settings, method)
    fast = {} if pieces is None else _segment_replicates(requests, obs, pred, gid, pieces, settings)
    dense = [r for r in requests if r not in fast]

    streams = [_Streams(group_seed(settings.seed, key)) for key in group_keys]
    chunk = max(1, min(settings.reps, _WEIGHT_BUDGET // max(obs.size, 1)))
    draws: dict[MetricRequest, list[FloatArray]] = {r: [fast[r]] for r in fast}
    draws.update({r: [] for r in dense})
    done = 0 if dense else settings.reps
    while done < settings.reps:
        size = min(chunk, settings.reps - done)
        weights = np.zeros((size, obs.size))
        for g in range(n_groups):
            lo, hi = bounds[g], bounds[g + 1]
            n_cases = int(local[g].max()) + 1 if hi > lo else 0
            counts = resample_counts(streams[g], n_cases, int(block_rows[g]), method, size)
            weights[:, lo:hi] = counts[:, local[g]]
        values = compute_metrics(dense, obs, pred, gid, n_groups, weights, columns)
        for r in dense:
            draws[r].append(values[r])
        done += size

    out: dict[MetricRequest, FloatArray] = {}
    units_per_group = np.array([int(r.max()) + 1 if r.size else 0 for r in local])
    if by_year or by_period:
        # Cases with values (as state counts them), so either path leaves out the same groups.
        valued = np.isfinite(obs) & np.isfinite(pred)
        cases_per_group = np.array(
            [
                np.unique(
                    case_of_row[bounds[g] : bounds[g + 1]][valued[bounds[g] : bounds[g + 1]]]
                ).size
                for g in range(n_groups)
            ]
        )
        too_small = cases_per_group < settings.min_count
        if by_year:
            too_small |= units_per_group < settings.min_years
    else:
        too_small = units_per_group < settings.min_count
    for r in requests:
        reps = np.concatenate(draws[r], axis=0)
        all_nan = np.all(np.isnan(reps), axis=0)
        filled = np.where(all_nan[None, :], 0.0, reps)  # avoid all-NaN warnings
        with np.errstate(invalid="ignore"):
            q = np.nanquantile(filled, list(settings.quantiles), axis=0)
        q[:, all_nan | too_small] = np.nan
        out[r] = q
    return out


class NotSegmentable(Exception):
    """A metric needs per-pair weights: it is bootstrapped the dense way."""


@dataclass(frozen=True)
class _Pieces:
    """One group's resamples as row ranges, relative to its first row ``lo``.

    Block ``j`` is rows ``[a1, b1)`` plus, where it wraps around the series, ``[0, b2)``
    (empty otherwise); replicate ``r``'s blocks are ``offsets[r]:offsets[r + 1]``.
    """

    lo: int
    hi: int
    a1: IntArray
    b1: IntArray
    b2: IntArray
    offsets: IntArray


def _pieces(
    local: list[IntArray],
    bounds: IntArray,
    block_rows: IntArray,
    group_keys: Sequence[Sequence[Any]],
    settings: BootstrapSettings,
    method: Method,
) -> dict[int, _Pieces] | None:
    """Each group's resamples as row ranges (None: not this way; use weights).

    Takes the same draws as :func:`resample_counts`. Needs each group's resampled units
    (cases or years) to be contiguous runs of rows, and the ranges to fit the weights'
    memory budget.
    """
    sizes: list[tuple[int, int]] = []  # (units, most blocks a resample draws) per group
    for g, units in enumerate(local):
        if units.size and np.any(np.diff(units) < 0):
            return None  # a unit's rows aren't together
        n = int(units[-1]) + 1 if units.size else 0
        b = 1 if method == "iid" else max(1, min(int(block_rows[g]), n))
        if not n:
            k = 0
        elif method == "stationary":
            k = _stationary_draws(n, b)
        else:
            k = -(-n // b)
        sizes.append((n, k))
    if sum(k for _, k in sizes) * settings.reps * 3 > _WEIGHT_BUDGET:
        return None
    out: dict[int, _Pieces] = {}
    for g, (n, _) in enumerate(sizes):
        if n == 0:
            continue
        streams = _Streams(group_seed(settings.seed, group_keys[g]))
        starts, lengths, offsets = _blocks(streams, n, int(block_rows[g]), method, settings.reps)
        rows = np.concatenate([[0], np.cumsum(np.bincount(local[g], minlength=n))])
        end = starts + lengths
        out[g] = _Pieces(
            int(bounds[g]),
            int(bounds[g + 1]),
            rows[starts],
            rows[np.minimum(end, n)],
            rows[np.maximum(end - n, 0)],
            offsets,
        )
    return out


def _segment_replicates(
    requests: Sequence[MetricRequest],
    obs: FloatArray,
    pred: FloatArray,
    gid: IntArray,
    pieces: dict[int, _Pieces],
    settings: BootstrapSettings,
) -> dict[MetricRequest, FloatArray]:
    """Replicate values ``(reps, groups)`` of the requests :class:`SegmentStats` can do."""
    by_transform: dict[Any, list[MetricRequest]] = {}
    for r in requests:
        by_transform.setdefault(r.transform, []).append(r)
    n_groups = int(gid.max()) + 1 if gid.size else 0
    n_groups = max(n_groups, max(pieces, default=-1) + 1)
    out: dict[MetricRequest, FloatArray] = {}
    for transform, reqs in by_transform.items():
        o, p, keep = transformed(obs, pred, transform)
        stats = SegmentStats(o, p, keep, gid, n_groups, pieces, settings.reps)
        for r in reqs:
            with contextlib.suppress(NotSegmentable):  # it needs the weights
                out[r] = METRICS[r.name].finalize(stats, r.params)  # type: ignore[arg-type]
    return out


class SegmentStats:
    """Group components of resamples made of whole row ranges (block bootstraps).

    Offers finalizers what :class:`~teehr_core.metrics.stats.GroupStats` does for sums and
    extremes of per-pair values: a replicate's sum is a sum of prefix-sum differences,
    its extremes are range queries on a sparse table, so it costs its blocks, not its
    pairs. Centered sums are taken about each group's mean over all its pairs, less the
    replicate's offset from it. Anything else (per-pair weights, columns, order
    statistics) raises :class:`NotSegmentable`. Pairs not in ``keep`` (non-finite after
    the transform) are drawn but count for nothing, as with weights.
    """

    def __init__(
        self,
        obs: FloatArray,
        pred: FloatArray,
        keep: npt.NDArray[np.bool_],
        gid: IntArray,
        n_groups: int,
        pieces: dict[int, _Pieces],
        replicates: int,
    ) -> None:
        self.obs, self.pred, self.gid, self.n_groups = obs, pred, gid, n_groups
        self.replicates = replicates
        self._keep = keep
        self._pieces = pieces
        self._cache: dict[str, FloatArray] = {}
        #: Components in the order they were computed (each appears once).
        self.computed: list[str] = []

    def __getattr__(self, name: str) -> Any:
        raise NotSegmentable(name)  # what GroupStats does with per-pair weights

    def __getitem__(self, name: str) -> FloatArray:
        if name not in self._cache:
            centered = _CENTERED.get(name)
            self._cache[name] = centered(self) if centered else COMPONENTS[name](self)  # type: ignore[arg-type]
            self.computed.append(name)
        return self._cache[name]

    def side(self, side: str) -> FloatArray:
        """Return ``obs`` or ``pred`` values."""
        return self.obs if side == "obs" else self.pred

    def _values(self, x: FloatArray | float, fill: float) -> FloatArray:
        """Return a per-pair array, with ``fill`` at pairs left out."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0:
            x = np.broadcast_to(x, self.obs.shape)
        if x.shape != self.obs.shape:
            raise NotSegmentable("values per replicate")
        return np.where(self._keep, x, fill)

    def gsum(self, x: FloatArray | float) -> FloatArray:
        """Sum per (replicate, group) of a per-pair array, over each replicate's draws."""
        v = self._values(x, 0.0)
        out = np.zeros((self.replicates, self.n_groups))
        for g, p in self._pieces.items():
            prefix = np.concatenate([[0.0], np.cumsum(v[p.lo : p.hi])])
            blocks = prefix[p.b1] - prefix[p.a1] + prefix[p.b2]
            out[:, g] = np.add.reduceat(blocks, p.offsets[:-1])
        return out

    def _extreme(self, x: FloatArray, fn: np.ufunc, empty: float) -> FloatArray:
        v = self._values(x, empty)
        out = np.full((self.replicates, self.n_groups), empty)
        for g, p in self._pieces.items():
            levels = _sparse_table(v[p.lo : p.hi], fn)
            first = _range(levels, p.a1, p.b1, fn, empty)
            wrapped = _range(levels, np.zeros_like(p.b2), p.b2, fn, empty)
            out[:, g] = fn.reduceat(fn(first, wrapped), p.offsets[:-1])
        return np.where(np.isinf(out), np.nan, out)  # no pairs drawn

    def gmin(self, x: FloatArray) -> FloatArray:
        """Minimum per (replicate, group) over the pairs drawn (NaN if none)."""
        return self._extreme(x, np.minimum, np.inf)

    def gmax(self, x: FloatArray) -> FloatArray:
        """Maximum per (replicate, group) over the pairs drawn (NaN if none)."""
        return self._extreme(x, np.maximum, -np.inf)

    def cached(self, key: str, compute: Callable[[], FloatArray]) -> FloatArray:
        """Return a parameterized component, computed once."""
        if key not in self._cache:
            self._cache[key] = compute()
            self.computed.append(key)
        return self._cache[key]


def _sparse_table(v: FloatArray, fn: np.ufunc) -> list[FloatArray]:
    """Level ``j`` holds ``fn`` over each run of ``2**j`` values starting there."""
    levels = [v]
    width = 1
    while 2 * width <= v.size:
        prev = levels[-1]
        levels.append(fn(prev[:-width], prev[width:]))
        width *= 2
    return levels


def _range(
    levels: list[FloatArray], lo: IntArray, hi: IntArray, fn: np.ufunc, empty: float
) -> FloatArray:
    """``fn`` over values ``[lo, hi)`` for each pair of bounds (``empty`` if none)."""
    out = np.full(lo.shape, empty)
    length = hi - lo
    has = length > 0
    j = np.zeros(lo.shape, dtype=np.int64)
    j[has] = np.log2(length[has]).astype(np.int64)
    j -= (np.left_shift(1, j) > length) & has  # guard float rounding at powers of two
    j += (np.left_shift(1, j + 1) <= length) & has
    for level in np.unique(j[has]):
        m = has & (j == level)
        table = levels[level]
        out[m] = fn(table[lo[m]], table[hi[m] - (1 << int(level))])
    return out


def _centered(a: str, b: str) -> Callable[[SegmentStats], FloatArray]:
    """Centered sum of products of ``a`` and ``b`` (squares when they're the same side).

    About each group's mean over all its pairs, ``c``: with ``D = sum(w (x - c))``,
    ``sum(w (x - m)(y - m')) = sum(w (x - c)(y - c')) - D_x D_y / n``.
    """

    def compute(s: SegmentStats) -> FloatArray:
        dev = {}
        for side in {a, b}:
            x = s.side(side)
            kept = np.bincount(s.gid, weights=s._keep.astype(np.float64), minlength=s.n_groups)
            total = np.bincount(s.gid, weights=np.where(s._keep, x, 0.0), minlength=s.n_groups)
            center = np.divide(total, kept, out=np.zeros_like(total), where=kept > 0)
            dev[side] = x - center[s.gid]
        n = s["n"]
        cross, da, db = s.gsum(dev[a] * dev[b]), s.gsum(dev[a]), s.gsum(dev[b])
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(n > 0, cross - da * db / n, 0.0)
        return np.maximum(out, 0.0) if a == b else out

    return compute


_CENTERED = {
    "ss_obs": _centered("obs", "obs"),
    "ss_pred": _centered("pred", "pred"),
    "sp_obs_pred": _centered("obs", "pred"),
}


def state_intervals(
    requests: Sequence[MetricRequest],
    states: dict[Transform | None, dict[str, FloatArray]],
    gid: IntArray,
    unit: IntArray,
    group_keys: Sequence[Sequence[Any]],
    block: int,
    settings: BootstrapSettings,
    cases: IntArray,
) -> dict[MetricRequest, FloatArray]:
    """Bootstrap quantiles per group from state per time unit: ``(len(quantiles), groups)``.

    ``states`` holds, per transform, state rows (:data:`~teehr_core.metrics.state.STATE_COLUMNS`)
    of one group and time unit each, rows of every transform in the same order. ``gid``
    is each row's group, ``unit`` its position in the group: the offset of its day or
    month from the group's first (periods, empty ones counting), or its year's rank
    (``year_block``). ``cases`` is each group's pair count, for ``min_count``.

    Draws are :func:`bootstrap_intervals`' for the same group keys, units and block, and
    a replicate's components are its blocks' states merged, so the intervals are the
    same as from the pairs: sums add, centered sums combine about the group's mean (as
    :func:`~teehr_core.metrics.state.merge_sql`), extremes take the extreme of the units drawn.
    """
    n_groups = len(group_keys)
    by_year = settings.method == "year_block"
    method: Method = "iid" if by_year else settings.method
    reps = settings.reps
    order = np.lexsort((unit, gid))
    gid, unit = gid[order], unit[order]
    bounds = np.searchsorted(gid, np.arange(n_groups + 1))
    # The state columns, and the metrics' additive sums (which add like the other sums).
    names = {
        t: [*STATE_COLUMNS, *(c for c in st if c.startswith(ADDITIVE))] for t, st in states.items()
    }
    sketched = {t: [c for c in st if c.startswith(SKETCH)] for t, st in states.items()}
    merged: dict[Transform | None, dict[str, Any]] = {
        t: {c: np.full((reps, n_groups), np.nan) for c in names[t]} for t in states
    }
    for t in states:
        for c in sketched[t]:  # one (ids, (reps, buckets) counts) per group
            merged[t][c] = [(np.zeros(0, np.int64), np.zeros((reps, 0)))] * n_groups
    span = np.zeros(n_groups, dtype=np.int64)
    for g in range(n_groups):
        lo, hi = int(bounds[g]), int(bounds[g + 1])
        if hi == lo:
            continue
        n = int(unit[hi - 1]) + 1
        span[g] = n
        streams = _Streams(group_seed(settings.seed, group_keys[g]))
        starts, lengths, offsets = _blocks(streams, n, 1 if by_year else block, method, reps)
        for t, st in states.items():
            dense = {c: np.zeros(n) for c in names[t]}
            for c in names[t]:
                dense[c][unit[lo:hi]] = np.asarray(st[c], np.float64)[order[lo:hi]]
            for c, column in _merge_blocks(dense, starts, lengths, offsets).items():
                merged[t][c][:, g] = column
            if sketched[t]:
                draws = _unit_draws(n, starts, lengths, offsets)
                rows = order[lo:hi]
                for c in sketched[t]:
                    merged[t][c][g] = _drawn_sketch([st[c][i] for i in rows], unit[lo:hi], draws)
    values = compute_from_state(requests, merged)
    too_small = np.asarray(cases) < settings.min_count
    if by_year:
        too_small |= span < settings.min_years
    out: dict[MetricRequest, FloatArray] = {}
    for r in requests:
        draws = values[r]
        all_nan = np.all(np.isnan(draws), axis=0)
        filled = np.where(all_nan[None, :], 0.0, draws)
        with np.errstate(invalid="ignore"):
            q = np.nanquantile(filled, list(settings.quantiles), axis=0)
        q[:, all_nan | too_small] = np.nan
        out[r] = q
    return out


def _unit_draws(n: int, starts: IntArray, lengths: IntArray, offsets: IntArray) -> FloatArray:
    """Return how often each replicate draws each of ``n`` units: ``(replicates, n)``.

    Blocks as :func:`_merge_blocks` takes them (wrapping around the ``n`` units).
    """
    reps = offsets.size - 1
    rep = np.repeat(np.arange(reps), np.diff(offsets))
    end = starts + lengths
    a1, b1, b2 = starts, np.minimum(end, n), np.maximum(end - n, 0)
    diff = np.zeros((reps, n + 1))
    np.add.at(diff, (rep, a1), 1.0)
    np.add.at(diff, (rep, b1), -1.0)
    wraps = b2 > 0
    np.add.at(diff, (rep[wraps], np.zeros(int(wraps.sum()), np.int64)), 1.0)
    np.add.at(diff, (rep[wraps], b2[wraps]), -1.0)
    out: FloatArray = np.cumsum(diff[:, :n], axis=1)
    return out


def _drawn_sketch(
    sketches: Sequence[tuple[IntArray, FloatArray]], units: IntArray, draws: FloatArray
) -> tuple[IntArray, FloatArray]:
    """Return each replicate's sketch: the units' sketches, each counted as often as drawn.

    ``sketches[i]`` is unit ``units[i]``'s (bucket ids, ``(1, buckets)`` counts).
    """
    ids = np.unique(np.concatenate([s[0] for s in sketches] or [np.zeros(0, np.int64)]))
    counts = np.zeros((draws.shape[1], ids.size))
    for (b, w), u in zip(sketches, units, strict=True):
        counts[u, np.searchsorted(ids, b)] += w[0]
    return ids, draws @ counts


def _merge_blocks(
    st: dict[str, FloatArray], starts: IntArray, lengths: IntArray, offsets: IntArray
) -> dict[str, FloatArray]:
    """Each replicate's state from one group's state per unit and its blocks of units.

    Blocks are ``[start, start + length)``, wrapping around the ``n`` units; replicate
    ``r``'s blocks are ``offsets[r]:offsets[r + 1]``. Sums are prefix-sum differences and
    extremes range queries, so a replicate costs its blocks, not its units.
    """
    n = st["n"].size
    end = starts + lengths
    a1, b1, b2 = starts, np.minimum(end, n), np.maximum(end - n, 0)
    first = offsets[:-1]

    def total(x: FloatArray) -> FloatArray:
        prefix = np.concatenate([[0.0], np.cumsum(x)])
        return np.add.reduceat(prefix[b1] - prefix[a1] + prefix[b2], first)

    has = st["n"] > 0
    out = {c: total(st[c]) for c in (*_SUMS, *(k for k in st if k.startswith(ADDITIVE)))}
    # Centered sums about the group's mean c over all its units: a unit adds
    # ss + n (m - c)^2, and the replicate's own mean is c + D with D = sum(n (m - c)) / N.
    n_all = st["n"].sum()
    dev = {}
    for side in ("obs", "pred"):
        c = st[f"sum_{side}"].sum() / n_all if n_all else 0.0
        mean = np.divide(st[f"sum_{side}"], st["n"], out=np.zeros(n), where=has)
        dev[side] = np.where(has, mean - c, 0.0)
    n_r = out["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        shift = {s: np.where(n_r > 0, total(st["n"] * dev[s]) / n_r, 0.0) for s in dev}
        for name, a, b in (
            ("ss_obs", "obs", "obs"),
            ("ss_pred", "pred", "pred"),
            ("sp_obs_pred", "obs", "pred"),
        ):
            part = np.where(has, st[name], 0.0) + st["n"] * dev[a] * dev[b]
            value = np.where(n_r > 0, total(part) - n_r * shift[a] * shift[b], 0.0)
            out[name] = np.maximum(value, 0.0) if a == b else value
    for name in _EXTREMES:
        fn, empty = (np.minimum, np.inf) if name.startswith("min") else (np.maximum, -np.inf)
        levels = _sparse_table(np.where(has, st[name], empty), fn)
        both = fn(
            _range(levels, a1, b1, fn, empty), _range(levels, np.zeros_like(b2), b2, fn, empty)
        )
        r = fn.reduceat(both, first)
        out[name] = np.where(np.isinf(r), np.nan, r)
    for name in _TIMES:  # when a replicate's maximum occurs: not kept (a blocker)
        out[name] = np.full(first.size, np.nan)
    return out
