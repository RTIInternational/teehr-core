"""Shared group statistics: the reusable components metrics are built from.

Ported from VeriRace (``verirace/metrics/stats.py`` at 668ab7d).

Metrics are small formulas ("finalizers") over named **components** such as the pair
count, centered sums of squares or sorted values. :class:`GroupStats` computes each
component at most once, on first use, for every group at once, so a stage that asks for
RMSE, NSE, KGE and R² computes the shared sums once.

Every component is a **weighted** group reduction. Normally each pair has weight 1. For
a bootstrap, each replicate gives each pair a weight equal to how often the resample
drew it, so one ``GroupStats`` evaluates many replicates at once and the same
finalizers produce both the point value and the bootstrap distribution. Arrays of
results have shape ``(replicates, groups)``; replicates is 1 without a bootstrap.

Conventions:

- NaN means the metric is undefined for that group (callers turn it into null).
- Variances use two-pass centered sums (means first, then squared deviations).
- "Constant" means the spread (max - min) is within ``CONSTANT_RTOL`` of the values'
  magnitude: differences at the level of floating-point rounding carry no information,
  and metrics computed from them are noise, so those results are undefined.
- Standard deviations are population standard deviations (divide by n).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from teehr_core.metrics.sketch import snap

__all__ = ["COMPONENTS", "CONSTANT_RTOL", "FloatArray", "GroupStats", "IntArray"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

#: Relative spread at or below which a group's values count as constant.
CONSTANT_RTOL = 16 * float(np.finfo(np.float64).eps)

Component = Callable[["GroupStats"], FloatArray]

#: Registered components by name.
COMPONENTS: dict[str, Component] = {}


def component(fn: Component) -> Component:
    """Register a component under its function name."""
    COMPONENTS[fn.__name__] = fn
    return fn


class GroupStats:
    """Lazily computed, cached components for grouped (optionally weighted) pairs.

    Parameters
    ----------
    obs, pred
        Paired values, one entry per pair (non-finite pairs already removed).
    gid
        Group id per pair, ``0 .. n_groups - 1``.
    n_groups
        Number of groups (groups without pairs give undefined results).
    weights
        ``(replicates, pairs)`` weights, e.g. bootstrap resample counts; ``None`` means
        one replicate with every weight 1.
    columns
        Other per-pair values some metrics need, e.g. ``value_time`` (epoch seconds) or
        a threshold column.
    """

    def __init__(
        self,
        obs: FloatArray,
        pred: FloatArray,
        gid: IntArray,
        n_groups: int,
        weights: FloatArray | None = None,
        columns: dict[str, FloatArray] | None = None,
    ) -> None:
        # One replicate of weight 1: a read-only view, not an array the size of the pairs.
        weights = weights if weights is not None else np.broadcast_to(1.0, (1, obs.size))
        columns = dict(columns or {})
        if gid.size and np.any(gid[1:] < gid[:-1]):  # group pairs contiguously
            order = np.argsort(gid, kind="stable")
            obs, pred, gid, weights = obs[order], pred[order], gid[order], weights[:, order]
            columns = {k: v[order] for k, v in columns.items()}
        self.columns = columns
        self.obs, self.pred, self.gid, self.n_groups = obs, pred, gid, n_groups
        self.weights = weights
        self.replicates = weights.shape[0]
        sizes = np.bincount(gid, minlength=n_groups)
        self._nonempty = sizes > 0
        self._starts = np.searchsorted(gid, np.arange(n_groups))[self._nonempty]
        self._cache: dict[str, FloatArray] = {}
        self._sorts: dict[str, tuple[IntArray, FloatArray, FloatArray]] = {}
        self._snapped: dict[str, FloatArray] = {}
        self._case_info: tuple[IntArray, IntArray] | None = None
        #: Components in the order they were computed (each appears once).
        self.computed: list[str] = []

    def __getitem__(self, name: str) -> FloatArray:
        if name not in self._cache:
            self._cache[name] = COMPONENTS[name](self)
            self.computed.append(name)
        return self._cache[name]

    def cached(self, key: str, compute: Callable[[], FloatArray]) -> FloatArray:
        """Return a parameterized component (e.g. hits at a threshold), computed once."""
        if key not in self._cache:
            self._cache[key] = compute()
            self.computed.append(key)
        return self._cache[key]

    def additive(
        self, keys: tuple[str, ...], compute: Callable[[], tuple[FloatArray, ...]]
    ) -> tuple[FloatArray, ...]:
        """Return group sums named ``keys`` (one ``(replicates, groups)`` array each).

        They are sums over pairs (e.g. hits at a fixed threshold, or a per-case score and
        the cases counted), so parts of a group's pairs add up to the group's: mergeable
        state keeps them (:mod:`teehr_core.metrics.state`). ``keys`` must name everything
        the sums depend on (the metric and its parameters). Computed once.
        """
        if any(k not in self._cache for k in keys):
            for k, v in zip(keys, compute(), strict=True):
                self._cache[k] = v
                self.computed.append(k)
        return tuple(self._cache[k] for k in keys)

    def column(self, name: str) -> FloatArray:
        """Return a per-pair column passed in (e.g. ``value_time`` in epoch seconds)."""
        if name not in self.columns:
            raise KeyError(f"metric needs pair column {name!r}")
        return self.columns[name]

    def side(self, side: str) -> FloatArray:
        """Return ``obs`` or ``pred`` values."""
        return self.obs if side == "obs" else self.pred

    def time_of_max(self, side: str) -> FloatArray:
        """Return when ``obs``/``pred`` first reaches its group maximum (epoch seconds).

        One value per (replicate, group).
        """
        values = self.side(side)
        peak = self.per_pair(self[f"max_{side}"])
        at_peak = (self.weights > 0) & (values == peak)
        t = np.where(at_peak, self.column("value_time"), np.inf)
        out = self._reduce(np.minimum, np.broadcast_to(t, self.weights.shape), np.inf)
        return np.where(np.isinf(out), np.nan, out)

    # -- forecast cases ------------------------------------------------------------
    def cases(self) -> tuple[IntArray, IntArray]:
        """Case id per pair (``0 .. cases - 1``) and the number of pairs in each case.

        A case is one forecast: the pairs sharing a ``_case`` key (all members or
        quantiles of one forecast for one time) within a group. Without a ``_case``
        column every pair is its own case.
        """
        if self._case_info is None:
            raw = self.columns.get("_case")
            if raw is None:
                ids = np.arange(self.obs.size, dtype=np.int64)
            else:
                order = np.lexsort((raw, self.gid))
                key_g, key_r = self.gid[order], raw[order]
                new = np.ones(order.size, dtype=bool)
                new[1:] = (key_g[1:] != key_g[:-1]) | (key_r[1:] != key_r[:-1])
                ids = np.empty(order.size, dtype=np.int64)
                ids[order] = np.cumsum(new) - 1
            self._case_info = (ids, np.bincount(ids))
        return self._case_info

    def case_obs(self) -> FloatArray:
        """Return the observation of each case."""
        ids, sizes = self.cases()
        return np.bincount(ids, weights=self.obs) / sizes

    def case_mean(self, key: str, case_score: Callable[[], FloatArray]) -> FloatArray:
        """Group mean of a per-case score over cases, per (replicate, group).

        Each case counts once (weighted by its resample weight, which is the same for
        all its pairs); cases with a NaN score are left out. ``case_score`` returns one
        value per case, or ``(replicates, cases)``. The score's sum and the cases counted
        are :meth:`additive` (``key`` names the score), so the mean merges from state.
        """
        num, den = self.additive(
            (f"{key}|sum", f"{key}|cases"), lambda: self._case_sums(case_score())
        )
        return _ratio(num, den)

    def mean_over_cases(self, case_score: FloatArray) -> FloatArray:
        """Return :meth:`case_mean` of a score that isn't an additive sum (needs the pairs)."""
        return _ratio(*self._case_sums(case_score))

    def _case_sums(self, case_score: FloatArray) -> tuple[FloatArray, FloatArray]:
        ids, sizes = self.cases()
        row = case_score[..., ids] / sizes[ids]
        valid = np.isfinite(row)
        num = self.gsum(np.where(valid, row, 0.0))
        return num, self.gsum(np.where(valid, 1.0 / sizes[ids], 0.0))

    def crps_cases(self, fair: bool) -> FloatArray:
        """CRPS of each case's ensemble (its predictions) against its observation.

        ``mean|x - y| - sum|x_i - x_j| / (2 m^2)``, or ``/ (2 m (m - 1))`` for the fair
        estimator; a one-member case gives ``|x - y|``.
        """
        ids, sizes = self.cases()
        order = np.lexsort((self.pred, ids))
        xs, cid = self.pred[order], ids[order]
        starts = np.searchsorted(cid, np.arange(sizes.size))
        rank = np.arange(xs.size) - starts[cid] + 1
        m = sizes[cid]
        spread = 2 * np.bincount(cid, weights=(2 * rank - m - 1) * xs, minlength=sizes.size)
        abs_err = np.bincount(ids, weights=np.abs(self.pred - self.obs), minlength=sizes.size)
        mf = sizes.astype(np.float64)
        den = 2 * mf * (mf - 1) if fair else 2 * mf * mf
        with np.errstate(invalid="ignore", divide="ignore"):
            second = np.where(sizes > 1, spread / den, 0.0)
        return abs_err / mf - second

    def case_moments(self) -> tuple[FloatArray, FloatArray, IntArray]:
        """Return each case's ensemble mean, member variance and member count.

        The variance uses an ``m - 1`` denominator (NaN for a one-member case).
        """
        ids, sizes = self.cases()
        mean = np.bincount(ids, weights=self.pred, minlength=sizes.size) / sizes
        dev = self.pred - mean[ids]
        ss = np.bincount(ids, weights=dev * dev, minlength=sizes.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.where(sizes > 1, ss / (sizes - 1), np.nan)
        return mean, var, sizes

    def case_ranks(self) -> tuple[FloatArray, IntArray]:
        """Each case's observation rank among its members (0: below all, m: above all).

        Ties count half, rounded down (members below + floor(members equal / 2)), so ranks
        are deterministic.
        """
        ids, sizes = self.cases()
        y = self.case_obs()[ids]
        below = np.bincount(ids, weights=(self.pred < y).astype(float), minlength=sizes.size)
        equal = np.bincount(ids, weights=(self.pred == y).astype(float), minlength=sizes.size)
        return below + np.floor(equal / 2), sizes

    def regroup(self, sub_gid: IntArray, n_sub: int) -> GroupStats:
        """Return the same weighted pairs grouped more finely (e.g. group x year)."""
        return GroupStats(self.obs, self.pred, sub_gid, n_sub, self.weights, self.columns)

    def to_parent(self, sub_values: FloatArray, parent: IntArray) -> FloatArray:
        """Sum ``(replicates, subgroups)`` values into this object's groups (NaN as 0)."""
        out = np.zeros((self.replicates, self.n_groups))
        values = np.nan_to_num(sub_values)
        for r in range(self.replicates):
            out[r] = np.bincount(parent, weights=values[r], minlength=self.n_groups)
        return out

    # -- reductions ----------------------------------------------------------------
    def _reduce(self, ufunc: np.ufunc, values: FloatArray, empty: float) -> FloatArray:
        out = np.full((self.replicates, self.n_groups), empty)
        if self._starts.size:
            out[:, self._nonempty] = ufunc.reduceat(values, self._starts, axis=1)
        return out

    def gsum(self, x: FloatArray) -> FloatArray:
        """Weighted sum per (replicate, group) of a per-pair array (1-D or 2-D)."""
        values = np.broadcast_to(self.weights * x, (self.replicates, self.obs.size))
        return self._reduce(np.add, values, 0.0)

    def gmin(self, x: FloatArray) -> FloatArray:
        """Minimum per (replicate, group) over pairs with positive weight (NaN if none)."""
        out = self._reduce(np.minimum, np.where(self.weights > 0, x, np.inf), np.inf)
        return np.where(np.isinf(out), np.nan, out)  # no pairs with positive weight

    def gmax(self, x: FloatArray) -> FloatArray:
        """Maximum per (replicate, group) over pairs with positive weight (NaN if none)."""
        out = self._reduce(np.maximum, np.where(self.weights > 0, x, -np.inf), -np.inf)
        return np.where(np.isinf(out), np.nan, out)  # no pairs with positive weight

    def per_pair(self, x: FloatArray) -> FloatArray:
        """Broadcast a per-(replicate, group) array back to pairs: ``(replicates, pairs)``."""
        return x[:, self.gid]

    # -- order statistics ----------------------------------------------------------
    def _values(self, side: str, snapped: bool = False) -> FloatArray:
        """Return ``obs`` or ``pred``, or (``snapped``) their quantile sketch buckets' values."""
        values = self.obs if side == "obs" else self.pred
        if not snapped:
            return values
        key = f"_snapped_{side}"
        if key not in self._snapped:
            self._snapped[key] = snap(values)
        return self._snapped[key]

    def _sorted(self, side: str, snapped: bool = False) -> tuple[IntArray, FloatArray, FloatArray]:
        """Sort pairs by (group, value) for ``obs`` or ``pred`` (or their snapped values).

        Returns the order, the cumulative weight before each group's first pair
        (``(replicates, groups)``), and cumulative weights along the sorted order
        (``(replicates, pairs)``).
        """
        key = f"_sorted_{side}{'_snapped' if snapped else ''}"
        if key not in self._sorts:
            values = self._values(side, snapped)
            order = np.lexsort((values, self.gid))
            starts = np.searchsorted(self.gid[order], np.arange(self.n_groups))
            cum = np.cumsum(self.weights[:, order], axis=1)
            padded = np.concatenate([np.zeros((self.replicates, 1)), cum], axis=1)
            self._sorts[key] = (order, padded[:, starts], cum)
        return self._sorts[key]

    def weighted_quantile(
        self, side: str, q: float, upper: bool = False, snapped: bool = False
    ) -> FloatArray:
        """Order statistic at weighted position ``q * n`` per (replicate, group).

        With ``upper`` False this is the first value whose cumulative weight reaches the
        position; with ``upper`` True the first value whose cumulative weight exceeds it.
        ``q = 0`` with ``upper`` gives the minimum; ``q = 1`` gives the maximum; the
        median is the mean of the two at ``q = 0.5``.
        """
        order, before, cum = self._sorted(side, snapped)
        values = self._values(side, snapped)[order]
        n = self["n"]
        # Round so float noise (1 - 0.33 = 0.66999...) can't shift the order statistic.
        target = before + np.round(q * n, 9)
        out = np.full((self.replicates, self.n_groups), np.nan)
        for r in range(self.replicates):
            pos = np.searchsorted(cum[r], target[r], side="right" if upper else "left")
            pos = np.minimum(pos, values.size - 1)
            out[r] = values[pos]
        out[n == 0] = np.nan
        return out

    def quantile(self, side: str, q: float, snapped: bool = False) -> FloatArray:
        """Return quantile ``q`` of ``obs``/``pred`` per (replicate, group).

        The mean of the two order statistics around position ``q * n`` (Hyndman-Fan
        type 2, numpy's ``averaged_inverted_cdf``); the median is ``q = 0.5``.
        """
        lower = self.weighted_quantile(side, q, snapped=snapped)
        return (lower + self.weighted_quantile(side, q, upper=True, snapped=snapped)) / 2

    def sketch_quantile(self, side: str, q: float) -> FloatArray:
        """Return quantile ``q`` as a merged quantile sketch gives it (metrics.sketch).

        The quantile of the values snapped to their sketch buckets: within 0.1% of the
        exact one, and the same from the pairs, from state, or from a bootstrap from state.
        """
        return self.quantile(side, q, snapped=True)

    def weighted_ranks(self, side: str) -> FloatArray:
        """Rank ``obs`` or ``pred`` within each group, counting each pair ``weight`` times.

        Ranks are 1-based and ties share their mean rank; the result is
        ``(replicates, pairs)``.

        This equals ranking the resampled data, so Spearman correlation of a bootstrap
        replicate is a weighted Pearson correlation of these ranks.
        """
        order, before_group, cum = self._sorted(side)
        values = (self.obs if side == "obs" else self.pred)[order]
        gid = self.gid[order]
        # Runs of equal (group, value) are ties.
        new_run = np.ones(values.size, dtype=bool)
        new_run[1:] = (values[1:] != values[:-1]) | (gid[1:] != gid[:-1])
        run_start = np.flatnonzero(new_run)
        run_id = np.cumsum(new_run) - 1
        w = self.weights[:, order]
        run_weight = np.add.reduceat(w, run_start, axis=1)
        run_end_cum = cum[:, np.r_[run_start[1:] - 1, values.size - 1]]
        below = run_end_cum - run_weight - before_group[:, gid[run_start]]
        rank_sorted = (below + (run_weight + 1) / 2)[:, run_id]
        ranks = np.empty_like(rank_sorted)
        ranks[:, order] = rank_sorted
        return ranks


def _ratio(num: FloatArray, den: FloatArray, undefined: BoolArray | None = None) -> FloatArray:
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    bad = den == 0
    if undefined is not None:
        bad = bad | undefined
    return np.where(bad, np.nan, out)


# -- components -----------------------------------------------------------------------
@component
def n(s: GroupStats) -> FloatArray:
    """Count pairs (weighted) per group."""
    return s.gsum(np.ones_like(s.obs))


@component
def sum_obs(s: GroupStats) -> FloatArray:
    """Sum of observations."""
    return s.gsum(s.obs)


@component
def sum_pred(s: GroupStats) -> FloatArray:
    """Sum of predictions."""
    return s.gsum(s.pred)


@component
def mean_obs(s: GroupStats) -> FloatArray:
    """Mean observation."""
    return _ratio(s["sum_obs"], s["n"])


@component
def mean_pred(s: GroupStats) -> FloatArray:
    """Mean prediction."""
    return _ratio(s["sum_pred"], s["n"])


@component
def ss_obs(s: GroupStats) -> FloatArray:
    """Centered sum of squares of observations: sum((obs - mean_obs)^2)."""
    return s.gsum((s.obs - s.per_pair(s["mean_obs"])) ** 2)


@component
def ss_pred(s: GroupStats) -> FloatArray:
    """Centered sum of squares of predictions."""
    return s.gsum((s.pred - s.per_pair(s["mean_pred"])) ** 2)


@component
def sp_obs_pred(s: GroupStats) -> FloatArray:
    """Centered sum of cross products: sum((obs - mean_obs)(pred - mean_pred))."""
    dev_o = s.obs - s.per_pair(s["mean_obs"])
    dev_p = s.pred - s.per_pair(s["mean_pred"])
    return s.gsum(dev_o * dev_p)


@component
def std_obs(s: GroupStats) -> FloatArray:
    """Return the population standard deviation of observations."""
    return np.sqrt(_ratio(s["ss_obs"], s["n"]))


@component
def std_pred(s: GroupStats) -> FloatArray:
    """Return the population standard deviation of predictions."""
    return np.sqrt(_ratio(s["ss_pred"], s["n"]))


@component
def sum_error(s: GroupStats) -> FloatArray:
    """Sum of errors: sum(pred - obs)."""
    return s.gsum(s.pred - s.obs)


@component
def sum_abs_error(s: GroupStats) -> FloatArray:
    """Sum of absolute errors."""
    return s.gsum(np.abs(s.pred - s.obs))


@component
def sum_sq_error(s: GroupStats) -> FloatArray:
    """Sum of squared errors."""
    return s.gsum((s.pred - s.obs) ** 2)


@component
def n_obs_nonzero(s: GroupStats) -> FloatArray:
    """Pairs with a non-zero observation (the pairs percentage errors can use)."""
    return s.gsum((s.obs != 0).astype(np.float64))


@component
def sum_abs_pct_error(s: GroupStats) -> FloatArray:
    """Sum of |pred - obs| / |obs| over pairs with non-zero observations."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ape = np.where(s.obs != 0, np.abs(s.pred - s.obs) / np.abs(s.obs), 0.0)
    return s.gsum(ape)


@component
def sum_sym_pct_error(s: GroupStats) -> FloatArray:
    """Sum of 2|pred - obs| / (|obs| + |pred|); a pair where both are 0 contributes 0."""
    den = np.abs(s.obs) + np.abs(s.pred)
    with np.errstate(invalid="ignore", divide="ignore"):
        sape = np.where(den != 0, 2 * np.abs(s.pred - s.obs) / den, 0.0)
    return s.gsum(sape)


@component
def min_obs(s: GroupStats) -> FloatArray:
    """Minimum observation."""
    return s.gmin(s.obs)


@component
def max_obs(s: GroupStats) -> FloatArray:
    """Maximum observation."""
    return s.gmax(s.obs)


@component
def min_pred(s: GroupStats) -> FloatArray:
    """Minimum prediction."""
    return s.gmin(s.pred)


@component
def max_pred(s: GroupStats) -> FloatArray:
    """Maximum prediction."""
    return s.gmax(s.pred)


@component
def median_obs(s: GroupStats) -> FloatArray:
    """Median observation (mean of the two middle values for an even count)."""
    return s.quantile("obs", 0.5)


@component
def median_pred(s: GroupStats) -> FloatArray:
    """Median prediction."""
    return s.quantile("pred", 0.5)


def _constant(lo: FloatArray, hi: FloatArray) -> BoolArray:
    scale = np.maximum(np.abs(lo), np.abs(hi))
    result: BoolArray = ~((hi - lo) > CONSTANT_RTOL * scale)
    return result


@component
def obs_constant(s: GroupStats) -> FloatArray:
    """1 where observations are constant (or missing), else 0."""
    return _constant(s["min_obs"], s["max_obs"]).astype(np.float64)


@component
def pred_constant(s: GroupStats) -> FloatArray:
    """1 where predictions are constant (or missing), else 0."""
    return _constant(s["min_pred"], s["max_pred"]).astype(np.float64)


@component
def pearson_r(s: GroupStats) -> FloatArray:
    """Pearson correlation; undefined if either side is constant."""
    undefined = (s["obs_constant"] > 0) | (s["pred_constant"] > 0)
    with np.errstate(invalid="ignore"):
        den = np.sqrt(s["ss_obs"] * s["ss_pred"])
    return np.clip(_ratio(s["sp_obs_pred"], den, undefined), -1.0, 1.0)


@component
def spearman_r(s: GroupStats) -> FloatArray:
    """Spearman rank correlation (Pearson correlation of average ranks)."""
    ro, rp = s.weighted_ranks("obs"), s.weighted_ranks("pred")
    mo, mp = _ratio(s.gsum(ro), s["n"]), _ratio(s.gsum(rp), s["n"])
    do, dp = ro - s.per_pair(mo), rp - s.per_pair(mp)
    undefined = (s["obs_constant"] > 0) | (s["pred_constant"] > 0)
    with np.errstate(invalid="ignore"):
        den = np.sqrt(s.gsum(do**2) * s.gsum(dp**2))
    return np.clip(_ratio(s.gsum(do * dp), den, undefined), -1.0, 1.0)
