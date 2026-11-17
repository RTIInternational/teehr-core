"""Mergeable group state: metric components that parts of a group's pairs combine into.

Ported from VeriRace (``verirace/metrics/state.py`` at 668ab7d).

Most metrics are formulas over a few components (:mod:`teehr_core.metrics.stats`): counts,
sums, centered sums of squares and products, minima and maxima. Computed over parts of a
group's pairs (e.g. one calendar month each), those components combine into the
group's without the pairs: sums add, extremes take the extreme, and centered sums add
with a correction for each part's mean against the group's (the parallel form of the
two-pass formula, so no precision is lost to ``sum(x^2) - n * mean^2``).

Metrics also keep **additive sums** of their own (:meth:`GroupStats.additive`): counts at
a fixed threshold (contingency metrics), and the sum of a per-case score with the number
of cases (CRPS, Brier, interval scores, ensemble spread, rank frequencies, pinball loss,
MASE). They are extra state columns named ``+<key>`` (:func:`additive_columns`), which
add like the other sums, so those metrics merge exactly too.

Quantile metrics (``relative_median``, ``fdc_slope``) keep **quantile sketches**: each
group's count per sketch bucket of obs or pred (columns ``~obs`` and ``~pred``, lists of
``{b, w}``), which merge by adding counts per bucket (:mod:`teehr_core.metrics.sketch`).

An incremental run keeps one state row per group and month, recomputes only the months
whose inputs changed, and merges all months into the group's components (:func:`merge_sql`).
A metric whose formula needs anything else (the pairs themselves: medians, quantiles,
ranks, thresholds from the group's own observations) can't be computed from state;
:func:`mergeable` tells, by trying it on state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pyarrow as pa

from teehr_core.metrics.registry import METRICS, MetricRequest, Transform, _stats
from teehr_core.metrics.sketch import sketch, sketch_quantiles
from teehr_core.metrics.stats import COMPONENTS, FloatArray, GroupStats, IntArray

__all__ = [
    "ADDITIVE",
    "SKETCH",
    "SKETCH_TYPE",
    "STATE_COLUMNS",
    "NotMergeable",
    "StateStats",
    "additive_columns",
    "compute_from_state",
    "group_state",
    "merge_sql",
    "mergeable",
]

#: Components that add across parts.
_SUMS = (
    "n",
    "sum_obs",
    "sum_pred",
    "sum_error",
    "sum_abs_error",
    "sum_sq_error",
    "n_obs_nonzero",
    "sum_abs_pct_error",
    "sum_sym_pct_error",
)
#: Centered sums, combined with each part's mean deviation (see :func:`merge_sql`).
_CENTERED = ("ss_obs", "ss_pred", "sp_obs_pred")
_EXTREMES = ("min_obs", "max_obs", "min_pred", "max_pred")
#: When each side first reaches its maximum (epoch seconds; NaN without value times).
_TIMES = ("time_max_obs", "time_max_pred")
#: A state row's columns, besides the group keys (and the metrics' additive sums).
STATE_COLUMNS = (*_SUMS, *_CENTERED, *_EXTREMES, *_TIMES)
#: Prefix of the state columns holding metrics' additive sums.
ADDITIVE = "+"
#: Prefix of the state columns holding quantile sketches (``~obs``, ``~pred``).
SKETCH = "~"
#: A sketch column's type: (bucket id, count) per bucket, in bucket order.
SKETCH_TYPE = pa.list_(pa.struct([("b", pa.int64()), ("w", pa.float64())]))
#: A group's sketch: bucket ids and their counts ``(replicates, buckets)``.
Sketch = tuple[IntArray, FloatArray]


class NotMergeable(Exception):
    """A metric needs more than mergeable state (e.g. the pairs themselves)."""


def group_state(
    obs: FloatArray,
    pred: FloatArray,
    gid: IntArray,
    n_groups: int,
    value_time: FloatArray | None = None,
    transform: Transform | None = None,
    requests: Sequence[MetricRequest] = (),
    columns: dict[str, FloatArray] | None = None,
) -> dict[str, Any]:
    """Return each group's state: one array per :data:`STATE_COLUMNS` entry.

    Pairs are filtered as metrics filter them (non-finite values dropped, after the
    ``transform``). ``value_time`` (epoch seconds) gives the times of the maxima. The
    additive sums ``requests`` keep (:func:`additive_columns`) come too; ``columns``
    holds the pair columns they read (e.g. ``_case``, ``quantile``).
    """
    cols = dict(columns or {})
    if value_time is not None:
        cols["value_time"] = value_time
    s = _stats(obs, pred, gid, n_groups, transform, None, cols or None)
    out = {name: s[name][0] for name in (*_SUMS, *_CENTERED, *_EXTREMES)}
    for side in ("obs", "pred"):
        out[f"time_max_{side}"] = (
            s.time_of_max(side)[0] if value_time is not None else np.full(n_groups, np.nan)
        )
    for column in additive_columns(requests):
        if column.startswith(SKETCH):  # one list of {b, w} per group
            values = s.obs if column == SKETCH + "obs" else s.pred
            out[column] = [
                [{"b": int(b), "w": float(w)} for b, w in zip(ids, counts, strict=True)]
                for ids, counts in sketch(values, s.gid, n_groups)
            ]
        else:
            out[column] = _additive_sum(s, requests, column.removeprefix(ADDITIVE))
    return out


def sketches_from(column: pa.ChunkedArray | pa.Array | Sequence[Any]) -> list[Sketch]:
    """Return a sketch column's sketches (one replicate each); NULL is an empty one."""
    values = column.to_pylist() if hasattr(column, "to_pylist") else list(column)
    out: list[Sketch] = []
    for entries in values:
        entries = entries or []
        ids = np.array([e["b"] for e in entries], np.int64)
        out.append((ids, np.array([[e["w"] for e in entries]], np.float64).reshape(1, -1)))
    return out


def _additive_sum(s: GroupStats, requests: Sequence[MetricRequest], key: str) -> FloatArray:
    """Return additive sum ``key`` per group, computing the metrics that make it."""
    if key not in s.computed:
        with np.errstate(all="ignore"):
            for r in requests:
                METRICS[r.name].finalize(s, r.params)
                if key in s.computed:
                    break
    value: FloatArray = np.asarray(s.cached(key, lambda: np.zeros((1, s.n_groups))))[0]
    return value


def additive_columns(requests: Sequence[MetricRequest]) -> list[str]:
    """Return the state columns mergeable ``requests`` keep besides STATE_COLUMNS, sorted.

    Additive sums (``+<key>``) and quantile sketches (``~obs``, ``~pred``). Found by
    computing the metrics on probe state, which records each one they ask for. Requests
    that can't be computed from state keep none.
    """
    keys: set[str] = set()
    for r in requests:
        probe = StateStats(_probe_state(), probe=True)
        try:
            with np.errstate(all="ignore"):
                METRICS[r.name].finalize(probe, r.params)  # type: ignore[arg-type]
        except NotMergeable:
            continue
        keys |= probe.asked
    return sorted(keys)


def _probe_state() -> dict[str, FloatArray]:
    pairs = np.array([1.0, 2.0, 4.0]), np.array([1.5, 1.0, 5.0])
    return group_state(*pairs, np.zeros(3, np.int64), 1, np.array([0.0, 3600.0, 7200.0]))


def merge_sql(keys: Sequence[str], source: str, additive: Sequence[str] = ()) -> str:
    """Return SQL merging the state rows in ``source`` into one row per ``keys`` group.

    With ``N``, ``M`` a group's count and mean, and ``n_i``, ``m_i``, ``ss_i`` a part's:
    ``SS = sum(ss_i) + sum(n_i * (m_i - M)^2)``, and likewise for the cross product.
    Parts without pairs add nothing; a group with none has zero sums and no extremes.
    ``additive`` names the metrics' own columns (:func:`additive_columns`): additive sums
    add, and quantile sketches add their counts per bucket.
    """
    sketches = [c for c in additive if c.startswith(SKETCH)]
    additive = [c for c in additive if not c.startswith(SKETCH)]
    q = [f'"{k}"' for k in keys]
    by = ", ".join(q)
    on = " AND ".join(f"t.{k} IS NOT DISTINCT FROM g.{k}" for k in q) or "TRUE"
    sums = ", ".join(
        [f"sum({c}) AS {c}" for c in _SUMS]
        + [f'coalesce(sum("{c}"), 0) AS "{c}"' for c in additive]
    )
    # Parts without pairs have NaN extremes, which DuckDB orders above every number.
    extremes = ", ".join(
        f"{'min' if c.startswith('min') else 'max'}({c}) FILTER (WHERE n > 0) AS {c}"
        for c in _EXTREMES
    )
    mean = {side: f"(t.sum_{side} / t.n - g.sum_{side} / g.n)" for side in ("obs", "pred")}
    centered = (
        f"coalesce(sum(t.ss_obs + t.n * {mean['obs']} ** 2) FILTER (WHERE t.n > 0), 0) AS ss_obs, "
        f"coalesce(sum(t.ss_pred + t.n * {mean['pred']} ** 2) FILTER (WHERE t.n > 0), 0) "
        "AS ss_pred, "
        f"coalesce(sum(t.sp_obs_pred + t.n * {mean['obs']} * {mean['pred']}) "
        "FILTER (WHERE t.n > 0), 0) AS sp_obs_pred"
    )
    times = ", ".join(
        f"min(t.time_max_{side}) FILTER (WHERE t.n > 0 AND t.max_{side} = g.max_{side}) "
        f"AS time_max_{side}"
        for side in ("obs", "pred")
    )
    group_cols = ", ".join(f"g.{k}" for k in q)
    first = f"{by}, " if keys else ""
    gsel = ", ".join(f'g."{c}"' for c in (*_SUMS, *additive, *_EXTREMES))
    merged = (
        f"WITH g AS (SELECT {first}{sums}, {extremes} FROM {source}"
        + (f" GROUP BY {by}" if keys else "")
        + f") SELECT {group_cols + ', ' if keys else ''}{gsel}, {centered}, {times} "
        f"FROM {source} t JOIN g ON {on} GROUP BY ALL"
    )
    if not sketches:
        return merged
    joins, cols = [], []
    for i, c in enumerate(sketches):
        counts = (
            f"SELECT {first}e.b AS b, sum(e.w) AS w "
            f'FROM (SELECT {first}unnest("{c}") AS e FROM {source}) GROUP BY ALL'
        )
        lists = f"SELECT {first}list({{'b': b, 'w': w}} ORDER BY b) AS \"{c}\" FROM ({counts})"
        lists += f" GROUP BY {by}" if keys else ""
        match = " AND ".join(f"m.{k} IS NOT DISTINCT FROM s{i}.{k}" for k in q) or "TRUE"
        joins.append(f"LEFT JOIN ({lists}) s{i} ON {match}")
        cols.append(f's{i}."{c}"')
    return f"SELECT m.*, {', '.join(cols)} FROM ({merged}) m {' '.join(joins)}"


class StateStats:
    """Components from merged state, with :class:`GroupStats`' interface for finalizers.

    Stored components are served as they are; others are derived with their usual
    formulas (mean, standard deviation, constant checks, Pearson r) where those use only
    components. Anything that needs the pairs raises :class:`NotMergeable`.
    """

    def __init__(self, state: dict[str, Any], probe: bool = False) -> None:
        """``state`` holds one array per group, or ``(replicates, groups)`` (bootstrap).

        Sketch columns (``~obs``, ``~pred``) hold one :data:`Sketch` per group. A ``probe``
        serves any additive sum or sketch asked for (as NaN) and records its column in
        :attr:`asked`: what a metric would need from state.
        """
        self._sketches: dict[str, list[Sketch]] = {
            k: v for k, v in state.items() if k.startswith(SKETCH)
        }
        self._state = {
            k: np.atleast_2d(np.asarray(v, np.float64))
            for k, v in state.items()
            if not k.startswith(SKETCH)
        }
        self.replicates, self.n_groups = (int(d) for d in self._state["n"].shape)
        self._cache: dict[str, FloatArray] = {}
        self.probe = probe
        self.asked: set[str] = set()

    def __getitem__(self, name: str) -> FloatArray:
        if name in self._state and name in STATE_COLUMNS:
            if name in _EXTREMES:  # GroupStats: no pairs, no extremes
                return np.where(self._state["n"] > 0, self._state[name], np.nan)
            return self._state[name]
        if name not in self._cache:
            if name not in COMPONENTS:
                raise NotMergeable(name)
            self._cache[name] = COMPONENTS[name](self)  # type: ignore[arg-type]
        return self._cache[name]

    def cached(self, key: str, compute: Callable[[], FloatArray]) -> FloatArray:
        """Return a parameterized component, computed once (if state allows it)."""
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    def additive(
        self, keys: tuple[str, ...], compute: Callable[[], tuple[FloatArray, ...]]
    ) -> tuple[FloatArray, ...]:
        """Return additive sums ``keys`` from state (``compute`` needs the pairs: unused)."""
        if self.probe:
            self.asked |= {ADDITIVE + k for k in keys}
            return tuple(np.full((self.replicates, self.n_groups), np.nan) for _ in keys)
        missing = [k for k in keys if ADDITIVE + k not in self._state]
        if missing:
            raise NotMergeable(", ".join(missing))
        return tuple(self._state[ADDITIVE + k] for k in keys)

    def sketch_quantile(self, side: str, q: float) -> FloatArray:
        """Return quantile ``q`` of ``obs``/``pred`` from merged sketches (GroupStats's)."""
        column = SKETCH + side
        out = np.full((self.replicates, self.n_groups), np.nan)
        if self.probe:
            self.asked.add(column)
            return out
        if column not in self._sketches:
            raise NotMergeable(column)
        n = self._state["n"]
        for g, (ids, weights) in enumerate(self._sketches[column]):
            out[:, g] = sketch_quantiles(ids, weights, n[:, g], q)
        return out

    def case_mean(self, key: str, case_score: Callable[[], FloatArray]) -> FloatArray:
        """Return a per-case score's group mean from its additive sums (GroupStats.case_mean)."""
        num, den = self.additive((f"{key}|sum", f"{key}|cases"), lambda: ())
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den == 0, np.nan, num / den)

    def time_of_max(self, side: str) -> FloatArray:
        """Return when ``side`` first reaches its group maximum (epoch seconds)."""
        return self._state[f"time_max_{side}"]

    def __getattr__(self, name: str) -> object:
        # Pair-level access (obs, gsum, quantile, column, cases, ...): not in state.
        raise NotMergeable(name)


def compute_from_state(
    requests: Sequence[MetricRequest], states: dict[Transform | None, dict[str, Any]]
) -> dict[MetricRequest, FloatArray]:
    """Compute metrics per group from merged state (one per transform the requests use).

    Returns ``(1, groups)`` arrays like :func:`~teehr_core.metrics.registry.compute_metrics`.
    """
    stats = {t: StateStats(s) for t, s in states.items()}
    return {r: METRICS[r.name].finalize(stats[r.transform], r.params) for r in requests}  # type: ignore[arg-type]


def mergeable(requests: Sequence[MetricRequest]) -> bool:
    """Whether every metric in ``requests`` can be computed from merged state.

    Tried on probe state, which serves any additive sum (those are kept as asked).
    """
    state = _probe_state()
    try:
        with np.errstate(all="ignore"):
            for r in requests:
                METRICS[r.name].finalize(StateStats(state, probe=True), r.params)  # type: ignore[arg-type]
    except (NotMergeable, KeyError):
        return False
    return True
