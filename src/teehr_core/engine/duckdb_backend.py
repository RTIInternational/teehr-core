"""Single-node backend: DuckDB for relational steps, NumPy kernels for metrics.

Ported from VeriRace (``verirace/engine/duckdb_backend.py`` at 668ab7d).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from teehr_core.durations import format_hours_label, parse_duration
from teehr_core.engine.connect import (
    S3Config,
    arrow_reader,
    arrow_table,
    memory_connection,
    register,
    storage_access,
)
from teehr_core.engine.memory import RUN_SHARE, default_memory_limit
from teehr_core.engine.plan import Plan, PlanError, PlannedInput, SidePeriods
from teehr_core.engine.sandbox import restrict
from teehr_core.metrics.bootstrap import BootstrapSettings, bootstrap_intervals, state_intervals
from teehr_core.metrics.registry import (
    METRICS,
    MetricRequest,
    TransformKey,
    compute_metrics,
    parse_transform_label,
    transform_label,
)
from teehr_core.metrics.significance import (
    DM_SUMS,
    default_lag,
    dm_from_sums,
    dm_lag_sums,
    time_step,
)
from teehr_core.metrics.state import (
    ADDITIVE,
    SKETCH,
    SKETCH_TYPE,
    STATE_COLUMNS,
    additive_columns,
    compute_from_state,
    group_state,
    merge_sql,
    mergeable,
    sketches_from,
)
from teehr_core.metrics.stats import FloatArray
from teehr_core.spec import (
    AttributeField,
    CalendarField,
    ChangeField,
    ConditionsField,
    EvaluationSpec,
    EventsField,
    ExceedanceField,
    ExprField,
    LeadTimeBinsField,
    NaiveScaleField,
    PercentileField,
    Stage,
    WithinField,
)
from teehr_core.spec.expressions import to_duckdb
from teehr_core.spec.models import (
    ALL_LABEL,
    GROUPING,
    REASON_COLUMN,
    REDUCE_KEYS,
    SERIES_CONSTANT,
    STAT,
    ColumnRef,
    FieldSpec,
    MetricSpec,
    SeriesQuantile,
)
from teehr_core.sql import lit, q

__all__ = ["DuckDBBackend", "EvaluationResult"]

#: Value columns pairing reads itself (others pass through with a role prefix).
#: ``period_start`` and ``period_end`` are absolute periods' windows (ADR-0009): pairs
#: carry every windowed value's window as ``period_start`` and ``period_end`` (the sides'
#: windows are equal, since values pair only then).
CONTRACT_COLUMNS = (
    "series_key", "value_time", "value", "variable", "unit", "reference_time", "member", "quantile",
    "period_start", "period_end",
)  # fmt: skip
_SUMMARY_SQL = {
    "mean": "avg", "median": "median", "min": "min", "max": "max", "std": "stddev_pop",
    "sum": "sum", "count": "count",
}  # fmt: skip


def _dm_from_state(table: pa.Table, stage: Stage, dm: pa.Table | None) -> pa.Table:
    """Add the DM test's columns to a first stage from merged state, from its lag sums."""
    skill = stage.skill
    if skill is None or skill.test is None:
        return table
    keys = list(stage.group_by)
    lags = int(pc.max(dm["dm_lag"]).as_py() or 0) + 1 if dm is not None and dm.num_rows else 1
    sums = np.zeros((4, table.num_rows, lags))
    if dm is not None and dm.num_rows:
        index = {
            row: i for i, row in enumerate(zip(*(table[k].to_pylist() for k in keys), strict=True))
        } if keys else {(): 0}  # fmt: skip
        where = [index.get(r, -1) for r in zip(*(dm[k].to_pylist() for k in keys), strict=True)]
        at = np.array(where if keys else [0] * dm.num_rows, np.int64)
        lag = dm["dm_lag"].to_numpy()
        ok = at >= 0
        for j, name in enumerate(DM_SUMS):
            values = pc.fill_null(dm[name], 0.0).to_numpy(zero_copy_only=False)
            np.add.at(sums[j], (at[ok], lag[ok]), values[ok])
    results = []
    n, a, b, sq = sums
    for i in range(table.num_rows):
        try:
            results.append(dm_from_sums(n[i], a[i], b[i], sq[i], skill.lag))
        except ValueError as exc:
            raise PlanError(
                f"{exc}: the stored state keeps fewer lags than the test now needs; "
                "run the evaluation in full"
            ) from exc
    names = skill.test_columns()
    table = table.append_column(
        names[0], pa.array([r[0] if r else None for r in results], pa.float64())
    )
    return table.append_column(
        names[1], pa.array([r[1] if r else None for r in results], pa.float64())
    )


def _max_lag(d: dict[str, np.ndarray], lag: int | None) -> int:
    """Return the most lags any group's DM test needs (``lag``, or the rule for its count)."""
    if lag is not None:
        return lag
    finite = np.isfinite(d["__d"])
    counts = np.bincount(d["__gid"][finite]) if finite.any() else np.zeros(1, np.int64)
    return default_lag(int(counts.max()))


def _names(names: list[str]) -> str:
    """List names for a sentence: ``a``, ``a and b``, ``a, b and c``."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


@dataclass(frozen=True)
class PairsFiles:
    """Pairs in Parquet files, read in place by DuckDB rather than loaded into memory.

    Aggregating them reads only the columns each query uses, and DuckDB spills to disk
    under its memory limit; only the arrays the metric kernels need are held in memory
    (a shuffle bucket: docs/design/scaling-runs.md).
    """

    files: str | tuple[str, ...]  # Parquet files (or a glob), hive-partitioned by __bucket
    root: str  # the directory (or object-storage prefix) the connection may read
    column_names: list[str]
    #: Object storage credentials, when ``root`` is in object storage.
    s3: S3Config | None = field(default=None, repr=False)

    @staticmethod
    def open(files: str | tuple[str, ...], root: str, s3: S3Config | None = None) -> PairsFiles:
        """Describe the pairs in ``files`` (under ``root``; in object storage with ``s3``)."""
        con = memory_connection()
        try:
            if s3 is not None:
                storage_access(con, s3)
            restrict(con, [root])
            con.execute(f"CREATE VIEW t AS {_files_sql(files)}")
            names = [r[0] for r in con.execute("DESCRIBE t").fetchall()]
        finally:
            con.close()
        return PairsFiles(files, root, names, s3)


@contextmanager
def _as_t(con: duckdb.DuckDBPyConnection, table: Pairs) -> Iterator[None]:
    """Make ``table`` the query's ``t`` (a view over files, or a registered table)."""
    if isinstance(table, PairsFiles):
        con.execute(f"CREATE OR REPLACE TEMP VIEW t AS {_files_sql(table.files)}")
    else:
        register(con, "t", table)
    try:
        yield
    except duckdb.Error as exc:
        raise PlanError(f"query failed: {exc}") from exc
    finally:
        if isinstance(table, PairsFiles):
            con.execute("DROP VIEW t")
        else:
            con.unregister("t")


#: Rows per batch when streaming a result into NumPy.
_BATCH_ROWS = 1 << 20


def _empty(dtype: pa.DataType) -> np.ndarray:
    """Return an empty array of the NumPy type an Arrow column converts to."""
    empty: np.ndarray = pa.array([], dtype).to_numpy(zero_copy_only=False)
    return empty


def _files_sql(files: str | tuple[str, ...]) -> str:
    where = lit(files) if isinstance(files, str) else f"[{', '.join(lit(f) for f in files)}]"
    return f"SELECT * EXCLUDE (__bucket) FROM read_parquet({where}, hive_partitioning = true)"


#: Pairs to aggregate: in memory, or in files (a shuffle bucket).
type Pairs = pa.Table | PairsFiles


@dataclass
class EvaluationResult:
    """Output of an evaluation."""

    spec: EvaluationSpec
    table: pa.Table
    pairs: pa.Table | None = None
    stages: list[pa.Table] = field(default_factory=list)


@dataclass(frozen=True)
class DmSettings:
    """How a run's Diebold-Mariano test steps through time (see metrics.significance).

    ``step`` is one time step in seconds (None: the differentials' own smallest spacing)
    and ``max_lag`` the lags state rows keep (None: as many as each group needs).
    """

    step: int | None = None
    max_lag: int | None = None


#: The ``__transform`` of state rows holding a group's DM sums for one lag and bucket.
DM_TRANSFORM = "__dm"


class DuckDBBackend:
    """Execute plans in an in-process DuckDB database."""

    def __init__(self, dm: DmSettings | None = None, finest_set_only: bool = False) -> None:
        self.dm = dm or DmSettings()
        #: Copy pairs into a rollup's finest set only: a run that merges the first stage
        #: from state builds the coarser sets from state rows (:func:`state_rollup_sql`).
        self.finest_set_only = finest_set_only

    def execute(self, plan: Plan) -> EvaluationResult:
        """Run ``plan`` and return its results."""
        pairs = self.pairs(plan)
        stages = self.aggregate(plan.spec, pairs)
        return EvaluationResult(
            spec=plan.spec,
            table=stages[-1],
            pairs=pairs if plan.spec.output.include_pairs else None,
            stages=stages,
        )

    def pairs(self, plan: Plan) -> pa.Table:
        """Pair, add fields and filter: everything before aggregation."""
        con = self._connect()
        try:
            pairs = self._pair(con, plan)
            conditions = [f for f in plan.spec.fields if isinstance(f, ConditionsField)]
            for f in plan.spec.fields:
                if not isinstance(f, ConditionsField):
                    pairs = self._add_field(con, pairs, f, plan)
            # Copies last: series fields (exceedance, events, change) must see each pair
            # once, and conditions may use any field.
            for c in conditions:
                pairs = self.expand_conditions(con, pairs, c)
            if plan.spec.filter is not None:
                where = to_duckdb(plan.spec.filter)
                pairs = self._sql(con, pairs, f"SELECT * FROM t WHERE {where}")
            if plan.spec.pair.common_sample and len(plan.predicted) > 1:
                pairs = self._sql(con, pairs, _common_sample_sql(len(plan.predicted)))
            if plan.spec.reduce is not None:
                pairs = self._sql(con, pairs, reduce_sql(plan.spec, pairs.column_names))
            first = plan.spec.aggregate[0]
            if first.grouping_sets:
                only = first.finest_set() if self.finest_set_only else None
                sets = None if only is None else [only]
                pairs = self.expand_grouping_sets(con, pairs, first, sets)
            return pairs
        finally:
            con.close()

    def aggregate(
        self,
        spec: EvaluationSpec,
        pairs: Pairs,
        stages: int | None = None,
        memory_limit: str | None = None,
    ) -> list[pa.Table]:
        """Run the aggregate stages on ``pairs`` (optionally only the first ``stages``)."""
        read = [pairs.root] if isinstance(pairs, PairsFiles) else []
        con = self._connect(read, memory_limit, pairs.s3 if isinstance(pairs, PairsFiles) else None)
        try:
            out: list[pa.Table] = []
            current = pairs
            bins = bin_fields(spec)
            for i, stage in enumerate(spec.aggregate[:stages]):
                current = (
                    self._first_stage(con, current, stage, bins)
                    if i == 0
                    else self._summary_stage(con, current, stage, bins)
                )
                out.append(current)
            return out
        finally:
            con.close()

    def summarize(self, spec: EvaluationSpec, first_stage: pa.Table) -> list[pa.Table]:
        """Run the later (summary) stages on a first-stage result."""
        con = self._connect()
        try:
            out = [first_stage]
            for stage in spec.aggregate[1:]:
                out.append(self._summary_stage(con, out[-1], stage, bin_fields(spec)))
            return out
        finally:
            con.close()

    # -- mergeable state (incremental runs by time bucket) ------------------------------

    @staticmethod
    def state_capable(spec: EvaluationSpec) -> bool:
        """Whether the first stage can be computed from merged state (see metrics.state)."""
        return not DuckDBBackend.state_blockers(spec)

    @staticmethod
    def state_blockers(spec: EvaluationSpec) -> list[str]:
        """Why the first stage can't be computed from merged state (empty: it can).

        Its metrics must need only state, on obs and pred (not other columns), without
        bootstrap intervals, guards, or skill matched on observations or tested.
        """
        stage = spec.aggregate[0]
        out: list[str] = []
        if spec.reduce is not None:
            out.append("reduce windows can span the time buckets state is kept by")
        out += _bootstrap_state_blockers(stage)
        if stage.guards:
            out.append("guards are checked on every pair")
        own = [m.output_name() for m in stage.metrics if m.values]
        if own:
            out.append(f"{_names(own)} read{'s' if len(own) == 1 else ''} their own columns")
        unmerged = [
            m.output_name()
            for m in stage.metrics
            if not m.values and not mergeable([m.to_request()])
        ]
        if unmerged:
            verb = "needs" if len(unmerged) == 1 else "need"
            out.append(f"{_names(unmerged)} {verb} every pair, not running sums")
        if stage.skill is not None and stage.skill.matched:
            out.append("skill matched on observations compares them one by one")
        return out

    def first_stage_state(
        self, spec: EvaluationSpec, pairs: pa.Table, bucket: str, counted: str | None = None
    ) -> pa.Table:
        """Return the first stage's state per group, transform and ``bucket`` of value_time.

        ``bucket`` is a ``date_trunc`` part (``month``, ``day``). Columns: the stage's
        group keys, ``__transform`` ('' for none), ``__bucket`` and the state columns.
        Only pairs where ``counted`` (SQL on pairs) holds are counted; the others (read
        before an incremental run's buckets) are only lag partners for the DM test, whose
        sums per lag come as rows of their own (:meth:`_dm_state`).
        """
        stage = spec.aggregate[0]
        keys = list(stage.group_by)
        transforms = sorted({m.transform_key() for m in stage.metrics}, key=transform_label)
        requests = [m.to_request() for m in stage.metrics]
        additive = state_additive_columns(spec)
        # Pair columns the metrics' additive sums read; forecast cases by their key.
        needed = {c for r in requests for c in r.columns()}
        needs_case = "_case" in needed or stage.bootstrap is not None
        metric_columns = sorted(needed - {"_case", "value_time"})
        extra = "".join(f", t.{q(c)}" for c in metric_columns)
        if needs_case:
            case_key = [c for c in _CASE_KEY if c in pairs.column_names]
            extra += f", hash({', '.join(f't.{q(c)}' for c in case_key)}) AS __case"
        where = f" WHERE {counted}" if counted else ""
        con = self._connect()
        try:
            cols = "".join(f"{q(k)}, " for k in keys)
            groups = self._sql(
                con,
                pairs,
                f"SELECT DISTINCT {cols}date_trunc('{bucket}', value_time) AS __bucket "
                f"FROM t{where}",
            )
            numbered = groups.append_column(
                "__gid", pa.array(np.arange(groups.num_rows, dtype=np.int64))
            )
            match = " AND ".join(
                [f"t.{q(k)} IS NOT DISTINCT FROM g.{q(k)}" for k in keys]
                + [f"date_trunc('{bucket}', t.value_time) = g.__bucket"]
            )
            register(con, "g", numbered)
            try:
                rows = self._sql(
                    con,
                    pairs,
                    "SELECT g.__gid AS gid, t.primary_value, t.secondary_value, t.value_time"
                    f"{extra} FROM t JOIN g ON {match}",
                )
            finally:
                con.unregister("g")
            dm = self._dm_state(con, pairs, stage, bucket, counted)
        finally:
            con.close()
        gid = rows["gid"].to_numpy().astype(np.int64)
        obs = rows["primary_value"].to_numpy(zero_copy_only=False).astype(np.float64)
        pred = rows["secondary_value"].to_numpy(zero_copy_only=False).astype(np.float64)
        vt = _numeric(rows, "value_time")
        columns_in = {c: _numeric(rows, c) for c in metric_columns}
        cases = None
        if needs_case:
            raw = rows["__case"].to_numpy(zero_copy_only=False)
            case_ids = np.unique(raw, return_inverse=True)[1].astype(np.int64)
            columns_in["_case"] = case_ids.astype(np.float64)
            # Cases with values, as the bootstrap's min_count counts them on the pairs.
            valued = np.isfinite(obs) & np.isfinite(pred)
            first = np.unique(case_ids[valued], return_index=True)[1]
            cases = np.bincount(gid[valued][first], minlength=groups.num_rows).astype(float)
        out = []
        for transform in transforms:
            reqs = [r for r in requests if r.transform == transform]
            state = group_state(obs, pred, gid, groups.num_rows, vt, transform, reqs, columns_in)
            if cases is not None:
                state[CASES] = cases
            table = groups.append_column(
                "__transform", pa.array([transform_label(transform)] * groups.num_rows, pa.string())
            )
            zeros = np.zeros(groups.num_rows)
            for c in (*STATE_COLUMNS, *additive):
                if c.startswith(SKETCH):  # quantile sketches: a list of {b, w} per group
                    values = state.get(c, [[] for _ in range(groups.num_rows)])
                    table = table.append_column(c, pa.array(values, SKETCH_TYPE))
                else:
                    table = table.append_column(c, pa.array(state.get(c, zeros), pa.float64()))
            if stage.skill is not None and stage.skill.test is not None:
                # Every state row of a spec with a DM test has its columns (see _dm_state).
                table = table.append_column("dm_lag", pa.nulls(table.num_rows, pa.int64()))
                for c in DM_SUMS:
                    table = table.append_column(c, pa.nulls(table.num_rows, pa.float64()))
            out.append(table)
        if dm is not None:
            out.append(dm)
        return pa.concat_tables(out, promote_options="default")

    def _dm_state(
        self,
        con: duckdb.DuckDBPyConnection,
        pairs: Pairs,
        stage: Stage,
        bucket: str,
        counted: str | None,
    ) -> pa.Table | None:
        """DM sums per group, bucket and lag 0..``max_lag`` (None without a DM test).

        A bucket holds the lag pairs whose later end is in it; their earlier ends may be
        in an earlier bucket (an incremental run reads them before its buckets). Rows
        have ``__transform`` :data:`DM_TRANSFORM`, ``dm_lag`` and :data:`DM_SUMS`; every
        state row of such a spec has these columns (NULL on the others).
        """
        skill = stage.skill
        if skill is None or skill.test is None:
            return None
        found = self._differentials(con, pairs, stage, None, counted, bucket)
        if found is None:
            return None
        groups, d = found
        max_lag = self.dm.max_lag if self.dm.max_lag is not None else _max_lag(d, skill.lag)
        step = self.dm.step or time_step(d["__series"], d["__time"]) or 3600
        n_out = groups.num_rows
        sums = dm_lag_sums(
            d["__d"], d["__gid"], d["__series"], d["__time"], step, max_lag, n_out,
            counted=d["__counted"], into=d["__out"],
        )  # fmt: skip
        lags = max_lag + 1
        table = groups.take(pa.array(np.repeat(np.arange(n_out), lags)))
        table = table.append_column(
            "__transform", pa.array([DM_TRANSFORM] * table.num_rows, pa.string())
        )
        table = table.append_column("dm_lag", pa.array(np.tile(np.arange(lags), n_out), pa.int64()))
        for name, values in zip(DM_SUMS, sums, strict=True):
            table = table.append_column(name, pa.array(values.reshape(-1), pa.float64()))
        return table

    def _differentials(
        self,
        con: duckdb.DuckDBPyConnection,
        pairs: Pairs,
        stage: Stage,
        table: pa.Table | None,
        counted: str | None = None,
        bucket: str | None = None,
    ) -> tuple[pa.Table, dict[str, np.ndarray]] | None:
        """Return the DM test's loss differentials, by group (None when there are none).

        Each dataset's loss is averaged per observation (observed series and value time,
        within a group) and differenced with the reference's on observations both have.
        Groups are ``table``'s rows (``__gid`` is the row), or else the groups found
        (with ``bucket``: per bucket of the later end, ``__out``). Columns: ``__gid``,
        ``__out``, ``__series`` (hashed), ``__time`` (epoch seconds), ``__d`` and
        ``__counted`` (``counted`` held).
        """
        skill = stage.skill
        assert skill is not None
        keys = [k for k in stage.group_by if k != "secondary_dataset"]
        per_obs = list(dict.fromkeys([*keys, "primary_series_key", "value_time"]))
        loss = (
            "(secondary_value - primary_value) ^ 2"
            if skill.loss == "squared"
            else "abs(secondary_value - primary_value)"
        )
        cols = ", ".join(q(c) for c in per_obs)
        on = " AND ".join(f"l.{q(c)} IS NOT DISTINCT FROM r.{q(c)}" for c in per_obs)
        ref = lit(skill.reference)
        group = [*keys, "secondary_dataset"]
        diffs = (
            f"WITH l AS (SELECT {cols}, secondary_dataset, avg({loss}) AS loss FROM t "
            f"WHERE primary_value IS NOT NULL AND secondary_value IS NOT NULL GROUP BY ALL), "
            f"d AS (SELECT l.*, l.loss - r.loss AS __d "
            f"FROM l JOIN (SELECT * FROM l WHERE secondary_dataset = {ref}) r ON {on} "
            f"WHERE l.secondary_dataset <> {ref}) "
        )
        is_counted = f"({counted})" if counted else "TRUE"
        if table is None:
            by = ", ".join(q(c) for c in group)
            groups = self._sql(
                con, pairs, diffs + f"SELECT DISTINCT {by} FROM d WHERE {is_counted} ORDER BY ALL"
            )
        else:
            groups = table.select(group)
        if not groups.num_rows:
            return None
        numbered = groups.append_column(
            "__gid", pa.array(np.arange(groups.num_rows, dtype=np.int64))
        )
        match = " AND ".join(f"d.{q(c)} IS NOT DISTINCT FROM g.{q(c)}" for c in group)
        when = f"date_trunc('{bucket}', d.value_time)" if bucket else "NULL::TIMESTAMPTZ"
        register(con, "g", numbered)
        try:
            d = self._arrays(
                con,
                pairs,
                diffs + "SELECT g.__gid, hash(d.primary_series_key) AS __series, "
                "CAST(epoch(d.value_time) AS BIGINT) AS __time, d.__d, "
                f"CAST(epoch({when}) AS BIGINT) AS __bucket, "
                f"{is_counted.replace('value_time', 'd.value_time')} AS __counted "
                f"FROM d JOIN g ON {match}",
            )
        finally:
            con.unregister("g")
        d["__counted"] = d["__counted"].astype(bool)
        if bucket is None:
            d["__out"] = d["__gid"]
            return groups, d
        # One output per (group, bucket of the later end) that has counted differentials.
        span = int(d["__bucket"].max() - d["__bucket"].min()) + 1
        combined = d["__gid"] * span + (d["__bucket"] - d["__bucket"].min())
        outs = np.unique(combined[d["__counted"]])
        d["__out"] = np.minimum(np.searchsorted(outs, combined), max(outs.size - 1, 0))
        d["__counted"] &= outs.size > 0
        if outs.size:
            d["__counted"] &= outs[d["__out"]] == combined
        starts = (outs % span + d["__bucket"].min()) * 1_000_000
        out_groups = groups.take(pa.array(outs // span)).append_column(
            "__bucket", pa.array(starts, pa.timestamp("us", tz="UTC"))
        )
        return out_groups, d

    @staticmethod
    def merge_state_sql(spec: EvaluationSpec, source: str) -> str:
        """Return SQL merging state rows (every bucket and shard) per group and transform."""
        keys = list(spec.aggregate[0].group_by)
        order = group_order(keys, bin_fields(spec)) if keys else "1"
        rows = f"(SELECT * FROM {source} WHERE __transform <> {lit(DM_TRANSFORM)})"
        merged = merge_sql([*keys, "__transform"], rows, state_additive_columns(spec))
        return f"SELECT * FROM ({merged}) ORDER BY {order}, __transform"

    @staticmethod
    def merge_state_by_bucket_sql(spec: EvaluationSpec, source: str) -> str:
        """Return SQL merging state rows per group, transform and bucket (and DM lag).

        For the state of a run done in shards: each shard computed a part of the groups
        spanning series (e.g. by lead time). The result is state like any other
        (``__bucket`` kept), so it merges further and an incremental run can still
        replace some of its buckets.
        """
        keys = [*spec.aggregate[0].group_by, "__transform", "__bucket"]
        dm = lit(DM_TRANSFORM)
        merged = merge_sql(
            keys,
            f"(SELECT * FROM {source} WHERE __transform <> {dm})",
            state_additive_columns(spec),
        )
        skill = spec.aggregate[0].skill
        if skill is not None and skill.test is not None:
            cols = ", ".join(q(k) for k in keys)
            sums = ", ".join(f"sum({c}) AS {c}" for c in DM_SUMS)
            merged = (
                f"({merged}) UNION ALL BY NAME (SELECT {cols}, dm_lag, {sums} FROM {source} "
                f"WHERE __transform = {dm} GROUP BY ALL)"
            )
        return merged

    @staticmethod
    def merge_dm_sql(spec: EvaluationSpec, source: str) -> str | None:
        """Return SQL summing DM state rows per group and lag (None without a DM test)."""
        stage = spec.aggregate[0]
        if stage.skill is None or stage.skill.test is None:
            return None
        keys = "".join(f"{q(k)}, " for k in stage.group_by)
        sums = ", ".join(f"sum({c}) AS {c}" for c in DM_SUMS)
        return (
            f"SELECT {keys}dm_lag, {sums} FROM {source} "
            f"WHERE __transform = {lit(DM_TRANSFORM)} GROUP BY ALL"
        )

    @staticmethod
    def bootstrap_state_sql(spec: EvaluationSpec, source: str) -> str | None:
        """Return SQL merging state rows per group, transform and bootstrap unit.

        The unit (``__unit``) is the day or month of each row's ``__bucket`` (numbered
        from 1970), or its (water) year with ``year_block``. None without a bootstrap.
        """
        stage = spec.aggregate[0]
        boot = stage.bootstrap
        if boot is None:
            return None
        t = "__bucket"
        if boot.method == "year_block":
            start = boot.year_start_month
            unit = (
                f"(year({t}) + CASE WHEN {start} > 1 AND month({t}) >= {start} THEN 1 ELSE 0 END)"
            )
        else:
            period = boot.block_periods()
            assert period is not None  # a state blocker otherwise
            unit = f"datediff('{period[0]}', TIMESTAMPTZ '1970-01-01 00:00:00+00', {t})"
        rows = (
            f"(SELECT *, {unit} AS __unit FROM {source} WHERE __transform <> {lit(DM_TRANSFORM)})"
        )
        return merge_sql(
            [*stage.group_by, "__transform", "__unit"], rows, state_additive_columns(spec)
        )

    def first_stage_from_state(self, spec: EvaluationSpec, state: pa.Table) -> pa.Table:
        """Return the first stage (as :meth:`aggregate` gives it) from state rows."""
        con = self._connect()
        try:
            register(con, "s", state)
            merged = arrow_table(con, self.merge_state_sql(spec, "s"))
            dm_sql = self.merge_dm_sql(spec, "s")
            dm = arrow_table(con, dm_sql) if dm_sql else None
            boot_sql = self.bootstrap_state_sql(spec, "s")
            units = arrow_table(con, boot_sql) if boot_sql else None
        finally:
            con.close()
        return self.first_stage_from_merged(spec, merged, dm, units)

    def first_stage_from_merged(
        self,
        spec: EvaluationSpec,
        merged: pa.Table,
        dm: pa.Table | None = None,
        units: pa.Table | None = None,
    ) -> pa.Table:
        """Return the first stage from merged state (:meth:`merge_state_sql`'s rows).

        ``dm`` holds the DM sums per group and lag (:meth:`merge_dm_sql`'s rows), and
        ``units`` the state per bootstrap unit (:meth:`bootstrap_state_sql`'s rows).
        """
        stage = spec.aggregate[0]
        keys = list(stage.group_by)
        names = merged["__transform"].to_pylist()
        states: dict[TransformKey | None, dict[str, FloatArray]] = {}
        groups: pa.Table | None = None
        for name in dict.fromkeys(names):
            part = merged.filter(pc.equal(merged["__transform"], name))
            groups = groups if groups is not None else part.select(keys)
            states[parse_transform_label(name)] = {
                c: _state_column(part[c]) for c in (*STATE_COLUMNS, *_additive_in(merged))
            }
        if groups is None:
            raise PlanError("the evaluation produced no pairs (check inputs, relation and filters)")
        requests = [m.to_request() for m in stage.metrics]
        values = compute_from_state(requests, states)
        columns: dict[str, pa.Array] = {k: groups[k].combine_chunks() for k in keys}
        for m, r in zip(stage.metrics, requests, strict=True):
            columns[m.output_name()] = _metric_array(m.name, values[r][0])
        if stage.bootstrap is not None:
            if units is None:
                raise ValueError("bootstrap intervals from state need its rows per time unit")
            columns.update(_state_bootstrap(stage, groups, units))
        return _dm_from_state(_add_skill(pa.table(columns), stage), stage, dm)

    @staticmethod
    def _connect(
        read: Sequence[str] = (), memory_limit: str | None = None, s3: S3Config | None = None
    ) -> duckdb.DuckDBPyConnection:
        con = memory_connection()
        limit = memory_limit or default_memory_limit(RUN_SHARE)  # one of several in a run
        con.execute(f"SET TimeZone = 'UTC'; SET memory_limit = '{limit}'")
        if s3 is not None:  # pairs files in object storage (a run's shuffle buckets)
            storage_access(con, s3)
        restrict(con, read)  # registered tables, and only the pairs files given: no others
        return con

    # -- helpers ---------------------------------------------------------------------------

    @staticmethod
    def _sql(con: duckdb.DuckDBPyConnection, table: Pairs, sql: str) -> pa.Table:
        with _as_t(con, table):
            return arrow_table(con, sql)

    def _metric_rows(
        self, con: duckdb.DuckDBPyConnection, pairs: Pairs, sql: str, sizes: np.ndarray | None
    ) -> tuple[pa.Table | None, dict[str, np.ndarray]]:
        """Return the first stage's rows and their gid, obs and pred arrays.

        With the groups' ``sizes`` (only gid, obs and pred are needed), only the arrays:
        each row streams straight into its group's place, so the arrays come out grouped
        without a sort or copy, and nothing else is held alongside them.
        """
        if sizes is not None:
            return None, self._grouped(con, pairs, sql, sizes)
        rows = self._sql(con, pairs, sql)
        return rows, {
            "gid": rows["gid"].to_numpy(),
            "primary_value": rows["primary_value"].to_numpy(zero_copy_only=False),
            "secondary_value": rows["secondary_value"].to_numpy(zero_copy_only=False),
        }

    @staticmethod
    def _grouped(
        con: duckdb.DuckDBPyConnection, table: Pairs, sql: str, sizes: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Stream ``sql``'s gid, obs and pred into arrays ordered by gid (a counting sort)."""
        total = int(sizes.sum())
        out = {
            "gid": np.repeat(np.arange(sizes.size, dtype=np.int64), sizes),
            "primary_value": np.empty(total),
            "secondary_value": np.empty(total),
        }
        cursor = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        with _as_t(con, table):
            for batch in arrow_reader(con, sql, _BATCH_ROWS):
                gid = batch.column("gid").to_numpy()
                order = np.argsort(gid, kind="stable")
                ids, first, counts = np.unique(gid[order], return_index=True, return_counts=True)
                rank = np.arange(gid.size) - np.repeat(first, counts)
                at = np.repeat(cursor[ids], counts) + rank
                for name in ("primary_value", "secondary_value"):
                    out[name][at] = batch.column(name).to_numpy(zero_copy_only=False)[order]
                cursor[ids] += counts
        return out

    @staticmethod
    def _arrays(con: duckdb.DuckDBPyConnection, table: Pairs, sql: str) -> dict[str, np.ndarray]:
        """Like ``_sql`` for numeric columns, as NumPy arrays (NULL as NaN).

        The result streams in batches, so no Arrow copy of it is held alongside.
        """
        with _as_t(con, table):
            reader = arrow_reader(con, sql, _BATCH_ROWS)
            parts: dict[str, list[np.ndarray]] = {n: [] for n in reader.schema.names}
            for batch in reader:
                for name, col in zip(reader.schema.names, batch.columns, strict=True):
                    parts[name].append(col.to_numpy(zero_copy_only=False))
        return {
            name: np.concatenate(p) if p else _empty(reader.schema.field(name).type)
            for name, p in parts.items()
        }

    # -- pairing ---------------------------------------------------------------------------

    def _pair(self, con: duckdb.DuckDBPyConnection, plan: Plan) -> pa.Table:
        obs = plan.observed
        register(con, "obs", obs.table)
        if obs.series is not None:
            register(con, "obs_series", obs.series)
        if plan.relation is not None:
            register(con, "xwalk", plan.relation.table)
        predicted = [
            self._generate(con, plan, i, p) if p.generate else p
            for i, p in enumerate(plan.predicted)
        ]
        plan = replace(plan, predicted=tuple(predicted))

        has_member = any("member" in p.dataset.table.column_names for p in plan.predicted)
        has_quantile = any("quantile" in p.dataset.table.column_names for p in plan.predicted)
        obs_extras = [c for c in obs.table.column_names if c not in CONTRACT_COLUMNS]
        # A predicted input's own "dataset" column would collide with secondary_dataset.
        pred_extras = sorted(
            {c for p in plan.predicted for c in p.dataset.table.column_names}
            - set(CONTRACT_COLUMNS)
            - {"dataset"}
        )
        selects = []
        for i, p in enumerate(plan.predicted):
            register(con, f"pred_{i}", p.dataset.table)
            if p.dataset.series is not None:
                register(con, f"pred_series_{i}", p.dataset.series)
            selects.append(
                self._pair_select(plan, i, p, has_member, has_quantile, obs_extras, pred_extras)
            )
        sql = (
            " UNION ALL ".join(f"({s})" for s in selects)
            + " ORDER BY secondary_dataset, primary_series_key, reference_time NULLS FIRST,"
            " value_time, secondary_series_key"
        )
        try:
            return arrow_table(con, sql)
        except duckdb.Error as exc:
            raise PlanError(f"pairing failed: {exc}") from exc

    def _pair_select(
        self,
        plan: Plan,
        i: int,
        p: PlannedInput,
        has_member: bool,
        has_quantile: bool,
        obs_extras: list[str],
        pred_extras: list[str],
    ) -> str:
        cols = p.dataset.table.column_names
        oc = plan.observed_conversion
        obs_src = (
            "obs"
            if plan.observed_filter is None
            else f"(SELECT * FROM obs WHERE {plan.observed_filter})"
        )
        pred_src = f"pred_{i}" if p.filter is None else f"(SELECT * FROM pred_{i} WHERE {p.filter})"

        def pred_col(name: str, typ: str) -> str:
            return f"p.{q(name)}" if name in cols else f"NULL::{typ}"

        select = [
            "o.series_key AS primary_series_key",
            "p.series_key AS secondary_series_key",
            f"{pred_col('reference_time', 'TIMESTAMPTZ')} AS reference_time",
            "o.value_time AS value_time",
            f"o.value * {oc.scale!r} + {oc.offset!r} AS primary_value",
            f"p.value * {p.conversion.scale!r} + {p.conversion.offset!r} AS secondary_value",
            f"{lit(plan.observed.variable or '')} AS variable",
            f"{lit(plan.unit or plan.observed.unit or '')} AS unit",
            f"{lit(p.dataset.name)} AS secondary_dataset",
        ]
        if has_member:
            select.append(f"{pred_col('member', 'VARCHAR')} AS member")
        if has_quantile:
            select.append(f"{pred_col('quantile', 'DOUBLE')} AS quantile")
        select += [f"o.{q(c)} AS {q('primary_' + c)}" for c in obs_extras]
        select += [f"{pred_col(c, 'VARCHAR')} AS {q('secondary_' + c)}" for c in pred_extras]

        joins = []
        if p.generate is not None:  # generated values are keyed by the observed series
            obs_key = "p.series_key"
        elif plan.relation is not None:
            joins.append("JOIN xwalk x ON x.right_key = p.series_key")
            obs_key = "x.left_key"
        else:
            attr = q(plan.on_attribute or "")
            joins.append(f"JOIN pred_series_{i} ps ON ps.series_key = p.series_key")
            joins.append(f"JOIN obs_series os ON os.{attr} = ps.{attr}")
            obs_key = "os.series_key"
        # Values pair when their windows are equal (ADR-0009). With one and the same
        # instant or duration period on both sides that's equal value times, a join on
        # the stored column (file pruning still works); otherwise on computed windows.
        obs_p = plan.observed_periods
        pred_series = "obs_series" if p.generate is not None else f"pred_series_{i}"
        if obs_p.same_by_value_time(p.periods):
            on = "o.value_time = p.value_time"
            windows = [*obs_p.window_sql("o"), *p.periods.window_sql("p")]
        else:
            obs_src = _windowed(obs_src, obs_p, "obs_series")
            pred_src = _windowed(pred_src, p.periods, pred_series)
            on = "o.__window_start = p.__window_start AND o.__window_end = p.__window_end"
            windows = ["o.__window_start", "o.__window_end"]
            windows += ["p.__window_start", "p.__window_end"]
        if obs_p.windowed:  # the same for every input: statistics match, inst is instant
            # Paired windows are equal, so the primary side's is the pair's.
            select += [f"{windows[0]} AS period_start", f"{windows[1]} AS period_end"]
        joins.append(f"JOIN {obs_src} o ON o.series_key = {obs_key} AND {on}")
        where = ["isfinite(o.value)", "isfinite(p.value)"]
        return (
            f"SELECT {', '.join(select)} FROM {pred_src} p {' '.join(joins)} "
            f"WHERE {' AND '.join(where)}"
        )

    @staticmethod
    def _generate(
        con: duckdb.DuckDBPyConnection, plan: Plan, i: int, p: PlannedInput
    ) -> PlannedInput:
        """Build a generated input's table, keyed by the observed series.

        Climatology: the stored value at each observation's day of year and hour.
        Persistence: for each forecast in ``like`` (series, reference time, value time),
        the last observation at or before the reference time for the linked observed
        series (one row per observed series even when several forecast series link to it).
        """
        assert p.generate is not None and p.like is not None
        register(con, f"like_{i}", p.like.table)
        if p.generate.generate == "climatology":  # look up each observation's day and hour
            sql = (
                "SELECT o.series_key, o.value_time, c.value FROM obs o "
                f"JOIN like_{i} c ON c.series_key = o.series_key "
                "AND c.day_of_year = dayofyear(o.value_time) AND c.hour = hour(o.value_time)"
            )
            try:
                table = arrow_table(con, sql)
            except duckdb.Error as exc:
                raise PlanError(f"generating {p.dataset.name!r} failed: {exc}") from exc
            return replace(p, dataset=replace(p.dataset, table=table))
        if plan.relation is not None:
            link = "JOIN xwalk x ON x.right_key = l.series_key"
            obs_key = "x.left_key"
        else:
            if p.like.series is None or plan.observed.series is None:
                raise PlanError(f"{p.dataset.name!r} needs series catalogs to pair on attributes")
            register(con, f"like_series_{i}", p.like.series)
            attr = q(plan.on_attribute or "")
            link = (
                f"JOIN like_series_{i} ls ON ls.series_key = l.series_key "
                f"JOIN obs_series os ON os.{attr} = ls.{attr}"
            )
            obs_key = "os.series_key"
        age = ""
        if p.generate.max_age is not None:
            secs = parse_duration(p.generate.max_age).total_seconds()
            age = f" WHERE g.reference_time - o.complete <= to_seconds({int(secs)})"
        # Observations usable at a reference time: those complete by then, when their
        # window ends (an hourly mean labelled by its start: an hour later).
        done = (
            "SELECT series_key, value_time, value, __window_end AS complete "
            f"FROM {_windowed('obs', plan.observed_periods, 'obs_series')}"
        )
        sql = (
            "SELECT g.obs_key AS series_key, g.reference_time, g.value_time, o.value, "
            "o.value_time AS persisted_time FROM ("
            f"SELECT DISTINCT {obs_key} AS obs_key, l.reference_time, l.value_time "
            f"FROM like_{i} l {link} WHERE l.reference_time IS NOT NULL) g "
            f"ASOF JOIN ({done}) o ON o.series_key = g.obs_key AND g.reference_time >= o.complete"
            + age
        )
        if p.generate.generate == "blend":
            sql = _blend_sql(sql, i, p)
        else:
            sql = f"SELECT * EXCLUDE (persisted_time) FROM ({sql})"
        # DuckDB runs ASOF joins whose left side it estimates at <= 64 rows as nested loops,
        # and it underestimates this DISTINCT badly: one gage over a year took 70 s instead
        # of 0.07 s. Always use the sort-merge ASOF join.
        con.execute("SET asof_loop_join_threshold = 0")
        if p.climatology is not None:
            register(con, f"clim_{i}", p.climatology.table)
        try:
            table = arrow_table(con, sql)
        except duckdb.Error as exc:
            raise PlanError(f"generating {p.dataset.name!r} failed: {exc}") from exc
        return replace(p, dataset=replace(p.dataset, table=table))

    # -- fields ----------------------------------------------------------------------------

    def _add_field(
        self, con: duckdb.DuckDBPyConnection, pairs: pa.Table, f: FieldSpec, plan: Plan
    ) -> pa.Table:
        if f.name in pairs.column_names:
            raise PlanError(f"field {f.name!r} would overwrite an existing column")
        if isinstance(f, ExprField):
            return self._sql(con, pairs, f"SELECT *, ({to_duckdb(f.expr)}) AS {q(f.name)} FROM t")
        if isinstance(f, CalendarField):
            return self._sql(con, pairs, f"SELECT *, {calendar_sql(f)} AS {q(f.name)} FROM t")
        if isinstance(f, ExceedanceField | EventsField | ChangeField):
            return self._sql(con, pairs, _series_field_sql(f, pairs.column_names))
        if isinstance(f, NaiveScaleField | PercentileField | AttributeField | WithinField):
            # A value per observed series (its naive scale, stored percentile, attribute or
            # polygon), joined to each pair in row order.
            column = "scale" if isinstance(f, NaiveScaleField) else "value"
            register(con, "_teehr_core_aux", plan.aux[f.name].table)
            try:
                return self._sql(
                    con,
                    pairs,
                    f"SELECT * EXCLUDE (__rn) FROM (SELECT t.*, x.{column} AS {q(f.name)} FROM "
                    "(SELECT *, row_number() OVER () AS __rn FROM t) t LEFT JOIN _teehr_core_aux x "
                    "ON x.series_key = t.primary_series_key) "
                    "ORDER BY __rn",
                )
            finally:
                con.unregister("_teehr_core_aux")
        if isinstance(f, ConditionsField):  # copies, made after every field (pairs())
            return self.expand_conditions(con, pairs, f)
        return pairs.append_column(f.name, lead_time_bins(pairs, f))

    def expand_conditions(
        self, con: duckdb.DuckDBPyConnection, pairs: pa.Table, f: ConditionsField
    ) -> pa.Table:
        """Copy each pair into every level of ``f`` it meets, labelled in column ``f.name``."""
        if f.name in pairs.column_names:
            raise PlanError(f"field {f.name!r} would overwrite an existing column")
        labels = ["'all'"] if f.all else []
        labels += [
            f"CASE WHEN ({to_duckdb(expr)}) THEN {lit(label)} END"
            for label, expr in f.levels.items()
        ]
        levels = f"list_filter([{', '.join(labels)}], lambda x: x IS NOT NULL)"
        return self._sql(
            con,
            pairs,
            f"SELECT * EXCLUDE (__level), __level AS {q(f.name)} "
            f"FROM (SELECT *, unnest({levels}) AS __level FROM t)",
        )

    def expand_grouping_sets(
        self,
        con: duckdb.DuckDBPyConnection,
        pairs: pa.Table,
        stage: Stage,
        sets: list[list[str]] | None = None,
    ) -> pa.Table:
        """Copy each pair once per grouping set of ``stage`` (or of ``sets``, a subset)."""
        if GROUPING in pairs.column_names:
            raise PlanError(f"column {GROUPING!r} would be overwritten by the rollup")
        missing = [c for c in stage.rolled_up() if c not in pairs.column_names]
        if missing:
            raise PlanError(f"rollup columns not found: {missing}; available: {pairs.column_names}")
        return self._sql(con, pairs, grouping_sets_sql(stage, "t", sets))

    # -- aggregation -----------------------------------------------------------------------

    def _first_stage(
        self,
        con: duckdb.DuckDBPyConnection,
        pairs: Pairs,
        stage: Stage,
        bins: frozenset[str] = frozenset(),
    ) -> pa.Table:
        missing = [k for k in stage.group_by if k not in pairs.column_names]
        if missing:
            raise PlanError(
                f"group_by columns not found: {missing}; available: {pairs.column_names}"
            )
        boot = stage.bootstrap
        requests = [m.to_request() for m in stage.metrics]
        needed = {c for r in requests for c in r.columns()}
        needed |= {
            c for m in stage.metrics if m.values for c in (m.values.primary, m.values.secondary)
        }
        # Forecast cases (members/quantiles of one forecast) are keyed by a hash the
        # engine computes; bootstrap resamples whole cases.
        needs_case = "_case" in needed or boot is not None
        metric_columns = sorted(needed - {"_case"})
        unknown_cols = [c for c in metric_columns if c not in pairs.column_names]
        if unknown_cols:
            raise PlanError(
                f"metric columns not found: {unknown_cols}; available: {pairs.column_names}"
            )
        extra: list[str] = list(metric_columns)
        if boot is not None:
            extra = list(dict.fromkeys([*extra, *boot.sort_by, "value_time"]))
            unknown = [c for c in boot.sort_by if c not in pairs.column_names]
            if unknown:
                raise PlanError(
                    f"bootstrap sort_by columns not found: {unknown}; "
                    f"available: {pairs.column_names}"
                )
        cols = ", ".join(["t.primary_value", "t.secondary_value", *(f"t.{q(c)}" for c in extra)])
        # Only group ids, obs and pred (the common case): fetched straight into NumPy.
        lean = not extra and not needs_case
        if needs_case:
            case_key = [c for c in _CASE_KEY if c in pairs.column_names]
            cols += f", hash({', '.join(f't.{q(c)}' for c in case_key)}) AS __case"
        sort = "gid"
        if boot is not None:
            # The pair key breaks ties, so the order (and the resamples) is the same
            # however the pairs were produced (e.g. split across shards).
            ties = [c for c in _PAIR_KEY if c in pairs.column_names and c not in boot.sort_by]
            sort = ", ".join(["gid", *(f"t.{q(c)} NULLS FIRST" for c in [*boot.sort_by, *ties])])
        if stage.group_by:
            order = group_order(stage.group_by, bins)
            keys = ", ".join(q(k) for k in stage.group_by)
            counted = self._sql(
                con,
                pairs,
                f"SELECT * FROM (SELECT {keys}, count(*) AS __n FROM t GROUP BY ALL) "
                f"ORDER BY {order}",
            )
            sizes = counted["__n"].to_numpy()
            groups = counted.drop_columns(["__n"])
            # Group ids: number the (small) distinct-groups table here and attach it with a
            # hash join. A window function over all pairs (dense_rank) intermittently stalled
            # DuckDB at one core on large inputs (P2 spike, P4 benchmark).
            numbered = groups.append_column(
                "__gid", pa.array(np.arange(groups.num_rows, dtype=np.int64))
            )
            match = " AND ".join(f"t.{q(k)} IS NOT DISTINCT FROM g.{q(k)}" for k in stage.group_by)
            register(con, "g", numbered)
            try:
                rows, fetched = self._metric_rows(
                    con,
                    pairs,
                    f"SELECT g.__gid AS gid, {cols} FROM t JOIN g ON {match}"
                    + (f" ORDER BY {sort}" if boot else ""),
                    sizes if lean else None,
                )
            finally:
                con.unregister("g")
            n_groups = groups.num_rows
            keys_per_group = [tuple(r.values()) for r in groups.to_pylist()]
        else:
            groups = pa.table({})
            sql = f"SELECT 0::BIGINT AS gid, {cols} FROM t" + (f" ORDER BY {sort}" if boot else "")
            n = self._sql(con, pairs, "SELECT count(*) AS n FROM t")["n"].to_numpy()
            rows, fetched = self._metric_rows(con, pairs, sql, n if lean else None)
            n_groups = 1
            keys_per_group = [()]
        gid = fetched["gid"].astype(np.int64, copy=False)
        obs = fetched["primary_value"].astype(np.float64, copy=False)
        sim = fetched["secondary_value"].astype(np.float64, copy=False)
        del fetched

        columns_in = {c: _numeric(rows, c) for c in metric_columns} if rows is not None else {}
        case_ids = None
        if needs_case:
            assert rows is not None
            raw = rows["__case"].to_numpy(zero_copy_only=False)
            case_ids = np.unique(raw, return_inverse=True)[1].astype(np.int64)
            columns_in["_case"] = case_ids.astype(np.float64)

        def arrays(m: MetricSpec) -> tuple[FloatArray, FloatArray]:
            """Return the values a metric reads: obs and pred, or the columns it names."""
            if m.values is None:
                return obs, sim
            return columns_in[m.values.primary], columns_in[m.values.secondary]

        values = _by_values(
            stage.metrics,
            requests,
            lambda ms, reqs: compute_metrics(
                reqs, *arrays(ms[0]), gid, n_groups, columns=columns_in
            ),
        )
        columns: dict[str, pa.Array] = {
            name: groups[name].combine_chunks() for name in stage.group_by
        }
        for m, value in zip(stage.metrics, values, strict=True):
            columns[m.output_name()] = _metric_array(m.name, value[0])
        if boot is None:
            table = _apply_guards(pa.table(columns), stage, obs, gid)
            del gid, obs, sim, columns_in  # skill and its test read the pairs again
            return self._skill(con, pairs, table, stage)
        settings = BootstrapSettings(
            boot.method, boot.reps, boot.seed, boot.quantiles, boot.min_count, boot.min_years
        )
        assert case_ids is not None
        period = boot.block_periods()
        if period is None:
            blocks = _block_rows(boot.block, rows, gid, n_groups, case_ids)
        else:
            blocks = np.full(n_groups, period[1], dtype=np.int64)
        years = _years(rows, boot.year_start_month) if boot.method == "year_block" else None
        periods = _periods(rows, period[0]) if period else None
        boot_requests = [m.to_request() for m in stage.bootstrapped()]
        intervals = _by_values(
            stage.bootstrapped(),
            boot_requests,
            lambda ms, reqs: bootstrap_intervals(
                reqs,
                *arrays(ms[0]),
                gid,
                keys_per_group,
                blocks,
                settings,
                columns_in,
                case_ids,
                years,
                periods,
            ),
        )
        for m, interval in zip(stage.bootstrapped(), intervals, strict=True):
            for name, qv in zip(boot.columns(m.output_name()), interval, strict=True):
                columns[name] = _metric_array(m.name, qv, interval=True)
        return self._skill(con, pairs, _apply_guards(pa.table(columns), stage, obs, gid), stage)

    def _skill(
        self, con: duckdb.DuckDBPyConnection, pairs: Pairs, table: pa.Table, stage: Stage
    ) -> pa.Table:
        """Add skill columns: against all pairs, or on matched observations; and a test."""
        skill = stage.skill
        if skill is None or not skill.matched:
            return self._test(con, pairs, _add_skill(table, stage), stage)
        keys = [k for k in stage.group_by if k != "secondary_dataset"]
        match_cols = list(dict.fromkeys([*keys, "primary_series_key", "value_time"]))
        sub_stage = Stage(
            group_by=stage.group_by,
            metrics=[m for m in stage.metrics if m.output_name() in skill.metrics],
            skill=skill.model_copy(update={"matched": False}),
        )
        plain = _add_skill(table, stage.model_copy(update={"skill": sub_stage.skill}))
        cols = ", ".join(q(c) for c in match_cols)
        on = " AND ".join(f"t.{q(c)} IS NOT DISTINCT FROM k.{q(c)}" for c in match_cols)
        ref = lit(skill.reference)
        matched: dict[tuple[Any, ...], dict[str, Any]] = {}
        datasets = sorted(set(table["secondary_dataset"].to_pylist()) - {skill.reference})
        for name in datasets:
            # Observations (within a group) that both this dataset and the reference forecast.
            subset = self._sql(
                con,
                pairs,
                f"WITH k AS (SELECT {cols} FROM t WHERE secondary_dataset = {lit(name)} "
                f"INTERSECT SELECT {cols} FROM t WHERE secondary_dataset = {ref}) "
                f"SELECT t.* FROM t JOIN k ON {on} "
                f"WHERE t.secondary_dataset IN ({lit(name)}, {ref})",
            )
            scored = self._first_stage(con, subset, sub_stage).to_pylist()
            for row in scored:
                if row["secondary_dataset"] == name:
                    matched[tuple(row[k] for k in stage.group_by)] = row
        out = table
        for col in skill.metrics:
            name = skill.column(col)
            values = []
            for i, row in enumerate(table.select(stage.group_by).to_pylist()):
                if row["secondary_dataset"] == skill.reference:
                    values.append(plain[name][i].as_py())
                else:
                    hit = matched.get(tuple(row[k] for k in stage.group_by))
                    values.append(hit[name] if hit else None)
            out = out.append_column(name, pa.array(values, pa.float64()))
        return self._test(con, pairs, out, stage)

    def _test(
        self, con: duckdb.DuckDBPyConnection, pairs: Pairs, table: pa.Table, stage: Stage
    ) -> pa.Table:
        """Add Diebold-Mariano test columns against the skill reference, if asked for."""
        skill = stage.skill
        if skill is None or skill.test is None:
            return table
        found = self._differentials(con, pairs, stage, table)
        results: list[tuple[float, float] | None] = [None] * table.num_rows
        if found is not None:
            _, d = found
            max_lag = _max_lag(d, skill.lag)
            step = self.dm.step or time_step(d["__series"], d["__time"]) or 3600
            n, a, b, sq = dm_lag_sums(
                d["__d"], d["__gid"], d["__series"], d["__time"], step, max_lag, table.num_rows
            )
            results = [dm_from_sums(n[i], a[i], b[i], sq[i], skill.lag) for i in range(n.shape[0])]
        stats = [r[0] if r else None for r in results]
        p_values = [r[1] if r else None for r in results]
        names = skill.test_columns()
        table = table.append_column(names[0], pa.array(stats, pa.float64()))
        return table.append_column(names[1], pa.array(p_values, pa.float64()))

    def _summary_stage(
        self,
        con: duckdb.DuckDBPyConnection,
        prev: pa.Table,
        stage: Stage,
        bins: frozenset[str] = frozenset(),
    ) -> pa.Table:
        aggs = []
        for m in stage.metrics:
            expr = f"{_SUMMARY_SQL[m.name]}({q(m.of or '')})"
            if m.name == "sum" and pa.types.is_integer(prev.schema.field(m.of or "").type):
                expr = f"CAST({expr} AS BIGINT)"  # DuckDB widens integer sums to HUGEINT
            aggs.append(f"{expr} AS {q(m.output_name())}")
        keys = ", ".join(q(k) for k in stage.group_by)
        sql = f"SELECT {', '.join([*map(q, stage.group_by), *aggs])} FROM t"
        if stage.group_by:
            sql += f" GROUP BY {keys} ORDER BY {group_order(stage.group_by, bins)}"
        return self._sql(con, prev, sql)


def _series_keys(side: str, columns: list[str]) -> list[str]:
    """Columns that identify one series: a primary series, or a forecast trajectory."""
    if side == "primary":
        return ["primary_series_key"]
    keys = ["secondary_dataset", "secondary_series_key", "reference_time"]
    return keys + [c for c in ("member", "quantile") if c in columns]


def _by_values(
    metrics: Sequence[MetricSpec],
    requests: Sequence[MetricRequest],
    compute: Callable[[list[MetricSpec], list[MetricRequest]], dict[MetricRequest, Any]],
) -> list[Any]:
    """Compute metrics batched by the values they read; results in the metrics' order.

    Metrics on obs and pred share one batch; each ``values`` pair of columns gets its own.
    """
    batches: dict[tuple[str, str] | None, list[int]] = {}
    for i, m in enumerate(metrics):
        key = (m.values.primary, m.values.secondary) if m.values else None
        batches.setdefault(key, []).append(i)
    out: list[Any] = [None] * len(metrics)
    for idx in batches.values():
        reqs = [requests[i] for i in idx]
        results = compute([metrics[i] for i in idx], reqs)
        for i, req in zip(idx, reqs, strict=True):
            out[i] = results[req]
    return out


def _blend_sql(persistence: str, i: int, p: PlannedInput) -> str:
    """Return SQL for the blend of persistence and climatology (Yang et al. 2020).

    ``value = c(vt) + g * (obs(tp) - c(tp))``: climatology at the value time plus the
    persisted anomaly (at the persisted observation's time ``tp``), damped by ``g``, the
    anomalies' correlation at lag ``vt - tp`` for that observed series over the
    observations read (clipped to [0, 1]). Anomalies are observed minus climatology.
    """
    cc = p.climatology_conversion
    clim = (
        f"SELECT series_key, day_of_year, hour, value * {cc.scale!r} + {cc.offset!r} AS cv "
        f"FROM clim_{i}"
    )

    def at(alias: str, time: str) -> str:
        return (
            f"JOIN c {alias} ON {alias}.series_key = p.series_key "
            f"AND {alias}.day_of_year = dayofyear({time}) AND {alias}.hour = hour({time})"
        )

    return f"""
        WITH p AS ({persistence}),
        c AS ({clim}),
        a AS (SELECT o.series_key, epoch(o.value_time) AS t, o.value - c.cv AS a FROM obs o
              JOIN c ON c.series_key = o.series_key
              AND c.day_of_year = dayofyear(o.value_time) AND c.hour = hour(o.value_time)),
        lags AS (SELECT DISTINCT series_key, epoch(value_time) - epoch(persisted_time) AS lag
                 FROM p),
        -- Each anomaly with the time one lag earlier, then a plain equi-join (a join key
        -- mixing both sides would make DuckDB compare every pair of times).
        shifted AS (SELECT l.series_key, l.lag, a1.a, a1.t - l.lag AS earlier
                    FROM lags l JOIN a a1 ON a1.series_key = l.series_key),
        g AS (SELECT s.series_key, s.lag,
                     greatest(0, least(1, sum(s.a * a2.a)
                         / sqrt(sum(s.a * s.a) * sum(a2.a * a2.a)))) AS gamma
              FROM shifted s
              JOIN a a2 ON a2.series_key = s.series_key AND a2.t = s.earlier
              GROUP BY ALL)
        SELECT p.series_key, p.reference_time, p.value_time,
               cv_vt.cv + coalesce(g.gamma, 0) * (p.value - cv_tp.cv) AS value
        FROM p {at("cv_vt", "p.value_time")} {at("cv_tp", "p.persisted_time")}
        LEFT JOIN g ON g.series_key = p.series_key
            AND g.lag = epoch(p.value_time) - epoch(p.persisted_time)
    """


def _common_sample_sql(n_predicted: int) -> str:
    """Return SQL keeping pairs (registered as ``t``) whose observation every input paired.

    An observation is an observed series at a value time; pairs keep their order.
    """
    common = (
        "SELECT primary_series_key, value_time FROM t GROUP BY ALL "
        f"HAVING count(DISTINCT secondary_dataset) = {n_predicted}"
    )
    return (
        "SELECT * EXCLUDE (__rn) FROM (SELECT *, row_number() OVER () AS __rn FROM t) t "
        f"SEMI JOIN ({common}) k ON k.primary_series_key = t.primary_series_key "
        "AND k.value_time = t.value_time ORDER BY __rn"
    )


def _series_field_sql(f: ExceedanceField | EventsField | ChangeField, columns: list[str]) -> str:
    """Return SQL adding a series-aware field to the pairs (registered as ``t``), in order.

    Each series' distinct times are ranked or scanned in time order, then joined back to
    every pair at that time (a forecast's observation repeats across reference times).
    """
    keys = _series_keys(f.side, columns)
    value = f"{f.side}_value"
    cols = ", ".join(q(k) for k in keys)
    part = f"PARTITION BY {cols}"
    match = " AND ".join(f"x.{q(k)} IS NOT DISTINCT FROM t.{q(k)}" for k in keys)
    # An events threshold from a pair column (e.g. a stored percentile) rides along.
    by_column = (
        f.threshold.column
        if isinstance(f, EventsField) and isinstance(f.threshold, ColumnRef)
        else None
    )
    if by_column is not None and by_column not in columns:
        raise PlanError(
            f"field {f.name!r}: threshold column {by_column!r} not found "
            "(define its field before this one)"
        )
    extra = f", {q(by_column)} AS thr" if by_column else ""
    distinct = f"SELECT DISTINCT {cols}, value_time AS vt, {value} AS v{extra} FROM t"
    if isinstance(f, ChangeField):
        secs = int(parse_duration(f.window).total_seconds())
        same = " AND ".join(f"b.{q(k)} IS NOT DISTINCT FROM a.{q(k)}" for k in keys)
        derived = (
            f"SELECT {', '.join(f'a.{q(k)}' for k in keys)}, a.vt, a.v - b.v AS out "
            f"FROM ({distinct}) a JOIN ({distinct}) b ON {same} "
            f"AND b.vt = a.vt - to_seconds({secs})"
        )
        on = f"{match} AND x.vt = t.value_time"
    elif isinstance(f, ExceedanceField):
        derived = (
            f"SELECT {cols}, vt, rank() OVER (w ORDER BY v DESC)::DOUBLE "
            f"/ (count(*) OVER w + 1) AS out FROM ({distinct}) WINDOW w AS ({part})"
        )
        on = f"{match} AND x.vt = t.value_time"
    else:
        if isinstance(f.threshold, SeriesQuantile):
            thr = f"quantile_cont(v, {f.threshold.quantile!r}) OVER ({part})"
        elif isinstance(f.threshold, ColumnRef):
            thr = "thr"
        else:
            thr = repr(f.threshold.value)  # in the evaluation unit (the plan's)
        gap = (
            f"start - lag(stop) OVER ({part} ORDER BY start) <= to_seconds("
            f"{int(parse_duration(f.merge_gap).total_seconds())})"
            if f.merge_gap
            else "false"
        )
        min_dur = int(parse_duration(f.min_duration).total_seconds()) if f.min_duration else 0
        label = " || '|' || ".join(f"coalesce(CAST({q(k)} AS VARCHAR), '')" for k in keys)
        derived = f"""
            WITH s AS (SELECT *, v >= {thr} AS hit FROM ({distinct})),
            c AS (SELECT *, CASE WHEN hit IS DISTINCT FROM lag(hit) OVER (w) THEN 1 ELSE 0 END
                  AS chg FROM s WINDOW w AS ({part} ORDER BY vt)),
            r AS (SELECT *, sum(chg) OVER ({part} ORDER BY vt) AS run FROM c),
            runs AS (SELECT {cols}, run, min(vt) AS start, max(vt) AS stop
                     FROM r WHERE hit GROUP BY ALL),
            m AS (SELECT *, CASE WHEN {gap} THEN 0 ELSE 1 END AS new FROM runs),
            e AS (SELECT *, sum(new) OVER ({part} ORDER BY start) AS ev FROM m)
            SELECT {cols}, min(start) AS start, max(stop) AS stop,
                   {label} || '@' || strftime(min(start), '%Y-%m-%dT%H:%M:%SZ') AS out
            FROM e GROUP BY {cols}, ev
            HAVING max(stop) - min(start) >= to_seconds({min_dur})
        """
        on = f"{match} AND t.value_time BETWEEN x.start AND x.stop"
    return (
        f"SELECT * EXCLUDE (__rn) FROM (SELECT t.*, x.out AS {q(f.name)} "
        f"FROM (SELECT *, row_number() OVER () AS __rn FROM t) t "
        f"LEFT JOIN ({derived}) x ON {on}) ORDER BY __rn"
    )


#: A reduce statistic's SQL aggregate.
_REDUCE_AGGS = {"mean": "avg", "min": "min", "max": "max", "sum": "sum"}


def reduce_sql(spec: EvaluationSpec, columns: Sequence[str]) -> str:
    """Return SQL reducing each window of pairs (``t``) to one pair per statistic.

    A window is a group of ``spec.reduce_keys()`` and the forecast's keys present
    (:data:`REDUCE_KEYS`). It keeps those keys, the per-series fields and the pairs'
    variable and unit (one value per window), its first ``value_time``, and each
    statistic's observed and predicted values labelled in ``stat``.
    """
    assert spec.reduce is not None
    keys = [*spec.reduce_keys(), *(k for k in REDUCE_KEYS if k not in spec.reduce_keys())]
    missing = [k for k in spec.reduce_keys() if k not in columns]
    if missing:
        raise PlanError(f"reduce.by columns not found: {missing}; available: {list(columns)}")
    keys = [k for k in keys if k in columns]
    carried = [f.name for f in spec.fields if isinstance(f, SERIES_CONSTANT)]
    carried += [c for c in ("variable", "unit") if c in columns]
    aggs = [
        f"{_REDUCE_AGGS[stat]}({side}_value) AS {q(f'__{side}_{stat}')}"
        for stat in spec.reduce.stats
        for side in ("primary", "secondary")
    ]
    window = ", ".join(
        [
            *map(q, keys),
            *(f"any_value({q(c)}) AS {q(c)}" for c in carried),
            "min(value_time) AS value_time",
            *aggs,
        ]
    )
    stats = ", ".join(
        f"{{'stat': {lit(stat)}, 'p': {q(f'__primary_{stat}')}, 's': {q(f'__secondary_{stat}')}}}"
        for stat in spec.reduce.stats
    )
    out = ", ".join([*map(q, keys), *map(q, carried), "value_time"])
    return (
        f"SELECT {out}, __s.stat AS {q(STAT)}, __s.p AS primary_value, __s.s AS secondary_value "
        f"FROM (SELECT *, unnest([{stats}]) AS __s FROM "
        f"(SELECT {window} FROM t GROUP BY {', '.join(map(q, keys))}))"
    )


def grouping_sets_sql(stage: Stage, source: str, sets: list[list[str]] | None = None) -> str:
    """Return SQL copying ``source``'s rows once per grouping set (``sets``: some of them).

    In each copy the columns its set aggregates across read ``(all)`` and ``grouping``
    names the set; rolled-up columns become text so that label fits beside the values.
    Works on pairs and on mergeable state rows alike.
    """
    assert stage.grouping_sets is not None
    columns = stage.rolled_up()
    names = dict(zip(map(tuple, stage.grouping_sets), stage.set_names(), strict=True))
    selects = []
    for s in sets if sets is not None else stage.grouping_sets:
        replace = ", ".join(
            f"CAST({q(c)} AS VARCHAR) AS {q(c)}" if c in s else f"{lit(ALL_LABEL)} AS {q(c)}"
            for c in columns
        )
        name = lit(names[tuple(s)])
        selects.append(f"SELECT * REPLACE ({replace}), {name} AS {q(GROUPING)} FROM {source}")
    return " UNION ALL ".join(selects)


def state_rollup_sql(stage: Stage, source: str) -> str:
    """Return SQL giving every grouping set's state rows from the finest set's (``source``).

    State merges, so a coarser set's rows are the finest set's rows with the columns it
    aggregates across relabelled ``(all)``, merged afterwards like any state.
    """
    finest = stage.finest_set()
    assert finest is not None and stage.grouping_sets is not None
    name = stage.set_names()[stage.grouping_sets.index(finest)]
    rows = f"(SELECT * EXCLUDE ({q(GROUPING)}) FROM {source} WHERE {q(GROUPING)} = {lit(name)})"
    return f"({grouping_sets_sql(stage, rows)})"


def calendar_sql(f: CalendarField) -> str:
    """Return SQL for a calendar field's label of ``f.of`` (null when that time is).

    Times are timezone-aware, and DuckDB takes their parts in the session's time zone: the
    engine's connections set ``TimeZone = 'UTC'``, so labels are UTC.
    """
    t = q(f.of)
    year, month = f"year({t})", f"month({t})"
    start = f.start_month()
    # The year a label belongs to: the calendar year, or (from year_start_month) the
    # year it ends in, so October 2024 is water year 2025.
    owner = year if start == 1 else f"({year} + CASE WHEN {month} >= {start} THEN 1 ELSE 0 END)"
    if f.part == "year":
        return f"CAST({year} AS VARCHAR)"
    if f.part == "month":
        return f"strftime({t}, '%Y-%m')"
    if f.part == "water_year":
        return f"CAST({owner} AS VARCHAR)"
    if f.part == "quarter":
        quarter = f"(({month} - {start} + 12) % 12) // 3 + 1"
        return f"CAST({owner} AS VARCHAR) || '-Q' || CAST({quarter} AS VARCHAR)"
    seasons = "['DJF', 'DJF', 'MAM', 'MAM', 'MAM', 'JJA', 'JJA', 'JJA', 'SON', 'SON', 'SON', 'DJF']"
    return f"CASE WHEN {t} IS NULL THEN NULL ELSE {seasons}[{month}] END"


def bin_fields(spec: EvaluationSpec) -> frozenset[str]:
    """Names of the spec's lead-time bin fields."""
    return frozenset(f.name for f in spec.fields if isinstance(f, LeadTimeBinsField))


def _bin_start(col: str) -> str:
    """Return SQL for a lead-time bin label's start in minutes (``(-inf, …)`` first)."""
    number = f"TRY_CAST(regexp_extract({col}, '^\\[(-?[0-9]+)', 1) AS DOUBLE)"
    scale = f"CASE WHEN regexp_matches({col}, '^\\[-?[0-9]+h') THEN 60 ELSE 1 END"
    return f"CASE WHEN starts_with({col}, '(-inf') THEN -1e300 ELSE {number} * {scale} END"


def group_order(group_by: Sequence[str], bins: frozenset[str] = frozenset()) -> str:
    """ORDER BY terms for group keys: lead-time bins by when they start, not as text."""
    terms = []
    for k in group_by:
        if k in bins:
            terms.append(f"{_bin_start(q(k))} NULLS LAST")
        terms.append(f"{q(k)} NULLS LAST")
    return ", ".join(terms)


def lead_time_bins(pairs: pa.Table, f: LeadTimeBinsField) -> pa.Array:
    """Label each pair's lead time (``value_time - reference_time``) with its bin."""
    vt = pairs["value_time"].to_numpy().astype("datetime64[us]").astype(np.int64)
    rt_col = pairs["reference_time"]
    rt = (
        rt_col.fill_null(pa.scalar(0, rt_col.type))
        .to_numpy()
        .astype("datetime64[us]")
        .astype(np.int64)
    )
    valid = ~rt_col.is_null().to_numpy(zero_copy_only=False)
    lead_us = vt - rt
    us = 1_000_000

    def label(a: timedelta | None, b: timedelta | None) -> str:
        lo = "-inf" if a is None else format_hours_label(a)
        hi = "inf" if b is None else format_hours_label(b)
        return f"[{lo}, {hi})" if a is not None else f"({lo}, {hi})"

    if f.width is not None:
        width_us = int(parse_duration(f.width).total_seconds()) * us
        idx = np.floor_divide(lead_us, width_us)
        uniq, inverse = np.unique(idx[valid], return_inverse=True)
        names = [
            label(
                timedelta(microseconds=int(k) * width_us),
                timedelta(microseconds=int(k + 1) * width_us),
            )
            for k in uniq
        ]
    else:
        assert f.edges is not None
        edges = [parse_duration(e) for e in f.edges]
        edges_us = np.array([int(e.total_seconds()) * us for e in edges], dtype=np.int64)
        idx = np.searchsorted(edges_us, lead_us, side="right") - 1
        uniq, inverse = np.unique(idx[valid], return_inverse=True)
        names = [
            label(None, edges[0])
            if k < 0
            else label(edges[k], edges[k + 1] if k + 1 < len(edges) else None)
            for k in uniq
        ]
    out = np.full(lead_us.shape, None, dtype=object)
    out[valid] = np.array(names, dtype=object)[inverse] if names else []
    return pa.array(out.tolist(), pa.string())


#: Columns that identify a forecast case: one forecast of one series for one time.
_CASE_KEY = (
    "primary_series_key",
    "secondary_dataset",
    "secondary_series_key",
    "reference_time",
    "value_time",
)

#: Columns that identify a pair (used to order pairs deterministically).
_PAIR_KEY = (
    "value_time",
    "reference_time",
    "secondary_dataset",
    "secondary_series_key",
    "primary_series_key",
    "member",
    "quantile",
)


def _metric_array(name: str, values: np.ndarray, interval: bool = False) -> pa.Array:
    """Arrow array for a metric's values (or its bootstrap quantiles)."""
    output = METRICS[name].output
    mask = np.isnan(values)
    if output == "timestamp":
        micros = np.where(mask, 0, np.round(values * 1e6)).astype(np.int64)
        return pa.array(micros, pa.timestamp("us", tz="UTC"), mask=mask)
    if output == "int" and not interval:
        return pa.array(values.astype(np.int64), pa.int64())
    return pa.array(values, pa.float64(), mask=mask)


def _numeric(rows: pa.Table, column: str) -> np.ndarray:
    """Return a pair column as float64 (timestamps as epoch seconds)."""
    arr = rows[column]
    out: np.ndarray
    if pa.types.is_timestamp(arr.type):
        micros = arr.cast(pa.timestamp("us", tz="UTC")).cast(pa.int64())
        out = micros.to_numpy(zero_copy_only=False).astype(np.float64) / 1e6
        return out
    t = arr.type
    if not (pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t)):
        raise PlanError(f"column {column!r} must be numeric for metrics, not {t}")
    out = arr.cast(pa.float64()).to_numpy(zero_copy_only=False)
    return out


def _apply_guards(table: pa.Table, stage: Stage, obs: np.ndarray, gid: np.ndarray) -> pa.Table:
    """Null guarded metrics (and their intervals) per group; add ``null_reason``."""
    if not stage.guards:
        return table
    n = table.num_rows
    count = np.bincount(gid, minlength=n).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(gid, weights=obs, minlength=n) / count
        std = np.sqrt(
            np.maximum(np.bincount(gid, weights=obs * obs, minlength=n) / count - mean**2, 0)
        )
    all_metrics = [m.output_name() for m in stage.metrics]
    nulled: dict[str, np.ndarray] = {}
    reasons: list[list[str]] = [[] for _ in range(n)]
    for guard in stage.guards:
        checks: list[tuple[np.ndarray, str, np.ndarray, float]] = []
        if guard.min_count is not None:
            checks.append((count < guard.min_count, "pairs", count, guard.min_count))
        if guard.min_obs_mean is not None:
            checks.append((~(mean >= guard.min_obs_mean), "obs mean", mean, guard.min_obs_mean))
        if guard.min_obs_std is not None:
            checks.append((~(std >= guard.min_obs_std), "obs std", std, guard.min_obs_std))
        metrics = guard.metrics or all_metrics
        for tripped, what, value, limit in checks:
            for g in np.flatnonzero(tripped):
                shown = f"{value[g]:.4g}" if what != "pairs" else str(int(value[g]))
                scope = "all metrics" if guard.metrics is None else ", ".join(metrics)
                reasons[g].append(f"{scope}: {what} {shown} < {limit:g}")
            for m in metrics:
                nulled[m] = nulled.get(m, np.zeros(n, bool)) | tripped
    boot = stage.bootstrap
    bootstrapped = {m.output_name() for m in stage.bootstrapped()} if boot else set()
    for metric, mask in nulled.items():
        names = [metric, *(boot.columns(metric) if boot and metric in bootstrapped else [])]
        for name in names:
            i = table.schema.get_field_index(name)
            col = table.column(name)
            table = table.set_column(
                i, name, pc.if_else(pa.array(mask), pa.nulls(n, col.type), col)
            )
    reason = pa.array(["; ".join(r) if r else None for r in reasons], pa.string())
    return table.append_column(REASON_COLUMN, reason)


def _bootstrap_state_blockers(stage: Stage) -> list[str]:
    """Why a stage's bootstrap can't be drawn from state per time unit (empty: it can)."""
    boot = stage.bootstrap
    if boot is None:
        return []
    if boot.method == "iid":
        return ["iid bootstrap intervals resample individual pairs"]
    out = []
    if boot.method != "year_block" and boot.block_periods() is None:
        what = f"{boot.block} rows" if isinstance(boot.block, int) else repr(boot.block)
        out.append(f"a bootstrap block of {what} isn't whole days or months")
    if boot.sort_by != ["value_time"]:
        out.append("the bootstrap orders pairs by more than value_time (sort_by)")
    timed = [m.output_name() for m in stage.bootstrapped() if m.name in _TIME_OF_MAX]
    if timed:
        out.append(f"bootstrap intervals of {_names(timed)} need each pair's time")
    return out


#: Metrics whose bootstrap needs when each replicate's maximum occurred (not in state).
_TIME_OF_MAX = frozenset({"max_value_time", "max_value_time_delta"})


def _state_bootstrap(stage: Stage, groups: pa.Table, units: pa.Table) -> dict[str, pa.Array]:
    """Interval columns of a first stage from its state per group, transform and unit."""
    boot = stage.bootstrap
    assert boot is not None
    keys = list(stage.group_by)
    index = {tuple(r.values()): g for g, r in enumerate(groups.to_pylist())}
    gid_all = np.array(
        [index[tuple(r.values())] for r in units.select(keys).to_pylist()], dtype=np.int64
    )
    unit_all = units["__unit"].to_numpy().astype(np.int64)
    names = units["__transform"].to_pylist()
    states: dict[TransformKey | None, dict[str, FloatArray]] = {}
    gid: np.ndarray | None = None
    unit: np.ndarray | None = None
    for name in dict.fromkeys(names):
        rows = np.flatnonzero(pc.equal(units["__transform"], name).to_numpy(zero_copy_only=False))
        order = rows[np.lexsort((unit_all[rows], gid_all[rows]))]
        if gid is None or unit is None:
            gid, unit = gid_all[order], unit_all[order]
        elif not (np.array_equal(gid, gid_all[order]) and np.array_equal(unit, unit_all[order])):
            raise ValueError("state rows per time unit differ between transforms")
        states[parse_transform_label(name)] = {
            c: _state_column(units[c].take(pa.array(order)))
            for c in (*STATE_COLUMNS, *_additive_in(units))
        }
    assert gid is not None and unit is not None
    n_groups = groups.num_rows
    starts = np.searchsorted(gid, np.arange(n_groups))
    first = np.where(starts < gid.size, unit[np.minimum(starts, gid.size - 1)], 0)
    # year_block: years drawn, by their rank in the group; otherwise days or months from the
    # group's first, empty ones counting.
    position = (
        np.arange(gid.size) - starts[gid] if boot.method == "year_block" else unit - first[gid]
    )
    counted = states.get(None) or max(states.values(), key=lambda s: float(np.nansum(s["n"])))
    # Forecast cases with values (ensembles: one per forecast, not per member).
    per_unit = counted[CASES] if CASES in counted else counted["n"]
    cases = np.bincount(gid, weights=np.nan_to_num(per_unit), minlength=n_groups)
    period = boot.block_periods()
    settings = BootstrapSettings(
        boot.method, boot.reps, boot.seed, boot.quantiles, boot.min_count, boot.min_years
    )
    requests = [m.to_request() for m in stage.bootstrapped()]
    keys_per_group = [tuple(r.values()) for r in groups.to_pylist()]
    intervals = state_intervals(
        requests,
        states,
        gid,
        position,
        keys_per_group,
        period[1] if period else 1,
        settings,
        cases.astype(np.int64),
    )
    columns: dict[str, pa.Array] = {}
    for m, r in zip(stage.bootstrapped(), requests, strict=True):
        for name, qv in zip(boot.columns(m.output_name()), intervals[r], strict=True):
            columns[name] = _metric_array(m.name, qv, interval=True)
    return columns


def _periods(rows: pa.Table, part: str) -> np.ndarray:
    """Each row's calendar day or month of value_time (UTC), numbered from 1970."""
    unit = "D" if part == "day" else "M"
    t = rows["value_time"].cast(pa.timestamp("us", tz="UTC")).to_numpy()
    out: np.ndarray = t.astype(f"datetime64[{unit}]").astype(np.int64)
    return out


def _years(rows: pa.Table, start_month: int) -> np.ndarray:
    """Each row's (water) year: the calendar year it ends in, starting in ``start_month``."""
    t = rows["value_time"].to_numpy().astype("datetime64[M]")
    year = t.astype("datetime64[Y]").astype(np.int64) + 1970
    month = (t - t.astype("datetime64[Y]")).astype(np.int64) + 1
    out: np.ndarray = year + ((month >= start_month) & (start_month > 1)).astype(np.int64)
    return out


def _block_rows(
    block: str | int | None,
    rows: pa.Table,
    gid: np.ndarray,
    n_groups: int,
    case_ids: np.ndarray,
) -> np.ndarray:
    """Return each group's bootstrap block length in rows.

    A number is used as is; a duration is scaled by the group's density of pairs over
    its time span (rows per second x block seconds).
    """
    if block is None:
        return np.ones(n_groups, dtype=np.int64)
    if isinstance(block, int):
        return np.full(n_groups, block, dtype=np.int64)
    seconds = parse_duration(block).total_seconds()
    t = (
        rows["value_time"]
        .cast(pa.timestamp("us", tz="UTC"))
        .cast(pa.int64())
        .to_numpy(zero_copy_only=False)
        / 1e6
    )
    lo = np.full(n_groups, np.inf)
    hi = np.full(n_groups, -np.inf)
    np.minimum.at(lo, gid, t)
    np.maximum.at(hi, gid, t)
    # Blocks are counted in forecast cases (one per pair for deterministic forecasts).
    first = np.unique(case_ids, return_index=True)[1]
    n = np.bincount(gid[first], minlength=n_groups)
    span = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        rows_per_block = np.where(span > 0, np.round(seconds * (n - 1) / span), 1)
    return np.clip(np.nan_to_num(rows_per_block, nan=1), 1, np.maximum(n, 1)).astype(np.int64)


def _add_skill(table: pa.Table, stage: Stage) -> pa.Table:
    """Add skill (or ratio) columns against the reference dataset's matching rows."""
    skill = stage.skill
    if skill is None:
        return table
    keys = [k for k in stage.group_by if k != "secondary_dataset"]
    key_values = list(zip(*(table[k].to_pylist() for k in keys), strict=True)) or [
        () for _ in range(table.num_rows)
    ]
    datasets = table["secondary_dataset"].to_pylist()
    reference = {
        k: i
        for i, (k, d) in enumerate(zip(key_values, datasets, strict=True))
        if d == skill.reference
    }
    for col in skill.metrics:
        values = table[col].to_pylist()
        out: list[float | None] = []
        for k, v in zip(key_values, values, strict=True):
            i = reference.get(k)
            ref = values[i] if i is not None else None
            if v is None or ref is None or ref == 0:
                out.append(None)
            else:
                out.append(1.0 - v / ref if skill.form == "skill" else v / ref)
        table = table.append_column(skill.column(col), pa.array(out, pa.float64()))
    return table


def _windowed(src: str, periods: SidePeriods, series: str) -> str:
    """Return SQL for the rows of ``src`` with their windows (``__window_start``, ``_end``).

    When the period differs by series, each row's comes from ``series`` (a registered
    series table with a ``period`` column), or else the dataset's default.
    """
    if not periods.by_series:
        start, end = periods.window_sql("w")
        return f"(SELECT w.*, {start} AS __window_start, {end} AS __window_end FROM {src} w)"
    default = "NULL::VARCHAR" if periods.default is None else lit(periods.default)
    start, end = periods.window_sql("w", f"coalesce(s.period, {default})")
    return (
        f"(SELECT w.*, {start} AS __window_start, {end} AS __window_end FROM {src} w "
        f"LEFT JOIN {series} s ON s.series_key = w.series_key)"
    )


#: State column counting each group's forecast cases with values (kept with a bootstrap,
#: whose min_count counts cases: an ensemble's members are one case).
CASES = ADDITIVE + "cases"


def state_additive_columns(spec: EvaluationSpec) -> list[str]:
    """Return the additive state columns runs of ``spec`` keep (metrics.state.additive_columns).

    The same for every bucket and shard of the spec, so stored state has fixed columns.
    """
    stage = spec.aggregate[0]
    requests = [m.to_request() for m in stage.metrics]
    out = additive_columns(requests)
    if stage.bootstrap is not None:
        out.append(CASES)
    return out


def _additive_in(table: pa.Table) -> list[str]:
    """Return ``table``'s additive state columns (sums and quantile sketches)."""
    return [c for c in table.column_names if c.startswith((ADDITIVE, SKETCH))]


def _state_column(column: pa.ChunkedArray) -> Any:
    """Return a state column as compute_from_state takes it (NULL: NaN, or no counts)."""
    if pa.types.is_list(column.type):
        return sketches_from(column)
    return pc.fill_null(column, np.nan).to_numpy(zero_copy_only=False).astype(np.float64)
