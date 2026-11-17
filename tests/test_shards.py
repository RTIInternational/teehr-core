"""Shards: key ranges, their sizing, and merging the state shards write."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pytest
import yaml

from teehr_core import engine
from teehr_core.engine import Catalog
from teehr_core.engine.connect import arrow_table, memory_connection, register
from teehr_core.engine.duckdb_backend import DM_TRANSFORM, DuckDBBackend
from teehr_core.engine.shards import ROW_BYTES, in_list, key_ranges, shard_rows
from teehr_core.model.periods import PeriodRegistry
from teehr_core.model.units import UnitRegistry
from teehr_core.spec import load_spec

SPECS = Path(__file__).parent / "specs"


@pytest.fixture(scope="module")
def hydro(fixtures_dir: Path, units: UnitRegistry, periods: PeriodRegistry) -> Catalog:
    return Catalog.from_yaml(fixtures_dir / "hydro" / "datasets.yaml", units, periods)


# -- key ranges ----------------------------------------------------------------------------


def test_key_ranges_are_contiguous_sorted_and_cover_every_key_once() -> None:
    keys = [*[f"g{i:02d}" for i in range(10)][::-1], "g03"]  # unsorted, a duplicate
    ranges = key_ranges(keys, rows_per_key=100, rows_per_shard=300)
    assert [k for r in ranges for k in r] == sorted(set(keys))
    assert len(ranges) == 4  # 1,000 rows at 300 a shard
    assert max(map(len, ranges)) - min(map(len, ranges)) <= 1


def test_key_ranges_take_a_floor_and_never_exceed_the_keys() -> None:
    assert len(key_ranges(["a", "b", "c"], rows_per_key=1, rows_per_shard=1_000)) == 1
    assert len(key_ranges(["a", "b", "c"], rows_per_key=1, rows_per_shard=1_000, min_shards=2)) == 2
    assert len(key_ranges(["a", "b", "c"], rows_per_key=1e9, rows_per_shard=1)) == 3
    assert key_ranges([], rows_per_key=10, rows_per_shard=1) == []


def test_shard_rows_is_a_share_of_memory_at_row_bytes() -> None:
    assert shard_rows(10 * 2**30, share=0.2) == int(10 * 2**30 * 0.2) // ROW_BYTES
    assert shard_rows(1, share=0.2) == 1


def test_in_list_quotes_literals_and_matches_nothing_when_empty() -> None:
    assert in_list(["a", "o'b"]) == "IN ('a', 'o''b')"
    con = memory_connection()
    assert con.execute(f"SELECT 'a' {in_list([])}").fetchone() == (None,)


# -- state from shards ---------------------------------------------------------------------

_METRICS: list[Any] = [
    "count",
    "root_mean_square_error",
    "kling_gupta_efficiency",
    "relative_bias",
    "maximum",
    {"name": "root_mean_square_error", "transform": "log"},
    {"name": "critical_success_index", "threshold": {"value": 2000, "unit": "[cft_i]/s"}},
    "relative_median",  # quantile sketches
]


def _spec(group_by: list[str]) -> Any:
    base = yaml.safe_load((SPECS / "hydro_by_lead_time_bin.yaml").read_text())
    stage = {"group_by": group_by, "metrics": _METRICS}
    return load_spec({**base, "aggregate": [stage], "output": {"include_pairs": True}})


def _sql(sql: str, **tables: pa.Table) -> pa.Table:
    con = memory_connection()
    con.execute("SET TimeZone = 'UTC'")
    for name, table in tables.items():
        register(con, name, table)
    return arrow_table(con, sql)


@pytest.mark.parametrize(
    "group_by",
    [
        ["secondary_dataset", "lead_time_bin"],  # spans series: every shard has a part
        ["primary_series_key", "secondary_dataset"],  # within a series: one shard each
    ],
)
def test_state_from_shards_merged_by_bucket_equals_one_run(
    hydro: Catalog, group_by: list[str]
) -> None:
    spec = _spec(group_by)
    assert DuckDBBackend.state_blockers(spec) == []
    pairs = engine.evaluate(spec, hydro).pairs
    assert pairs is not None
    backend = DuckDBBackend()
    want = backend.aggregate(spec, pairs, stages=1)[0]

    keys = pc.unique(pairs["primary_series_key"]).to_pylist()
    shards = key_ranges(keys, rows_per_key=1, rows_per_shard=1)  # one series each
    assert len(shards) > 1
    parts = [
        backend.first_stage_state(
            spec, pairs.filter(pc.is_in(pairs["primary_series_key"], pa.array(s))), "day"
        )
        for s in shards
    ]
    state = pa.concat_tables(parts, promote_options="permissive")
    merged = _sql(DuckDBBackend.merge_state_by_bucket_sql(spec, "s"), s=state)

    by = [*spec.aggregate[0].group_by, "__transform", "__bucket"]
    distinct = _sql(f"SELECT count(*) AS n FROM (SELECT DISTINCT {', '.join(by)} FROM s)", s=state)
    assert merged.num_rows == distinct["n"][0].as_py()  # one row per group, transform, bucket
    if "lead_time_bin" in group_by:
        assert merged.num_rows < state.num_rows  # shards' parts of a group became one

    got = backend.first_stage_from_state(spec, merged)
    assert got.column_names == want.column_names
    assert got.num_rows == want.num_rows
    for a, b in zip(got.to_pylist(), want.to_pylist(), strict=True):
        for k, v in b.items():
            assert a[k] == (pytest.approx(v, rel=1e-9) if isinstance(v, float) else v), k


def test_merge_by_bucket_sums_dm_rows_per_group_bucket_and_lag() -> None:
    spec = load_spec(
        {
            "version": 1,
            "name": "dm",
            "inputs": {
                "primary": {"dataset": "obs"},
                "secondary": [{"dataset": "a"}, {"dataset": "b"}],
            },
            "pair": {"relation": "xw"},
            "aggregate": [
                {
                    "group_by": ["secondary_dataset"],
                    "metrics": ["root_mean_square_error"],
                    "skill": {
                        "reference": "b",
                        "metrics": ["root_mean_square_error"],
                        "test": "dm",
                    },
                }
            ],
        }
    )
    bucket = pa.scalar(0, pa.timestamp("us", tz="UTC"))
    rows = pa.table(
        {
            "secondary_dataset": ["a", "a", "a"],
            "__transform": [DM_TRANSFORM] * 3,
            "__bucket": pa.array([bucket.value] * 3, pa.timestamp("us", tz="UTC")),
            "dm_lag": [0, 0, 1],
            "dm_n": [2.0, 3.0, 1.0],
            "dm_a": [1.0, 1.0, 1.0],
            "dm_b": [1.0, 2.0, 3.0],
            "dm_s": [0.5, 0.5, 0.5],
        }
    )
    sql = DuckDBBackend.merge_state_by_bucket_sql(spec, "s")
    con = memory_connection()
    register(con, "s", rows)
    # The non-DM half needs state columns these rows don't have: take the DM half alone.
    dm_half = sql.split(" UNION ALL BY NAME ", 1)[1].strip("()")
    got = sorted(con.execute(dm_half).fetchall(), key=lambda r: r[3])
    assert [(r[3], r[4], r[6]) for r in got] == [(0, 5.0, 3.0), (1, 1.0, 3.0)]
