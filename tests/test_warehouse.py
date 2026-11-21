"""teehr_core.warehouse against a local Iceberg catalog (SQLite and files)."""

import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.expressions import EqualTo

from teehr_core import warehouse as wh
from teehr_core.warehouse.changes import month_range

T0 = datetime(2026, 1, 1, tzinfo=UTC)
COLUMNS = [
    {"name": "series_key", "type": "string", "required": True},
    {"name": "value_time", "type": "timestamptz", "required": True},
    {"name": "value", "type": "float64"},
]
KEY = ["series_key", "value_time"]
SCHEMA = wh.iceberg_schema(COLUMNS).as_arrow()


@pytest.fixture
def catalog(tmp_path):
    catalog = SqlCatalog("test", uri=f"sqlite:///{tmp_path}/c.db", warehouse=f"file://{tmp_path}/w")
    catalog.create_namespace("p")
    return catalog


def create(catalog, name="ts", columns=COLUMNS, key=KEY, layout=None):
    layout = layout or wh.Layout(partition=("value_time", "month"), sort=tuple(key))
    schema = wh.iceberg_schema(columns, key)
    return catalog.create_table(
        ("p", name),
        schema=schema,
        partition_spec=wh.partition_spec(schema, layout.partition),
        sort_order=wh.sort_order(schema, layout.sort),
        properties=wh.table_properties(layout),
    )


@pytest.fixture
def table(catalog):
    return create(catalog)


def rows(*items):
    """Rows of (series, days after 2026-01-01, value)."""
    return pa.table(
        {
            "series_key": [k for k, _, _ in items],
            "value_time": [T0 + timedelta(days=d) for _, d, _ in items],
            "value": [v for _, _, v in items],
        },
        schema=SCHEMA,
    )


def stored(table):
    return sorted(
        (r["series_key"], (r["value_time"] - T0).days, r["value"])
        for r in table.refresh().scan().to_arrow().to_pylist()
    )


def current(table):
    return table.refresh().current_snapshot().snapshot_id


def test_import_needs_no_pyiceberg_until_warehouse() -> None:
    code = "import sys, teehr_core, teehr_core.engine; print('pyiceberg' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_layout_sets_key_partition_sort_and_defaults(table) -> None:
    assert wh.key_columns(table) == KEY
    assert str(table.spec().fields[0].transform) == "month"
    assert [table.schema().find_column_name(f.source_id) for f in table.sort_order().fields] == KEY
    assert table.properties["write.parquet.row-group-limit"] == "150000"


def test_upsert_replaces_matching_keys_keeps_the_rest(table) -> None:
    wh.upsert(table, rows(("a", 0, 1.0), ("a", 1, 2.0), ("a", 40, 3.0), ("b", 1, 4.0)))
    snapshot = wh.upsert(table, rows(("a", 1, 20.0), ("a", 2, 5.0), ("c", 1, 6.0)))
    assert stored(table) == [
        ("a", 0, 1.0), ("a", 1, 20.0), ("a", 2, 5.0), ("a", 40, 3.0), ("b", 1, 4.0), ("c", 1, 6.0),
    ]  # fmt: skip
    assert snapshot == current(table)


def test_key_range_is_small_for_many_rows() -> None:
    data = rows(*[(f"s{i % 50}", i % 30, 0.0) for i in range(5000)])
    where = repr(wh.key_range(data, KEY))
    assert where.count("GreaterThanOrEqual") == 1 and where.count("LessThanOrEqual") == 1
    assert len(where) < 5000


def test_upsert_on_a_number_key(catalog) -> None:
    columns = [{"name": "id", "type": "int64", "required": True}, {"name": "v", "type": "string"}]
    table = create(catalog, "ids", columns, ["id"], wh.Layout())
    schema = wh.iceberg_schema(columns).as_arrow()

    def batch(*pairs):
        return pa.table({"id": [i for i, _ in pairs], "v": [v for _, v in pairs]}, schema=schema)

    wh.upsert(table, batch((1, "a"), (5, "b"), (9, "c")))
    wh.upsert(table, batch((5, "B"), (6, "d")))
    got = sorted((r["id"], r["v"]) for r in table.refresh().scan().to_arrow().to_pylist())
    assert got == [(1, "a"), (5, "B"), (6, "d"), (9, "c")]


def test_upsert_needs_a_key(catalog) -> None:
    table = create(catalog, "nokey", key=[], layout=wh.Layout())
    with pytest.raises(ValueError, match="key"):
        wh.upsert(table, rows(("a", 0, 1.0)))


def test_insert_missing_leaves_existing_rows(table) -> None:
    wh.upsert(table, rows(("a", 0, 1.0)))
    assert wh.insert_missing(table, rows(("a", 0, 9.0), ("b", 0, 2.0))) == 1
    assert stored(table) == [("a", 0, 1.0), ("b", 0, 2.0)]


def test_merge_changes_only_provided_columns(catalog) -> None:
    columns = [
        {"name": "series_key", "type": "string", "required": True},
        {"name": "location_key", "type": "string"},
        {"name": "name", "type": "string"},
    ]
    table = create(catalog, "series", columns, ["series_key"], wh.Layout())
    schema = wh.iceberg_schema(columns).as_arrow()
    wh.upsert(table, pa.table({"series_key": ["a"], "location_key": [None], "name": ["A"]},
                              schema=schema))  # fmt: skip
    update = pa.table({"series_key": ["a"], "location_key": ["L1"], "name": [None]}, schema=schema)
    wh.merge(table, update, ["series_key"], ["series_key", "location_key"])
    assert table.refresh().scan().to_arrow().to_pylist() == [
        {"series_key": "a", "location_key": "L1", "name": "A"}
    ]


def test_merging_unchanged_values_commits_nothing(catalog) -> None:
    """An ingest that sets locations every batch doesn't rewrite the series table."""
    columns = [
        {"name": "series_key", "type": "string", "required": True},
        {"name": "location_key", "type": "string"},
    ]
    table = create(catalog, "series", columns, ["series_key"], wh.Layout())
    schema = wh.iceberg_schema(columns).as_arrow()
    rows_ = pa.table({"series_key": ["a", "b"], "location_key": ["L1", "L2"]}, schema=schema)
    wh.merge(table, rows_, ["series_key"], ["series_key", "location_key"])
    before = state(table)
    wh.merge(table.refresh(), rows_, ["series_key"], ["series_key", "location_key"])
    assert not wh.changes(table, before).changed


def test_commit_files_replaces_in_one_transaction(table) -> None:
    wh.upsert(table, rows(("old", 0, 1.0)))
    paths = [
        wh.write_file(table, rows(("a", 0, 1.0)), "part-0.parquet"),
        wh.write_file(table, rows(("b", 0, 2.0)), "part-1.parquet"),
    ]
    assert stored(table) == [("old", 0, 1.0)]  # written, not committed
    wh.commit_files(table, paths)
    assert stored(table) == [("a", 0, 1.0), ("b", 0, 2.0)]


def test_commit_files_replaces_only_the_filter(table) -> None:
    wh.upsert(table, rows(("a", 0, 1.0), ("b", 0, 2.0)))
    path = wh.write_file(table, rows(("a", 0, 10.0)), "part-a.parquet")
    wh.commit_files(table, [path], overwrite_filter=EqualTo("series_key", "a"))
    assert stored(table) == [("a", 0, 10.0), ("b", 0, 2.0)]


def test_writes_carry_the_producer(table) -> None:
    wh.upsert(table, rows(("a", 0, 1.0)), {wh.PRODUCER: "flow:usgs"})
    summary = table.refresh().current_snapshot().summary
    assert summary is not None and summary[wh.PRODUCER] == "flow:usgs"


def test_conform_refuses_what_doesnt_fit() -> None:
    with pytest.raises(wh.DataMismatch, match="not in the dataset"):
        wh.conform(pa.table({"series_key": ["a"], "other": [1]}), COLUMNS)
    with pytest.raises(wh.DataMismatch, match="missing required"):
        wh.conform(pa.table({"series_key": ["a"]}), COLUMNS)
    naive = pa.table({"series_key": ["a"], "value_time": [datetime(2026, 1, 1)]})
    with pytest.raises(wh.DataMismatch, match="time-zone-aware"):
        wh.conform(naive, COLUMNS)
    out = wh.conform(naive, COLUMNS, timezone="America/New_York")
    assert out["value_time"][0].as_py() == datetime(2026, 1, 1, 5, tzinfo=UTC)
    assert out["value"].null_count == 1


def test_duplicate_keys() -> None:
    assert wh.duplicate_keys(rows(("a", 0, 1.0), ("a", 0, 2.0), ("b", 0, 1.0)), KEY) == 1


# Change detection.


def state(table):
    return wh.table_state(table.refresh())


def test_months_of_each_kind_of_write(table) -> None:
    """The KinD spike's sequence: an upsert across a month boundary, a filtered overwrite, a
    later month."""
    wh.upsert(table, rows(*[("a", d, 1.0) for d in range(31)]))  # January
    first = state(table)

    wh.upsert(table, rows(*[("a", d, 2.0) for d in range(25, 35)]))  # Jan 26 - Feb 4
    second = state(table)
    jan_feb = wh.changes(table, first)
    assert jan_feb.months == ("2026-01", "2026-02")
    assert not jan_feb.added_only  # January's file was rewritten

    days = [("a", d, 3.0) for d in range(9, 14)]  # Jan 10 - 14
    table.refresh().overwrite(rows(*days), overwrite_filter=wh.key_range(rows(*days), KEY))
    third = state(table)
    assert wh.changes(table, second).months == ("2026-01",)

    wh.upsert(table, rows(("a", 160, 4.0)))
    june = wh.changes(table, third)
    assert june.months == ("2026-06",) and june.added_only

    everything = wh.changes(table, first)
    assert everything.months == ("2026-01", "2026-02", "2026-06")


def test_as_of_a_pinned_snapshot(table) -> None:
    wh.upsert(table, rows(("a", 0, 1.0)))
    first = state(table)
    pinned = wh.upsert(table, rows(("a", 40, 1.0)))
    wh.upsert(table, rows(("a", 70, 1.0)))  # after the pin
    assert wh.changes(table.refresh(), first, pinned).months == ("2026-02",)


def test_appends_are_added_only(table) -> None:
    table.append(rows(("a", 0, 1.0)))
    first = state(table)
    table.refresh().append(rows(("b", 40, 1.0)))
    found = wh.changes(table, first)
    assert found.added_only and found.months == ("2026-02",) and found.changed


def test_nothing_changed(table) -> None:
    wh.upsert(table, rows(("a", 0, 1.0)))
    found = wh.changes(table, state(table))
    assert not found.changed and found.months == ()


def test_survives_expired_snapshots(table) -> None:
    """The point of comparing states: history between them can be gone."""
    wh.upsert(table, rows(("a", 0, 1.0)))
    first = state(table)
    wh.upsert(table, rows(("a", 40, 1.0)))
    wh.upsert(table, rows(("a", 70, 1.0)))
    table = table.refresh()
    current = table.current_snapshot().snapshot_id
    old = [s.snapshot_id for s in table.metadata.snapshots if s.snapshot_id != current]
    table.maintenance.expire_snapshots().by_ids(old).commit()
    table = table.refresh()
    assert len(table.metadata.snapshots) == 1
    assert wh.changes(table, first).months == ("2026-02", "2026-03")


def test_deletes_count_their_months(table) -> None:
    wh.upsert(table, rows(("a", 0, 1.0), ("a", 40, 1.0)))
    first = state(table)
    table.refresh().delete(EqualTo("series_key", "a"))  # every row: no new files
    found = wh.changes(table, first)
    assert found.new_files == 0 and not found.added_only
    assert found.months == ("2026-01", "2026-02")


def test_a_rewrite_counts_as_changed_never_missed(table) -> None:
    wh.upsert(table, rows(("a", 0, 1.0)))
    first = state(table)
    data = table.refresh().scan().to_arrow()
    table.refresh().overwrite(data)  # the same rows, new files (as a compaction would)
    assert wh.changes(table, first).months == ("2026-01",)


def test_no_earlier_state(table) -> None:
    assert wh.changes(table, None).unknown == "no earlier state to compare with"


def test_empty_tables(table) -> None:
    empty = state(table)
    assert empty == wh.TableState(sequence=0)
    wh.upsert(table, rows(("a", 0, 1.0)))
    assert wh.changes(table, empty).months == ("2026-01",)


def test_tables_without_the_column(catalog) -> None:
    columns = [{"name": "series_key", "type": "string", "required": True},
               {"name": "location_key", "type": "string"}]  # fmt: skip
    table = create(catalog, "series", columns, ["series_key"], wh.Layout())
    schema = wh.iceberg_schema(columns).as_arrow()
    wh.insert_missing(table, pa.table({"series_key": ["a"], "location_key": [None]}, schema=schema))
    first = state(table)
    wh.insert_missing(
        table, pa.table({"series_key": ["a", "b"], "location_key": [None, None]}, schema=schema)
    )
    added = wh.changes(table, first)
    assert added.changed and added.added_only and added.months == ()
    second = state(table)
    update = pa.table({"series_key": ["a"], "location_key": ["L1"]}, schema=schema)
    wh.merge(table.refresh(), update, ["series_key"], ["series_key", "location_key"])
    assert not wh.changes(table, second).added_only


def test_state_round_trips_through_json(table) -> None:
    import json

    wh.upsert(table, rows(("a", 0, 1.0), ("a", 40, 1.0)))
    recorded = state(table)
    again = wh.TableState.from_json(json.loads(json.dumps(recorded.to_json())))
    assert again == recorded
    assert not wh.changes(table, again).changed


def test_month_range_crosses_years() -> None:
    low, high = datetime(2025, 11, 30, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    assert month_range(low, high) == ["2025-11", "2025-12", "2026-01", "2026-02"]
