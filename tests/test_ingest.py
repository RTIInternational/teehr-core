"""write_timeseries: a producer's write into a registered timeseries dataset."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest
from pyiceberg.catalog.sql import SqlCatalog

from teehr_core import warehouse as wh
from teehr_core.model.dataset_types import series_columns, timeseries_columns
from teehr_core.model.units import UnitRegistry
from teehr_core.warehouse.ingest import columns_of, write_timeseries

T0 = datetime(2026, 1, 31, tzinfo=UTC)
UNITS = UnitRegistry.from_rows([
    {"symbol": "m3/s", "dimension": "volume/time", "to_si": 1.0},
    {"symbol": "[cft_i]/s", "dimension": "volume/time", "to_si": 0.3048**3},
])  # fmt: skip


def _create(catalog, name, columns, key, properties, partition=None):
    schema = wh.iceberg_schema(columns, key)
    layout = wh.Layout(partition=partition, sort=tuple(key), properties=properties)
    return catalog.create_table(
        ("p", name),
        schema=schema,
        partition_spec=wh.partition_spec(schema, layout.partition),
        sort_order=wh.sort_order(schema, layout.sort),
        properties=wh.table_properties(layout),
    )


@pytest.fixture
def catalog(tmp_path):
    """A registered observed dataset (as the control plane makes one): flow, its series
    table, in layer gauges, in m3/s."""
    catalog = SqlCatalog("test", uri=f"sqlite:///{tmp_path}/c.db", warehouse=f"file://{tmp_path}/w")
    catalog.create_namespace("p")
    columns, key = timeseries_columns([], "instant")
    _create(catalog, "flow", columns, key, {
        "teehr.type": "timeseries", "teehr.unit": "m3/s", "teehr.series_table": "flow_series",
        "teehr.location_layer": "gauges",
    }, partition=("value_time", "month"))  # fmt: skip
    _create(catalog, "flow_series", series_columns(), ["series_key"], {"teehr.type": "series"})
    return catalog


def rows(*items):
    """Rows of (series, hours after 2026-01-31 00:00 UTC, value)."""
    return pa.table({
        "series_key": [k for k, _, _ in items],
        "value_time": pa.array([T0 + timedelta(hours=h) for _, h, _ in items],
                               pa.timestamp("us", "UTC")),
        "value": pa.array([v for _, _, v in items], pa.float64()),
    })  # fmt: skip


def stored(catalog, name="flow"):
    return sorted(catalog.load_table(("p", name)).scan().to_arrow().to_pylist(),
                  key=lambda r: tuple(str(v) for v in r.values()))  # fmt: skip


def test_writes_batches_and_adds_their_series(catalog):
    written = write_timeseries(
        catalog, "p", "flow", [rows(("a", 0, 1.0), ("b", 0, 2.0)), rows(("a", 24, 3.0))],
        producer="test", location_of=lambda key: key,
    )  # fmt: skip
    assert (written.rows, written.batches, written.series_added) == (3, 2, 2)
    assert written.months == ["2026-01", "2026-02"]
    assert [(r["series_key"], r["value"]) for r in stored(catalog)] == [
        ("a", 1.0),
        ("a", 3.0),
        ("b", 2.0),
    ]
    series = {r["series_key"]: r for r in stored(catalog, "flow_series")}
    assert series["a"]["location_key"] == "a" and series["a"]["location_layer"] == "gauges"
    table = catalog.load_table(("p", "flow"))
    assert table.current_snapshot().snapshot_id == written.snapshot_id
    assert table.current_snapshot().summary[wh.PRODUCER] == "test"


def test_new_series_get_their_other_fields(catalog):
    write_timeseries(
        catalog,
        "p",
        "flow",
        rows(("a", 0, 1.0), ("b", 0, 2.0)),
        producer="test",
        fields_of=lambda key: {"resolution": "PT6H" if key == "a" else None},
    )
    again = write_timeseries(catalog, "p", "flow", rows(("a", 1, 3.0)), producer="test",
                             fields_of=lambda key: {"resolution": "PT1H"})  # fmt: skip
    assert again.series_added == 0  # an existing series keeps its fields
    series = {r["series_key"]: r["resolution"] for r in stored(catalog, "flow_series")}
    assert series == {"a": "PT6H", "b": None}


def test_each_commit_loads_the_tables_again(catalog, monkeypatch):
    """A catalog's vended storage credentials expire (Polaris: an hour) during long writes."""
    loaded = []
    load = catalog.load_table

    def tracked(identifier):
        loaded.append(load(identifier))
        return loaded[-1]

    class Expired:
        def __getattr__(self, name):
            raise OSError("expired credentials")

    def batches():
        yield rows(("a", 0, 1.0))
        for table in loaded:
            table.io = Expired()
        yield rows(("b", 24, 2.0))

    monkeypatch.setattr(catalog, "load_table", tracked)
    written = write_timeseries(catalog, "p", "flow", batches(), producer="test")
    assert (written.batches, written.series_added) == (2, 2)


def test_writing_again_replaces_and_adds_nothing(catalog):
    write_timeseries(catalog, "p", "flow", rows(("a", 0, 1.0)), producer="test")
    again = write_timeseries(catalog, "p", "flow", rows(("a", 0, 5.0)), producer="test")
    assert again.series_added == 0
    assert [r["value"] for r in stored(catalog)] == [5.0]
    assert len(stored(catalog, "flow_series")) == 1


def test_converts_another_unit(catalog):
    write_timeseries(catalog, "p", "flow", rows(("a", 0, 100.0)), producer="test",
                     unit="[cft_i]/s", units=UNITS)  # fmt: skip
    assert stored(catalog)[0]["value"] == pytest.approx(100 * 0.3048**3)


def test_another_unit_needs_a_registry(catalog):
    with pytest.raises(ValueError, match="no unit registry"):
        write_timeseries(catalog, "p", "flow", rows(("a", 0, 1.0)), producer="test",
                         unit="[cft_i]/s")  # fmt: skip


@pytest.mark.parametrize(
    "batch, problem",
    [
        (rows(("a", 0, 1.0), ("a", 0, 2.0)), "1 key values occur more than once"),
        (rows(("", 0, 1.0)), "empty series key"),
        (rows(("a", 0, 1.0)).append_column("extra", pa.array([1])), "not in the dataset: extra"),
        (rows(("a", 0, 1.0)).drop_columns(["value_time"]), "missing required column"),
    ],
)
def test_refuses_rows_that_dont_fit(catalog, batch, problem):
    with pytest.raises(wh.DataMismatch, match=problem):
        write_timeseries(catalog, "p", "flow", batch, producer="test")
    assert stored(catalog) == [] and stored(catalog, "flow_series") == []


def test_refuses_what_isnt_a_timeseries(catalog):
    with pytest.raises(wh.DataMismatch, match="isn't a timeseries dataset"):
        write_timeseries(catalog, "p", "flow_series", rows(("a", 0, 1.0)), producer="test")


def test_empty_batches_commit_nothing(catalog):
    written = write_timeseries(catalog, "p", "flow", [rows(), rows()], producer="test")
    assert (written.rows, written.batches, written.snapshot_id) == (0, 0, None)


def test_columns_of_round_trips_declared_columns(catalog):
    columns, _ = timeseries_columns(["reference_time", "member"], "instant")
    declared = [{k: c.get(k) for k in ("name", "type", "required", "doc")} for c in columns]
    schema = wh.iceberg_schema(columns)
    table = catalog.create_table(("p", "fc"), schema=schema)
    assert columns_of(table) == declared


def test_location_keys_from_the_layer(catalog):
    from teehr_core.model.dataset_types import layer_columns

    attr = {"name": "is_active", "type": "boolean", "required": False, "doc": None}
    _create(catalog, "gauges", layer_columns([attr]), ["location_key"], {"teehr.type": "layer"})
    point = bytes.fromhex("0101000000000000000000f03f000000000000f03f")
    catalog.load_table(("p", "gauges")).append(pa.table({
        "location_key": ["b", "a", "c"],
        "name": [None, None, None],
        "geometry": [point] * 3,
        **{c: pa.array([1.0] * 3) for c in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")},
        "is_active": [True, True, False],
    }, schema=catalog.load_table(("p", "gauges")).schema().as_arrow()))  # fmt: skip
    assert wh.location_keys(catalog, "p", "flow") == ["a", "b", "c"]
    assert wh.location_keys(catalog, "p", "flow", "is_active == True") == ["a", "b"]
    with pytest.raises(wh.DataMismatch, match="no location layer"):
        wh.location_keys(catalog, "p", "flow_series")


def test_combine_joins_small_batches():
    batches = [rows(("a", 0, 1.0)), rows(), rows(("b", 0, 2.0), ("c", 0, 3.0)), rows(("d", 0, 4.0))]
    assert [t.num_rows for t in wh.combine(batches, rows=2)] == [3, 1]
    assert list(wh.combine([rows()])) == []


def test_staged_months_round_trip(tmp_path):
    staged = wh.stage_by_month([rows(("a", 0, 1.0), ("a", 24, 2.0)), rows(("b", 1, 3.0))],
                               str(tmp_path / "stage"))  # fmt: skip
    assert staged == 3
    months = list(wh.staged_months(str(tmp_path / "stage")))
    assert [sorted(m["series_key"].to_pylist()) for m in months] == [["a", "b"], ["a"]]
    only = list(wh.staged_months(str(tmp_path / "stage"), months=["2026-02"]))
    assert [m.num_rows for m in only] == [1]
    assert list(wh.staged_months(str(tmp_path / "nothing"))) == []


def test_replace_months_replaces_whole_months(catalog):
    write_timeseries(catalog, "p", "flow", rows(("a", 0, 1.0), ("b", 1, 2.0), ("a", 24, 9.0)),
                     producer="test")  # fmt: skip
    written = write_timeseries(catalog, "p", "flow", rows(("a", 2, 5.0)), producer="test",
                               replace_months=True)  # fmt: skip
    assert written.months == ["2026-01"]
    # January is only the new row (b's is gone); February is untouched.
    assert [(r["series_key"], r["value"]) for r in stored(catalog)] == [("a", 5.0), ("a", 9.0)]


def test_replace_months_over_centuries(catalog):
    """A commit of a long record: months far apart (so not one range) stay shallow."""
    months = [datetime(1900 + i // 6, 2 * (i % 6) + 1, 15, tzinfo=UTC) for i in range(1000)]
    many = pa.table({
        "series_key": ["a"] * len(months),
        "value_time": pa.array(months, pa.timestamp("us", "UTC")),
        "value": pa.array([1.0] * len(months), pa.float64()),
    })  # fmt: skip
    write_timeseries(catalog, "p", "flow", many, producer="test")
    written = write_timeseries(catalog, "p", "flow", many, producer="test",
                               replace_months=True)  # fmt: skip
    assert len(written.months) == 1000
    assert len(stored(catalog)) == 1000
