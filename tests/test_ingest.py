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
