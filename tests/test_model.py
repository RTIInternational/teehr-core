"""The data model: dataset types and units (conversion over a supplied unit table)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from teehr_core.model.dataset_types import (
    COLUMN_TYPES,
    arrow_schema,
    crosswalk_columns,
    layer_columns,
    series_columns,
    timeseries_columns,
)
from teehr_core.model.units import (
    IDENTITY,
    Conversion,
    Unit,
    UnitError,
    UnitRegistry,
    conversion_between,
    merge,
)

# --- dataset types ------------------------------------------------------------------


def test_timeseries_columns_and_key() -> None:
    columns, key = timeseries_columns([], "instant")
    assert [c["name"] for c in columns] == ["series_key", "value_time", "value"]
    assert key == ["series_key", "value_time"]
    columns, key = timeseries_columns(
        ["quality", "member", "reference_time"],
        "absolute",
        [{"name": "flag", "type": "boolean", "required": False}],
    )
    assert [c["name"] for c in columns] == [
        "series_key", "reference_time", "value_time", "member", "value", "quality",
        "period_start", "period_end", "flag",
    ]  # fmt: skip
    assert key == ["series_key", "reference_time", "value_time", "member"]


def test_arrow_schema() -> None:
    columns, _ = timeseries_columns(["quantile"], "instant")
    schema = arrow_schema(columns)
    assert schema.field("value_time").type == pa.timestamp("us", tz="UTC")
    assert schema.field("quantile").type == pa.decimal128(5, 4)
    assert not schema.field("series_key").nullable and schema.field("value").nullable
    assert schema.field("value").metadata == {b"doc": b"Null: a missing value"}


def test_every_type_has_an_arrow_type() -> None:
    declared = [
        *timeseries_columns(["reference_time", "member", "quantile", "quality"], "absolute")[0],
        *series_columns(),
        *layer_columns(),
        *crosswalk_columns(),
    ]
    assert {c["type"] for c in declared} <= set(COLUMN_TYPES)
    arrow_schema([{"name": t, "type": t, "required": False} for t in COLUMN_TYPES])


# --- units (VeriRace's tests, on a unit table supplied as data) -------------------------


def test_flow_conversions(units: UnitRegistry) -> None:
    assert units.conversion("[cft_i]/s", "m3/s").scale == 0.028316846592
    assert units.conversion("m3/s", "[cft_i]/s").scale == 1 / 0.028316846592
    assert units.conversion("MW", "MW").is_identity
    assert units.conversion("kW", "MW").scale == pytest.approx(1e-3)


def test_aliases(units: UnitRegistry) -> None:
    assert units.get("cfs").symbol == "[cft_i]/s"
    assert units.conversion("ft3/s", "[cft_i]/s").is_identity
    assert "cms" in units and "furlongs/fortnight" not in units


def test_temperature_offsets(units: UnitRegistry) -> None:
    c_to_f = units.conversion("Cel", "[degF]")
    assert c_to_f.apply(100.0) == pytest.approx(212)
    assert c_to_f.apply(0.0) == pytest.approx(32)
    assert units.conversion("degF", "K").apply(32.0) == pytest.approx(273.15)
    # Differences take the scale alone: a 5 K ramp is a 5 Cel ramp, 9 degF.
    assert units.conversion("K", "Cel").of_differences.apply(5.0) == 5.0
    assert units.conversion("K", "Cel").apply(300.0) == pytest.approx(26.85)
    assert units.conversion("K", "[degF]").of_differences.apply(5.0) == pytest.approx(9)


def test_apply_to_arrays(units: UnitRegistry) -> None:
    conv = units.conversion("Cel", "[degF]")
    np.testing.assert_allclose(conv.apply(np.array([0.0, 100.0])), [32, 212])
    got = conv.apply(pa.chunked_array([[0, None], [100]]))
    assert got.type == pa.float64()
    np.testing.assert_allclose(got.to_numpy(zero_copy_only=False), [32, np.nan, 212])
    assert IDENTITY.apply(values := pa.array([1, 2])) is values


def test_errors(units: UnitRegistry) -> None:
    with pytest.raises(UnitError, match="cannot convert"):
        units.conversion("MW", "m3/s")
    with pytest.raises(UnitError, match="unknown unit"):
        units.get("furlongs/fortnight")
    with pytest.raises(UnitError, match="defined twice"):
        UnitRegistry([Unit("m", "length", 1.0), Unit("m", "length", 1.0)])
    with pytest.raises(UnitError, match="no unit registry"):
        conversion_between(None, "m", "mm")
    assert conversion_between(None, "m", "m") is IDENTITY


def test_merge_public_and_project_units(units: UnitRegistry) -> None:
    project = UnitRegistry([Unit("[acr_us].[ft_i]", "volume", 1233.48)])
    both = merge([units, project])
    assert len(both) == len(units) + 1 and "[acr_us].[ft_i]" in both
    with pytest.raises(UnitError, match="defined twice"):
        merge([units, units])


def test_control_plane_rows_without_aliases() -> None:
    rows = [{"symbol": "m", "dimension": "length", "to_si": 1.0, "offset": None}]
    registry = UnitRegistry.from_rows(rows)
    assert registry.get("m") == Unit("m", "length", 1.0)
    assert registry.conversion("m", "m") == Conversion(1.0, 0.0)
