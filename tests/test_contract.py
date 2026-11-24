from datetime import UTC, datetime

import pyarrow as pa
import pytest

from teehr_core.model.contract import (
    TIMESTAMP,
    ContractError,
    normalize_crosswalk,
    normalize_series_catalog,
    normalize_timeseries,
)
from teehr_core.model.units import UnitRegistry

T0 = datetime(2026, 10, 1, tzinfo=UTC)
T1 = datetime(2026, 10, 1, 1, tzinfo=UTC)


def ts(**overrides: object) -> pa.Table:
    cols: dict[str, object] = {
        "series_key": pa.array(["a", "a"]),
        "value_time": pa.array([T0, T1], pa.timestamp("ns", tz="America/New_York")),
        "value": pa.array([1, 2], pa.int32()),
    }
    cols.update(overrides)
    return pa.table(cols)


def test_normalizes_types_and_keeps_variable_and_unit_off_rows() -> None:
    out = normalize_timeseries(ts(), role="observed", variable="streamflow", unit="m3/s")
    assert out.schema.field("value_time").type == TIMESTAMP
    assert out.schema.field("value").type == pa.float64()
    assert "variable" not in out.column_names and "unit" not in out.column_names
    # Columns with the dataset's variable and unit are checked, then dropped.
    given = ts(variable=pa.array(["streamflow"] * 2), unit=pa.array(["m3/s"] * 2))
    out = normalize_timeseries(given, role="observed", variable="streamflow", unit="m3/s")
    assert out.column_names == ["series_key", "value_time", "value"]


def test_extra_columns_pass_through() -> None:
    out = normalize_timeseries(
        ts(qualifier=pa.array(["ICE", None])), role="observed", variable="v", unit="m"
    )
    assert out["qualifier"].to_pylist() == ["ICE", None]


def test_reports_all_problems_at_once() -> None:
    bad = ts(value=pa.array(["x", "y"]), value_time=pa.array([T0, T1], pa.timestamp("us")))
    with pytest.raises(ContractError) as info:
        normalize_timeseries(bad, role="observed")
    text = "\n".join(info.value.problems)
    assert "must be numeric" in text
    assert "timezone-aware" in text
    assert "variable" in text and "unit" in text


def test_forecast_requires_reference_time() -> None:
    with pytest.raises(ContractError, match="reference_time"):
        normalize_timeseries(ts(), role="forecast", variable="v", unit="m")


def test_duplicate_keys_rejected() -> None:
    dupes = ts(value_time=pa.array([T0, T0], TIMESTAMP))
    with pytest.raises(ContractError, match="duplicate"):
        normalize_timeseries(dupes, role="observed", variable="v", unit="m")


def test_forecast_duplicates_consider_reference_time() -> None:
    table = ts(
        value_time=pa.array([T1, T1], TIMESTAMP),
        reference_time=pa.array([T0, datetime(2026, 9, 30, tzinfo=UTC)], TIMESTAMP),
    )
    out = normalize_timeseries(table, role="forecast", variable="v", unit="m")
    assert out.num_rows == 2


def test_unknown_and_mixed_units_rejected(units: UnitRegistry) -> None:
    with pytest.raises(ContractError, match="unknown unit"):
        normalize_timeseries(ts(), role="observed", variable="v", unit="furlong", units=units)
    # Without a registry, any single unit passes (the warehouse always checks).
    assert normalize_timeseries(ts(), role="observed", variable="v", unit="furlong").num_rows
    with pytest.raises(ContractError, match="one unit"):
        normalize_timeseries(
            ts(unit=pa.array(["m", "mm"])), role="observed", variable="v", unit="m"
        )
    with pytest.raises(ContractError, match="not the dataset's 'v'"):
        normalize_timeseries(
            ts(variable=pa.array(["w", "w"])), role="observed", variable="v", unit="m"
        )


def test_series_catalog_checks() -> None:
    ok = normalize_series_catalog(
        pa.table({"series_key": ["a", "b"], "period": ["hour_starting", None]})
    )
    assert ok.num_rows == 2
    with pytest.raises(ContractError, match="duplicate"):
        normalize_series_catalog(pa.table({"series_key": ["a", "a"]}))


def test_crosswalk_checks() -> None:
    assert normalize_crosswalk(pa.table({"left_key": ["a"], "right_key": ["x"]})).num_rows == 1
    with pytest.raises(ContractError, match="right_key"):
        normalize_crosswalk(pa.table({"left_key": ["a"]}))


def test_quantile_levels_have_four_decimal_places() -> None:
    """Levels compare exactly across systems: rounded to 4 places, finer ones refused."""
    noisy = pa.array([0.05 + 1e-12, 0.975], pa.float64())
    out = normalize_timeseries(ts(quantile=noisy), role="observed", variable="v", unit="m")
    assert out["quantile"].to_pylist() == [0.05, 0.975]
    bad = pa.array([0.12345, 1.5], pa.float64())
    with pytest.raises(ContractError) as err:
        normalize_timeseries(ts(quantile=bad), role="observed", variable="v", unit="m")
    text = str(err.value)
    assert "more than 4 decimal places" in text and "outside 0 to 1" in text
