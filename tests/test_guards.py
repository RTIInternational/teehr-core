"""Guards: null metrics on thin or flat data, with a reason."""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from teehr_core import engine
from teehr_core.model.units import UnitRegistry
from teehr_core.spec import load_spec
from tests.engine_support import SPECS

CATALOG = "tests/fixtures/hydro/datasets.yaml"


def _rows(units: UnitRegistry, **stage: Any) -> list[dict[str, Any]]:
    spec: dict[str, Any] = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())
    spec["aggregate"] = [
        {
            "group_by": ["primary_series_key", "secondary_dataset"],
            "metrics": ["count", "root_mean_square_error", "nash_sutcliffe_efficiency"],
            **stage,
        }
    ]
    rows: list[dict[str, Any]] = engine.evaluate(spec, CATALOG, units).table.to_pylist()
    return rows


def test_min_count_nulls_every_metric(units: UnitRegistry) -> None:
    rows = _rows(units, guards=[{"min_count": 1_000_000}])
    for r in rows:
        assert (
            r["root_mean_square_error"] is None
            and r["nash_sutcliffe_efficiency"] is None
            and r["count"] is None
        )
        assert r["null_reason"] == f"all metrics: pairs {_count(units, r)} < 1e+06"


def _count(units: UnitRegistry, row: dict[str, Any]) -> int:
    plain = {(r["primary_series_key"], r["secondary_dataset"]): r["count"] for r in _rows(units)}
    return int(plain[(row["primary_series_key"], row["secondary_dataset"])])


def test_guard_on_some_metrics_with_intervals(units: UnitRegistry) -> None:
    rows = _rows(
        units,
        guards=[{"metrics": ["nash_sutcliffe_efficiency"], "min_obs_std": 1e12}, {"min_count": 1}],
        bootstrap={"method": "iid", "reps": 20, "seed": 1, "min_count": 2},
    )
    for r in rows:
        assert (
            r["nash_sutcliffe_efficiency"] is None
            and r["nash_sutcliffe_efficiency_q05"] is None
            and r["nash_sutcliffe_efficiency_q95"] is None
        )
        assert (
            r["root_mean_square_error"] is not None and r["root_mean_square_error_q05"] is not None
        )
        assert r["null_reason"].startswith("nash_sutcliffe_efficiency: obs std ") and r[
            "null_reason"
        ].endswith("< 1e+12")


def test_untripped_guards_leave_no_reason(units: UnitRegistry) -> None:
    rows = _rows(units, guards=[{"min_count": 1, "min_obs_std": 0}])
    assert all(
        r["null_reason"] is None and r["nash_sutcliffe_efficiency"] is not None for r in rows
    )


def test_guard_spec_checks() -> None:
    base: dict[str, Any] = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())
    for stage in (
        {"guards": [{}]},
        {"guards": [{"metrics": ["kling_gupta_efficiency"], "min_count": 2}]},
    ):
        base["aggregate"] = [
            {"group_by": ["primary_series_key"], "metrics": ["root_mean_square_error"], **stage}
        ]
        with pytest.raises(ValidationError):
            load_spec(base)
    base["aggregate"] = [
        {"group_by": ["primary_series_key"], "metrics": ["root_mean_square_error"]},
        {
            "group_by": [],
            "metrics": [{"name": "mean", "of": "root_mean_square_error"}],
            "guards": [{"min_count": 2}],
        },
    ]
    with pytest.raises(ValidationError, match="first stage"):
        load_spec(base)
