"""End-to-end engine tests on the committed fixtures."""

import json
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import yaml

from teehr_core import engine
from teehr_core.engine import Catalog, Dataset, PlanError
from teehr_core.model.contract import TIMESTAMP
from teehr_core.model.periods import PeriodRegistry
from teehr_core.model.units import UnitRegistry
from tests.golden_utils import assert_matches

SPECS = Path(__file__).parent / "specs"


@pytest.fixture(scope="module")
def hydro(fixtures_dir: Path, units: UnitRegistry, periods: PeriodRegistry) -> Catalog:
    return Catalog.from_yaml(fixtures_dir / "hydro" / "datasets.yaml", units, periods)


@pytest.fixture(scope="module")
def energy(fixtures_dir: Path, units: UnitRegistry, periods: PeriodRegistry) -> Catalog:
    return Catalog.from_yaml(fixtures_dir / "energy" / "datasets.yaml", units, periods)


@pytest.mark.parametrize("fixture", ["hydro", "energy"])
@pytest.mark.parametrize("grouping", ["by_location", "by_lead_time_bin"])
def test_matches_golden(
    fixture: str, grouping: str, golden_dir: Path, request: pytest.FixtureRequest
) -> None:
    catalog: Catalog = request.getfixturevalue(fixture)
    golden = json.loads((golden_dir / f"{fixture}_metrics.json").read_text())
    expected = golden["evaluations"][grouping]
    group_by = expected["group_by"]

    # Every golden metric (all registered metrics and parameter variants), one stage.
    spec = yaml.safe_load((SPECS / f"{fixture}_{grouping}.yaml").read_text())
    spec["aggregate"] = [{"group_by": group_by, "metrics": golden["metric_specs"]}]
    result = engine.evaluate(spec, catalog)

    got = {tuple(r[k] for k in group_by): r for r in result.table.to_pylist()}
    want = {tuple(g["keys"][k] for k in group_by): g["metrics"] for g in expected["groups"]}
    assert got.keys() == want.keys()
    for key, metrics in want.items():
        for name, value in metrics.items():
            assert_matches(got[key][name], value, (key, name))


def test_pairs_convert_units_and_keep_extras(hydro: Catalog, golden_dir: Path) -> None:
    spec = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())
    assert engine.evaluate(spec, hydro).pairs is None

    pairs = engine.evaluate({**spec, "output": {"include_pairs": True}}, hydro).pairs
    assert pairs is not None
    golden = json.loads((golden_dir / "hydro_metrics.json").read_text())
    assert pairs.num_rows == golden["pair_count"]
    assert set(pairs["unit"].to_pylist()) == {"[cft_i]/s"}
    assert "primary_quality" in pairs.column_names
    assert pairs.schema.field("value_time").type == TIMESTAMP


def spec_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "version": 1,
        "name": "t",
        "inputs": {
            "primary": {"dataset": "usgs_streamflow"},
            "secondary": {"dataset": "nwm31_short_range"},
        },
        "pair": {"relation": "usgs_nwm31_crosswalk"},
        "aggregate": [{"group_by": ["primary_series_key"], "metrics": ["count"]}],
    }
    base.update(overrides)
    return base


def test_multi_stage_and_filter(hydro: Catalog) -> None:
    result = engine.evaluate(
        spec_dict(
            fields=[
                {"name": "lead_time_bin", "fn": "lead_time_bins", "width": "PT6H"},
                {"name": "high", "expr": "primary_value > 1000"},
            ],
            filter="high",
            aggregate=[
                {
                    "group_by": ["primary_series_key", "lead_time_bin"],
                    "metrics": ["count", "root_mean_square_error"],
                },
                {
                    "group_by": ["primary_series_key"],
                    "metrics": [
                        {"name": "mean", "of": "root_mean_square_error"},
                        {"name": "sum", "of": "count"},
                    ],
                },
            ],
        ),
        hydro,
    )
    rows = result.table.to_pylist()
    assert [r["primary_series_key"] for r in rows] == ["usgs-01646500"]  # only gage above 1000 cfs
    first = result.stages[0].to_pylist()
    assert result.table.schema.field("sum_count").type == pa.int64()
    assert rows[0]["sum_count"] == sum(r["count"] for r in first)
    assert rows[0]["mean_root_mean_square_error"] == pytest.approx(
        sum(r["root_mean_square_error"] for r in first) / len(first)
    )


def test_lead_time_bin_edges(hydro: Catalog) -> None:
    result = engine.evaluate(
        spec_dict(
            fields=[{"name": "b", "fn": "lead_time_bins", "edges": ["PT0H", "PT3H", "PT12H"]}],
            aggregate=[{"group_by": ["b"], "metrics": ["count"]}],
        ),
        hydro,
    )
    counts = dict(
        zip(result.table["b"].to_pylist(), result.table["count"].to_pylist(), strict=True)
    )
    # Short-range leads are 1..18 h, 3 gages x 2 runs.
    assert counts == {"[0h, 3h)": 2 * 3 * 2, "[3h, 12h)": 9 * 3 * 2, "[12h, inf)": 7 * 3 * 2}


@pytest.mark.parametrize(
    ("field", "want"),
    [
        ({"part": "year"}, ["2024", "2024", "2025", "2025", None]),
        ({"part": "month"}, ["2024-09", "2024-10", "2025-01", "2025-12", None]),
        ({"part": "water_year"}, ["2024", "2025", "2025", "2026", None]),
        ({"part": "water_year", "year_start_month": 1}, ["2024", "2024", "2025", "2025", None]),
        ({"part": "quarter"}, ["2024-Q3", "2024-Q4", "2025-Q1", "2025-Q4", None]),
        (
            {"part": "quarter", "year_start_month": 10},
            ["2024-Q4", "2025-Q1", "2025-Q2", "2026-Q1", None],
        ),
        ({"part": "season"}, ["SON", "SON", "DJF", "DJF", None]),
    ],
)
def test_calendar_labels(field: dict[str, Any], want: list[str | None]) -> None:
    import duckdb

    from teehr_core.engine.duckdb_backend import calendar_sql
    from teehr_core.spec import CalendarField

    times = [
        datetime(2024, 9, 30, 23),
        datetime(2024, 10, 1, 0),
        datetime(2025, 1, 15),
        datetime(2025, 12, 31),
        None,
    ]
    t = pa.table({"value_time": pa.array(times, TIMESTAMP)})
    f = CalendarField.model_validate({"name": "c", "fn": "calendar", **field})
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")  # as the engine's connection does
    con.register("t", t)
    assert [r[0] for r in con.execute(f"SELECT {calendar_sql(f)} FROM t").fetchall()] == want


def test_calendar_field_groups_by_reference_time(hydro: Catalog) -> None:
    result = engine.evaluate(
        spec_dict(
            fields=[{"name": "q", "fn": "calendar", "part": "quarter", "of": "reference_time"}],
            aggregate=[{"group_by": ["q"], "metrics": ["count"]}],
        ),
        hydro,
    )
    labels = result.table["q"].to_pylist()
    assert labels and all(re.fullmatch(r"\d{4}-Q[1-4]", x) for x in labels)
    total = engine.evaluate(spec_dict(aggregate=[{"group_by": [], "metrics": ["count"]}]), hydro)
    assert sum(result.table["count"].to_pylist()) == total.table["count"][0].as_py()


def test_conditions_copy_pairs_into_overlapping_levels(hydro: Catalog) -> None:
    seen = engine.evaluate(spec_dict(output={"include_pairs": True}), hydro).pairs
    assert seen is not None
    obs = sorted(seen["primary_value"].to_pylist())
    mid, high = obs[len(obs) // 2], obs[len(obs) * 9 // 10]  # nested, overlapping levels
    levels = {"above_mid": f"primary_value > {mid}", "above_high": f"primary_value > {high}"}
    conditions = {"name": "flow", "fn": "conditions", "levels": levels}
    by_level = engine.evaluate(
        spec_dict(fields=[conditions], aggregate=[{"group_by": ["flow"], "metrics": ["count"]}]),
        hydro,
    )
    flows, ns = by_level.table["flow"].to_pylist(), by_level.table["count"].to_pylist()
    counts = dict(zip(flows, ns, strict=True))

    def count(where: str | None) -> int:
        result = engine.evaluate(
            spec_dict(filter=where, aggregate=[{"group_by": [], "metrics": ["count"]}]), hydro
        )
        return int(result.table["count"][0].as_py())

    # Each level holds exactly the pairs meeting its condition; "all" holds every pair.
    assert counts == {
        "all": count(None),
        "above_mid": count(f"primary_value > {mid}"),
        "above_high": count(f"primary_value > {high}"),
    }
    assert counts["above_high"] < counts["above_mid"] < counts["all"]
    # filter runs after the copies, so it can keep one level.
    one = engine.evaluate(spec_dict(fields=[conditions], filter="flow = 'above_high'"), hydro)
    assert sum(one.table["count"].to_pylist()) == counts["above_high"]
    # Without 'all', pairs meeting no condition are gone.
    no_all = engine.evaluate(
        spec_dict(
            fields=[{**conditions, "all": False}],
            aggregate=[{"group_by": ["flow"], "metrics": ["count"]}],
        ),
        hydro,
    )
    assert sorted(no_all.table["flow"].to_pylist()) == ["above_high", "above_mid"]


def test_conditions_copy_after_series_fields(hydro: Catalog) -> None:
    """Exceedance ranks each series once, not its copies, whatever the field order."""
    exceed = {"name": "p", "fn": "exceedance"}
    conditions = {"name": "flow", "fn": "conditions", "levels": {"high": "p <= 0.5"}}
    plain = engine.evaluate(spec_dict(fields=[exceed], output={"include_pairs": True}), hydro)
    copied = engine.evaluate(
        spec_dict(
            fields=[conditions, exceed],
            filter="flow = 'all'",
            output={"include_pairs": True},
        ),
        hydro,
    )
    assert plain.pairs is not None and copied.pairs is not None
    assert sorted(copied.pairs["p"].to_pylist()) == sorted(plain.pairs["p"].to_pylist())


def test_rollup_in_the_run_adds_coarser_rows(hydro: Catalog) -> None:
    metrics = ["count", "root_mean_square_error", "kling_gupta_efficiency"]
    rolled = engine.evaluate(
        spec_dict(
            aggregate=[{"group_by": [], "rollup": ["primary_series_key"], "metrics": metrics}]
        ),
        hydro,
    )
    rows = rolled.table.to_pylist()
    whole = [r for r in rows if r["grouping"] == "all"]
    per_gage = [r for r in rows if r["grouping"] == "primary_series_key"]
    assert len(whole) == 1 and whole[0]["primary_series_key"] == "(all)"
    assert len(per_gage) == 3 and all(r["primary_series_key"].startswith("usgs-") for r in per_gage)
    # The "all" row is the metrics over every pair, as if never grouped.
    alone = engine.evaluate(spec_dict(aggregate=[{"group_by": [], "metrics": metrics}]), hydro)
    for m in metrics:
        assert whole[0][m] == pytest.approx(alone.table[m][0].as_py())
    assert whole[0]["count"] == sum(r["count"] for r in per_gage)


def test_rollup_of_calendar_fields_names_each_set(hydro: Catalog) -> None:
    fields = [
        {"name": "wy", "fn": "calendar", "part": "water_year"},
        {"name": "q", "fn": "calendar", "part": "quarter", "year_start_month": 10},
    ]
    result = engine.evaluate(
        spec_dict(
            fields=fields,
            aggregate=[
                {"group_by": ["secondary_dataset"], "rollup": ["wy", "q"], "metrics": ["count"]}
            ],
        ),
        hydro,
    )
    rows = result.table.to_pylist()
    assert {r["grouping"] for r in rows} == {"q", "wy", "all"}
    for r in rows:
        assert (r["q"] == "(all)") == (r["grouping"] != "q")
        assert (r["wy"] == "(all)") == (r["grouping"] == "all")
    by_set = {g: sum(r["count"] for r in rows if r["grouping"] == g) for g in ("q", "wy", "all")}
    assert by_set["q"] == by_set["wy"] == by_set["all"]  # each set covers every pair once


REDUCE_FIELDS = [{"name": "bin", "fn": "lead_time_bins", "width": "PT6H"}]
WINDOW = ["primary_series_key", "secondary_dataset", "reference_time", "bin"]


def test_reduce_scores_one_pair_per_window_and_statistic(hydro: Catalog) -> None:
    raw = engine.evaluate(
        spec_dict(fields=REDUCE_FIELDS, output={"include_pairs": True}), hydro
    ).pairs
    assert raw is not None
    windows: dict[tuple[Any, ...], list[tuple[float, float]]] = {}
    for r in raw.to_pylist():
        windows.setdefault(tuple(r[k] for k in WINDOW), []).append(
            (r["primary_value"], r["secondary_value"])
        )
    reduced = engine.evaluate(
        spec_dict(
            fields=REDUCE_FIELDS,
            reduce={"by": WINDOW, "stats": ["mean", "max"]},
            aggregate=[{"group_by": ["stat"], "metrics": ["count"]}],
            output={"include_pairs": True},
        ),
        hydro,
    )
    counts = dict(
        zip(reduced.table["stat"].to_pylist(), reduced.table["count"].to_pylist(), strict=True)
    )
    assert counts == {"max": len(windows), "mean": len(windows)}
    assert reduced.pairs is not None
    for r in reduced.pairs.to_pylist():
        vals = windows[tuple(r[k] for k in WINDOW)]
        if r["stat"] == "max":
            assert r["primary_value"] == max(v[0] for v in vals)
            assert r["secondary_value"] == max(v[1] for v in vals)
        else:
            assert r["primary_value"] == pytest.approx(sum(v[0] for v in vals) / len(vals))


def test_reduce_keeps_condition_levels_apart_and_rolls_up(hydro: Catalog) -> None:
    conditions = {"name": "flow", "fn": "conditions", "levels": {"none": "primary_value < 0"}}
    result = engine.evaluate(
        spec_dict(
            fields=[*REDUCE_FIELDS, conditions],
            reduce={"by": WINDOW, "stats": ["max"]},
            aggregate=[
                {
                    "group_by": ["flow", "stat"],
                    "rollup": ["bin"],
                    "metrics": ["count", "kling_gupta_efficiency"],
                }
            ],
        ),
        hydro,
    )
    rows = result.table.to_pylist()
    # No pair is below zero: the "none" level has no windows; "all" keeps every window.
    assert {r["flow"] for r in rows} == {"all"}
    whole = [r for r in rows if r["grouping"] == "all"]
    per_bin = [r for r in rows if r["grouping"] == "bin"]
    assert len(whole) == 1 and whole[0]["count"] == sum(r["count"] for r in per_bin)


def test_lead_time_bins_sort_by_time(hydro: Catalog) -> None:
    """Bins come out in lead-time order ([6h, 12h) before [12h, 18h)), in every stage."""
    result = engine.evaluate(
        spec_dict(
            fields=[{"name": "b", "fn": "lead_time_bins", "width": "PT3H"}],
            aggregate=[
                {"group_by": ["primary_series_key", "b"], "metrics": ["count"]},
                {"group_by": ["b"], "metrics": [{"name": "sum", "of": "count"}]},
            ],
        ),
        hydro,
    )
    want = [
        "[0h, 3h)",
        "[3h, 6h)",
        "[6h, 9h)",
        "[9h, 12h)",
        "[12h, 15h)",
        "[15h, 18h)",
        "[18h, 21h)",
    ]
    assert result.table["b"].to_pylist() == want
    first = result.stages[0].to_pylist()
    assert [
        r["b"] for r in first if r["primary_series_key"] == first[0]["primary_series_key"]
    ] == want


def test_no_group_by_gives_one_row(energy: Catalog) -> None:
    result = engine.evaluate(
        {
            "version": 1,
            "name": "all",
            "inputs": {
                "primary": {"dataset": "load_actual"},
                "secondary": {"dataset": "load_dayahead"},
            },
            "pair": {"on_attribute": "zone"},
            "aggregate": [{"metrics": ["count", "nash_sutcliffe_efficiency"]}],
        },
        energy,
    )
    assert result.table.num_rows == 1
    assert result.table["count"][0].as_py() == 288


def test_input_filter(hydro: Catalog) -> None:
    result = engine.evaluate(
        spec_dict(
            inputs={
                "primary": {"dataset": "usgs_streamflow"},
                "secondary": {
                    "dataset": "nwm31_short_range",
                    "filter": "hour(reference_time) = 12",
                },
            }
        ),
        hydro,
    )
    assert result.table["count"].to_pylist() == [18, 18, 18]


def test_plan_errors(hydro: Catalog) -> None:
    with pytest.raises(PlanError, match="not in the catalog"):
        engine.evaluate(
            spec_dict(inputs={"primary": {"dataset": "nope"}, "secondary": {"dataset": "x"}}),
            hydro,
        )
    with pytest.raises(PlanError, match="role"):
        engine.evaluate(
            spec_dict(
                inputs={
                    "primary": {"dataset": "nwm31_short_range"},
                    "secondary": {"dataset": "nwm31_short_range"},
                }
            ),
            hydro,
        )
    with pytest.raises(PlanError, match="group_by columns not found"):
        engine.evaluate(spec_dict(aggregate=[{"group_by": ["nope"], "metrics": ["count"]}]), hydro)
    with pytest.raises(PlanError, match="overwrite"):
        engine.evaluate(spec_dict(fields=[{"name": "primary_value", "expr": "1"}]), hydro)


def _energy_with(catalog: Catalog, **changes: Any) -> Catalog:
    fc = catalog["load_dayahead"]
    return Catalog([catalog["load_actual"], replace(fc, **changes)], catalog.units, catalog.periods)


ENERGY_SPEC = {
    "version": 1,
    "name": "e",
    "inputs": {"primary": {"dataset": "load_actual"}, "secondary": {"dataset": "load_dayahead"}},
    "pair": {"on_attribute": "zone"},
    "aggregate": [{"group_by": ["primary_series_key"], "metrics": ["count"]}],
}


def test_unit_conversion_across_power_units(energy: Catalog) -> None:
    fc = energy["load_dayahead"]
    kw = fc.table.set_column(
        fc.table.schema.get_field_index("value"),
        "value",
        pa.compute.multiply(fc.table["value"], 1000.0),
    )  # a dataset's unit is metadata, not a column
    catalog = _energy_with(energy, table=kw, unit="kW")
    a = engine.evaluate(
        {**ENERGY_SPEC, "aggregate": [{"metrics": ["root_mean_square_error"]}]}, energy
    )
    b = engine.evaluate(
        {**ENERGY_SPEC, "aggregate": [{"metrics": ["root_mean_square_error"]}]}, catalog
    )
    assert b.table["root_mean_square_error"][0].as_py() == pytest.approx(
        a.table["root_mean_square_error"][0].as_py(), rel=1e-12
    )


def test_input_filter_numbers_are_in_the_canonical_unit(energy: Catalog) -> None:
    """An input filter keeps the same rows whatever unit its dataset is stored in."""
    fc = energy["load_dayahead"]
    kw = fc.table.set_column(
        fc.table.schema.get_field_index("value"),
        "value",
        pa.compute.multiply(fc.table["value"], 1000.0),
    )
    threshold = float(pa.compute.approximate_median(fc.table["value"]).as_py())
    spec = {
        **ENERGY_SPEC,
        "inputs": {
            "primary": {"dataset": "load_actual", "filter": f"value > {threshold}"},
            "secondary": {"dataset": "load_dayahead", "filter": f"value > {threshold}"},
        },
        "aggregate": [{"metrics": ["count"]}],
    }
    variables = {"load": "MW"}  # the canonical unit
    in_mw = Catalog(list(energy), energy.units, energy.periods, variables)
    in_kw = Catalog(
        [energy["load_actual"], replace(fc, table=kw, unit="kW")],
        energy.units,
        energy.periods,
        variables,
    )
    a = engine.evaluate(spec, in_mw).table["count"][0].as_py()
    b = engine.evaluate(spec, in_kw).table["count"][0].as_py()
    assert 0 < a == b
    # Read in the dataset's own unit (kW), the same number keeps every forecast.
    unregistered = Catalog(list(in_kw), energy.units, energy.periods)
    assert engine.evaluate(spec, unregistered).table["count"][0].as_py() > a


def test_incompatible_units_and_statistics_rejected(energy: Catalog) -> None:
    fc = energy["load_dayahead"]
    with pytest.raises(PlanError, match="cannot pair"):
        engine.evaluate(ENERGY_SPEC, _energy_with(energy, table=fc.table, unit="m3/s"))
    with pytest.raises(PlanError, match="statistics differ"):
        engine.evaluate(ENERGY_SPEC, _energy_with(energy, statistic="max"))


def test_values_pair_only_on_equal_windows(energy: Catalog) -> None:
    """A series naming its own period (ADR-0009): a day doesn't pair with an hour."""
    fc = energy["load_dayahead"]
    assert fc.series is not None
    mixed = fc.series.append_column("period", pa.array(["hour_starting", "day_starting"]))
    result = engine.evaluate(ENERGY_SPEC, _energy_with(energy, series=mixed))
    assert result.table["primary_series_key"].to_pylist() == ["meter-north"]


def test_hours_labelled_by_start_pair_with_hours_labelled_by_end(energy: Catalog) -> None:
    """The same hour pairs whichever end its value time marks; value times alone don't."""
    fc = energy["load_dayahead"]
    shifted = fc.table.set_column(
        fc.table.schema.get_field_index("value_time"),
        "value_time",
        pa.compute.add(fc.table["value_time"], pa.scalar(timedelta(hours=1))),
    )
    spec = {
        **ENERGY_SPEC,
        "aggregate": [{"group_by": ["primary_series_key"], "metrics": ["root_mean_square_error"]}],
    }
    spec["output"] = {"include_pairs": True}
    base = engine.evaluate(spec, energy)
    ending = engine.evaluate(spec, _energy_with(energy, table=shifted, period="hour_ending"))
    assert ending.table.to_pylist() == base.table.to_pylist()
    assert base.pairs is not None and ending.pairs is not None
    windows = [
        "period_start",
        "period_end",
    ]
    assert base.pairs.select(windows).equals(ending.pairs.select(windows))
    first = base.pairs.slice(0, 1).to_pylist()[0]
    assert first["period_start"] == first["value_time"]
    assert first["period_end"] - first["period_start"] == timedelta(hours=1)
    # Labelled by its end but called hour_starting, each value is an hour off.
    off = engine.evaluate(spec, _energy_with(energy, table=shifted))
    assert (
        off.table["root_mean_square_error"].to_pylist()
        != base.table["root_mean_square_error"].to_pylist()
    )


def test_periods_checked_when_planned(energy: Catalog, hydro: Catalog) -> None:
    with pytest.raises(PlanError, match="unknown period 'fortnight'"):
        engine.evaluate(ENERGY_SPEC, _energy_with(energy, period="fortnight"))
    with pytest.raises(PlanError, match="no period"):
        engine.evaluate(ENERGY_SPEC, _energy_with(energy, period=None))
    with pytest.raises(PlanError, match="only inst values cover an instant"):
        engine.evaluate(ENERGY_SPEC, _energy_with(energy, period="instant"))
    obs = replace(hydro["usgs_streamflow"], period="hour_starting")
    catalog = Catalog([obs, *(d for d in hydro if d.name != obs.name)], hydro.units, hydro.periods)
    spec = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())
    with pytest.raises(PlanError, match="'inst' values can't cover period 'hour_starting'"):
        engine.evaluate(spec, catalog)


def test_simulated_dataset_without_reference_time(energy: Catalog) -> None:
    fc = energy["load_dayahead"]
    one_run = fc.table.filter(
        pa.compute.equal(
            fc.table["reference_time"], pa.scalar(datetime(2026, 9, 1, 10, tzinfo=UTC), TIMESTAMP)
        )
    ).drop_columns(["reference_time"])
    catalog = _energy_with(energy, table=one_run, role="simulated")
    result = engine.evaluate(ENERGY_SPEC, catalog)
    assert result.table["count"].to_pylist() == [24, 24]
    with pytest.raises(PlanError, match="reference_time"):
        engine.evaluate(
            {**ENERGY_SPEC, "fields": [{"name": "b", "fn": "lead_time_bins", "width": "PT6H"}]},
            catalog,
        )


def test_dataset_metadata_validated() -> None:
    table = pa.table(
        {
            "series_key": ["a"],
            "value_time": pa.array([datetime(2026, 1, 1, tzinfo=UTC)], TIMESTAMP),
            "value": [1.0],
        }
    )
    with pytest.raises(engine.ContractError, match="statistic"):
        Catalog([Dataset("d", "observed", table, variable="v", unit="m", statistic=None)])


def test_pred_dataset_names_the_predicted_dataset(hydro: Catalog) -> None:
    spec = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())
    pairs = engine.evaluate({**spec, "output": {"include_pairs": True}}, hydro).pairs
    assert pairs is not None
    assert set(pairs["secondary_dataset"].to_pylist()) == {
        "nwm31_short_range",
        "nwm31_medium_range_mem1",
    }

    # A predicted input's own "dataset" column doesn't replace it.
    fc = hydro["nwm31_short_range"]
    tagged = replace(
        fc, table=fc.table.append_column("dataset", pa.array(["x"] * fc.table.num_rows))
    )
    catalog = Catalog(
        [hydro["usgs_streamflow"], hydro["usgs_nwm31_crosswalk"], tagged], hydro.units
    )
    one = {**spec, "inputs": {**spec["inputs"], "secondary": [{"dataset": "nwm31_short_range"}]}}
    pairs = engine.evaluate({**one, "output": {"include_pairs": True}}, catalog).pairs
    assert pairs is not None and set(pairs["secondary_dataset"].to_pylist()) == {
        "nwm31_short_range"
    }


BOOTSTRAP = {"method": "circular_block", "block": "P1D", "reps": 200, "seed": 7, "min_count": 20}


def test_bootstrap_through_the_engine(hydro: Catalog) -> None:
    spec = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())
    spec["aggregate"] = [
        {
            "group_by": ["primary_series_key", "secondary_dataset"],
            "metrics": [
                "root_mean_square_error",
                "kling_gupta_efficiency",
                {"name": "nash_sutcliffe_efficiency", "as": "nse_x"},
            ],
            "bootstrap": BOOTSTRAP,
        },
        {
            "group_by": ["secondary_dataset"],
            "metrics": [{"name": "median", "of": "kling_gupta_efficiency_q05"}],
        },
    ]
    result = engine.evaluate(spec, hydro)
    first = result.stages[0]
    for col in (
        "root_mean_square_error_q05",
        "root_mean_square_error_q95",
        "kling_gupta_efficiency_q05",
        "kling_gupta_efficiency_q95",
        "nse_x_q05",
        "nse_x_q95",
    ):
        assert col in first.column_names
    rows = [r for r in first.to_pylist() if r["root_mean_square_error_q05"] is not None]
    assert rows
    for r in rows:
        assert (
            r["root_mean_square_error_q05"]
            <= r["root_mean_square_error"]
            <= r["root_mean_square_error_q95"]
        )
    assert "median_kling_gupta_efficiency_q05" in result.table.column_names
    again = engine.evaluate(spec, hydro).stages[0]
    assert again.equals(first)


def test_bootstrap_spec_rules() -> None:
    from pydantic import ValidationError

    from teehr_core.spec import load_spec

    base = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())
    with pytest.raises(ValidationError, match="needs 'block'"):
        load_spec(
            {
                **base,
                "aggregate": [
                    {"metrics": ["root_mean_square_error"], "bootstrap": {"method": "stationary"}}
                ],
            }
        )
    with pytest.raises(ValidationError, match="only allowed in the first stage"):
        load_spec(
            {
                **base,
                "aggregate": [
                    {"group_by": ["secondary_dataset"], "metrics": ["root_mean_square_error"]},
                    {
                        "metrics": [{"name": "mean", "of": "root_mean_square_error"}],
                        "bootstrap": {"method": "iid"},
                    },
                ],
            }
        )
    ok = load_spec(
        {
            **base,
            "aggregate": [
                {
                    "metrics": ["kling_gupta_efficiency"],
                    "bootstrap": {"method": "iid", "quantiles": [0.025, 0.975]},
                }
            ],
        }
    )
    assert ok.aggregate[0].output_columns() == [
        "kling_gupta_efficiency",
        "kling_gupta_efficiency_q2_5",
        "kling_gupta_efficiency_q97_5",
    ]


def test_count_gets_no_bootstrap_interval() -> None:
    from teehr_core.spec import load_spec

    base = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())
    stage = {"metrics": ["count", "root_mean_square_error"], "bootstrap": {"method": "iid"}}
    spec = load_spec({**base, "aggregate": [stage]})
    assert spec.aggregate[0].output_columns() == [
        "count",
        "root_mean_square_error",
        "root_mean_square_error_q05",
        "root_mean_square_error_q95",
    ]


def test_metric_option_rules(hydro: Catalog) -> None:
    from pydantic import ValidationError

    from teehr_core.spec import load_spec

    base = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())

    def stage(*metrics: Any) -> dict[str, Any]:
        return {
            **base,
            "aggregate": [{"group_by": ["secondary_dataset"], "metrics": list(metrics)}],
        }

    with pytest.raises(ValidationError, match="'probability_of_detection' needs \\['threshold'\\]"):
        load_spec(stage("probability_of_detection"))
    cfs = lambda v: {"value": v, "unit": "[cft_i]/s"}  # noqa: E731
    with pytest.raises(
        ValidationError, match="'root_mean_square_error' doesn't take \\['threshold'\\]"
    ):
        load_spec(stage({"name": "root_mean_square_error", "threshold": cfs(3)}))
    with pytest.raises(ValidationError, match="doesn't take a transform"):
        load_spec(
            stage({"name": "critical_success_index", "threshold": cfs(3), "transform": "log"})
        )
    with pytest.raises(ValidationError, match="duplicate output columns"):
        load_spec(
            stage(
                {"name": "probability_of_detection", "threshold": cfs(1)},
                {"name": "probability_of_detection", "threshold": cfs(2)},
            )
        )
    # A value carries its unit: a bare number is refused, a wrong unit fails the plan.
    with pytest.raises(ValidationError, match="3 needs its unit"):
        load_spec(stage({"name": "probability_of_detection", "threshold": 3}))
    with pytest.raises(ValidationError, match="needs its unit"):
        load_spec(stage({"name": "normalized_mean_absolute_error", "normalize_by": 150}))
    with pytest.raises(PlanError, match=r"threshold \(3.0 MW\)"):
        engine.evaluate(
            stage({"name": "probability_of_detection", "threshold": {"value": 3, "unit": "MW"}}),
            hydro,
        )
    with pytest.raises(PlanError, match="metric columns not found: \\['nope'\\]"):
        engine.evaluate(
            stage({"name": "probability_of_detection", "threshold": {"column": "nope"}}), hydro
        )

    # A threshold column from a field equals the same fixed threshold.
    by_field = {
        **stage({"name": "critical_success_index", "threshold": {"column": "flood"}}),
        "fields": [{"name": "flood", "expr": "2000.0"}],
    }
    fixed = engine.evaluate(
        stage({"name": "critical_success_index", "threshold": cfs(2000)}), hydro
    ).table
    assert (
        engine.evaluate(by_field, hydro)
        .table.column("critical_success_index")
        .equals(fixed.column("critical_success_index"))
    )
    # The same threshold in another unit is converted to the evaluation unit (cfs here).
    metric = {
        "name": "critical_success_index",
        "threshold": {"value": 2000 * 0.028316846592, "unit": "m3/s"},
    }
    si = engine.evaluate(stage(metric), hydro).table
    assert si.column("critical_success_index").equals(fixed.column("critical_success_index"))


def test_only_the_same_variable_pairs(energy: Catalog) -> None:
    """Each dataset has one variable; pairing different ones is refused when planning."""
    with pytest.raises(PlanError, match="only the same variable pairs"):
        engine.evaluate(ENERGY_SPEC, _energy_with(energy, variable="price"))


@pytest.mark.parametrize(
    ("bootstrap", "blocker"),
    [
        ({"block": "P1D"}, None),
        ({"block": "P2M", "method": "stationary"}, None),
        ({"method": "year_block"}, None),
        ({"method": "iid"}, "iid bootstrap intervals resample individual pairs"),
        ({"block": "PT12H"}, "a bootstrap block of 'PT12H' isn't whole days or months"),
        ({"block": 24}, "a bootstrap block of 24 rows isn't whole days or months"),
        ({"block": "P1D", "sort_by": ["reference_time"]}, "orders pairs by more than value_time"),
    ],
)
def test_bootstrap_from_state_blockers(bootstrap: dict[str, Any], blocker: str | None) -> None:
    from teehr_core.engine.duckdb_backend import DuckDBBackend
    from teehr_core.spec import load_spec

    base = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())
    stage = {
        "group_by": ["primary_series_key"],
        "metrics": ["root_mean_square_error", "kling_gupta_efficiency"],
        "bootstrap": bootstrap,
    }
    blockers = DuckDBBackend.state_blockers(load_spec({**base, "aggregate": [stage]}))
    if blocker is None:
        assert blockers == []
    else:
        assert len(blockers) == 1 and blocker in blockers[0], blockers


@pytest.mark.parametrize(
    "bootstrap",
    [
        {"block": "P1D", "reps": 100, "seed": 3, "min_count": 5},
        {"block": "P1M", "method": "stationary", "reps": 100, "seed": 3, "min_count": 5},
    ],
)
def test_bootstrap_from_state_equals_the_pairs(hydro: Catalog, bootstrap: dict[str, Any]) -> None:
    """Intervals from state per day equal the pair path's, rollup rows and transforms too."""
    from teehr_core.engine.duckdb_backend import DuckDBBackend
    from teehr_core.spec import load_spec

    base = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())
    stage = {
        "group_by": ["secondary_dataset"],
        "rollup": ["primary_series_key"],
        "metrics": [
            "count",
            "root_mean_square_error",
            "kling_gupta_efficiency",
            "maximum",
            {"name": "root_mean_square_error", "transform": "log"},
            {
                "name": "critical_success_index",
                "threshold": {"value": 2000, "unit": "[cft_i]/s"},
            },  # additive sums
            "relative_median",  # kept as quantile sketches
            {"name": "flow_duration_curve_slope", "side": "secondary"},
        ],
        "bootstrap": bootstrap,
    }
    spec = load_spec({**base, "aggregate": [stage], "output": {"include_pairs": True}})
    assert DuckDBBackend.state_blockers(spec) == []
    pairs = engine.evaluate(spec, hydro).pairs
    assert pairs is not None
    backend = DuckDBBackend()
    want = backend.aggregate(spec, pairs, stages=1)[0]
    got = backend.first_stage_from_state(spec, backend.first_stage_state(spec, pairs, "day"))
    assert got.column_names == want.column_names
    assert got.num_rows == want.num_rows and any(
        r["root_mean_square_error_q95"] for r in want.to_pylist()
    )
    for a, b in zip(got.to_pylist(), want.to_pylist(), strict=True):
        for k, v in b.items():
            assert a[k] == (pytest.approx(v, rel=1e-10) if isinstance(v, float) else v), k


def test_ramp_thresholds_and_normalizers_convert_by_scale_alone() -> None:
    """A 5 K ramp is a 5 Cel ramp; a 300 K threshold on values is 26.85 Cel."""
    from teehr_core.engine.plan import quantities_in
    from teehr_core.model.units import Unit, UnitRegistry
    from teehr_core.spec import load_spec

    units = UnitRegistry(
        [Unit("K", "temperature", 1.0), Unit("Cel", "temperature", 1.0, offset=273.15)]
    )
    kelvin = lambda v: {"value": v, "unit": "K"}  # noqa: E731
    spec = load_spec(
        {
            "version": 1,
            "name": "t",
            "inputs": {"primary": {"dataset": "obs"}, "secondary": [{"dataset": "fc"}]},
            "pair": {"on_attribute": "series_key"},
            "fields": [
                {"name": "d_obs", "fn": "change", "side": "primary", "window": "PT1H"},
                {"name": "d_pred", "fn": "change", "side": "secondary", "window": "PT1H"},
                {"name": "hot", "fn": "events", "threshold": kelvin(300)},
            ],
            "aggregate": [
                {
                    "metrics": [
                        {"name": "probability_of_detection", "threshold": kelvin(300)},
                        {
                            "name": "probability_of_detection",
                            "threshold": kelvin(5),
                            "values": {"primary": "d_obs", "secondary": "d_pred"},
                            "as": "ramp_pod",
                        },
                        {"name": "normalized_mean_absolute_error", "normalize_by": kelvin(10)},
                    ]
                }
            ],
        }
    )
    out = quantities_in(spec, units, "Cel")
    (ev,) = [f for f in out.fields if f.name == "hot"]
    level, ramp, nmae = out.aggregate[0].metrics
    assert ev.threshold.value == pytest.approx(26.85)  # type: ignore[union-attr]
    assert level.threshold.value == pytest.approx(26.85)  # type: ignore[union-attr]
    assert ramp.threshold.value == 5.0 and nmae.normalize_by.value == 10.0  # type: ignore[union-attr]
    assert {ev.threshold.unit, ramp.threshold.unit} == {"Cel"}  # type: ignore[union-attr]


def test_transform_offset_converts_by_scale_to_the_evaluation_unit(hydro: Catalog) -> None:
    """The observations are in cfs: an offset of 1 cfs is an offset of 0.0283... m3/s."""

    def run(offset: dict[str, Any], unit: str | None = None) -> list[Any]:
        spec: dict[str, Any] = {
            "version": 1,
            "name": "o",
            "inputs": {
                "primary": {"dataset": "usgs_streamflow"},
                "secondary": {"dataset": "nwm31_short_range"},
            },
            "pair": {"relation": "usgs_nwm31_crosswalk"},
            "aggregate": [
                {
                    "group_by": ["primary_series_key"],
                    "metrics": [
                        "count",
                        {"name": "rmse", "transform": {"name": "log", "offset": offset}},
                    ],
                }
            ],
            **({"output": {"unit": unit}} if unit else {}),
        }
        table = engine.evaluate(spec, hydro).table.sort_by("primary_series_key")
        return table["root_mean_square_error_log"].to_pylist()

    in_cfs = run({"value": 1, "unit": "[cft_i]/s"})
    assert run({"value": 0.028316846592, "unit": "m3/s"}) == pytest.approx(in_cfs, rel=1e-12)
    # Differences of logs don't change with the unit when the offset scales with it.
    assert run({"value": 1, "unit": "[cft_i]/s"}, unit="m3/s") == pytest.approx(in_cfs, rel=1e-9)
