"""Ensemble and quantile forecasts, case-level bootstrap, and skill scores."""

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import yaml

from teehr_core import engine, metrics
from teehr_core.engine import Catalog, Dataset
from teehr_core.metrics import MetricRequest, compute_metrics
from teehr_core.model.contract import TIMESTAMP
from teehr_core.spec.models import MetricSpec
from tests.golden_utils import assert_matches
from tests.test_engine import hydro  # noqa: F401  (fixture)

SPECS = Path("tests/specs")
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def golden(golden_dir: Path, kind: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((golden_dir / "metric_vectors.json").read_text())
    return data["probabilistic"][kind]


def spec_of(entry: Any) -> MetricSpec:
    return MetricSpec.model_validate({"name": entry} if isinstance(entry, str) else entry)


@pytest.mark.parametrize("kind", ["ensemble", "ensemble_equal", "quantile"])
def test_registry_matches_golden(golden_dir: Path, kind: str) -> None:
    g = golden(golden_dir, kind)
    rows = g["rows"]
    obs = np.array([r["obs"] for r in rows], dtype=float)
    pred = np.array([r["pred"] for r in rows], dtype=float)
    cols = {"_case": np.array([r["case"] for r in rows], dtype=float)}
    if kind == "quantile":
        cols["quantile"] = np.array([r["quantile"] for r in rows])
    gid = np.zeros(obs.size, dtype=np.int64)
    for entry in g["metric_specs"]:
        spec = spec_of(entry)
        req = MetricRequest(spec.name, None, spec.to_params())
        value = compute_metrics([req], obs, pred, gid, 1, columns=cols)[req][0, 0]
        want = g["expected"][spec.output_name()]
        if want is None:  # not defined for this data (e.g. mixed ensemble sizes)
            assert math.isnan(value), spec.output_name()
        else:
            assert value == pytest.approx(want, rel=1e-12), spec.output_name()


def test_standalone_functions_match_golden(golden_dir: Path) -> None:
    ens = golden(golden_dir, "ensemble")
    cases: dict[int, tuple[float, list[float]]] = {}
    for r in ens["rows"]:
        cases.setdefault(r["case"], (r["obs"], []))[1].append(r["pred"])
    obs = [y for y, _ in cases.values()]
    members = [xs for _, xs in cases.values()]
    assert metrics.crps(obs, members) == pytest.approx(
        ens["expected"]["mean_crps_ensemble"], rel=1e-12
    )
    fair = metrics.crps(obs, members, estimator="fair")
    assert fair == pytest.approx(ens["expected"]["crps_fair"], rel=1e-12)
    brier = metrics.brier_score(obs, members, threshold=3)
    assert brier == pytest.approx(ens["expected"]["mean_brier_score"], rel=1e-12)

    q = golden(golden_dir, "quantile")
    rows = q["rows"]
    o, x, lv = ([r[k] for r in rows] for k in ("obs", "pred", "quantile"))
    assert metrics.pinball_loss(o, x, lv) == pytest.approx(q["expected"]["mean_pinball_loss"])
    lower = [r["pred"] for r in rows if r["quantile"] == 0.05]
    upper = [r["pred"] for r in rows if r["quantile"] == 0.95]
    y = [r["obs"] for r in rows if r["quantile"] == 0.05]
    assert metrics.winkler_score(y, lower, upper) == pytest.approx(
        q["expected"]["mean_interval_score"]
    )
    assert metrics.interval_coverage(y, lower, upper) == q["expected"]["interval_coverage"]
    assert metrics.interval_width(y, lower, upper) == pytest.approx(
        q["expected"]["mean_interval_width"]
    )
    # A deterministic forecast is a one-member ensemble: CRPS = MAE.
    assert metrics.crps([1, 2, 3], [[2], [2], [5]]) == pytest.approx(
        metrics.mean_absolute_error([1, 2, 3], [2, 2, 5])
    )  # type: ignore[arg-type]


def catalog(rows: list[dict[str, Any]], kind: str) -> Catalog:
    """An observed and a forecast dataset built from golden long-format rows."""
    times = {c: T0 + timedelta(hours=c) for c in sorted({r["case"] for r in rows})}
    seen: dict[int, float] = {}
    for r in rows:
        seen.setdefault(r["case"], r["obs"])
    obs = pa.table(
        {
            "series_key": ["o1"] * len(seen),
            "value_time": pa.array([times[c] for c in seen], TIMESTAMP),
            "value": [float(v) for v in seen.values()],
        }
    )
    fc: dict[str, Any] = {
        "series_key": ["f1"] * len(rows),
        "reference_time": pa.array([T0 - timedelta(hours=1)] * len(rows), TIMESTAMP),
        "value_time": pa.array([times[r["case"]] for r in rows], TIMESTAMP),
        "value": [float(r["pred"]) for r in rows],
    }
    if kind.startswith("ensemble"):
        counter: dict[int, int] = {}
        member = []
        for r in rows:
            counter[r["case"]] = counter.get(r["case"], 0) + 1
            member.append(str(counter[r["case"]]))
        fc["member"] = member
    else:
        fc["quantile"] = [float(r["quantile"]) for r in rows]
    xw = pa.table({"left_key": ["o1"], "right_key": ["f1"]})
    common = {"variable": "v", "unit": "m", "statistic": "inst"}
    return Catalog(
        [
            Dataset("obs", "observed", obs, **common),  # type: ignore[arg-type]
            Dataset("fc", "forecast", pa.table(fc), **common),  # type: ignore[arg-type]
            Dataset("xw", "auxiliary", xw, kind="crosswalk"),
        ]
    )


@pytest.mark.parametrize("kind", ["ensemble", "ensemble_equal", "quantile"])
def test_engine_matches_golden(golden_dir: Path, kind: str) -> None:
    g = golden(golden_dir, kind)
    spec = {
        "version": 1,
        "name": "p",
        "inputs": {"primary": {"dataset": "obs"}, "secondary": [{"dataset": "fc"}]},
        "pair": {"relation": "xw"},
        "aggregate": [{"group_by": ["primary_series_key"], "metrics": g["metric_specs"]}],
    }
    row = engine.evaluate(spec, catalog(g["rows"], kind)).table.to_pylist()[0]
    for entry in g["metric_specs"]:
        out = spec_of(entry).output_name()
        assert_matches(row[out], g["expected"][out], out)


def test_whole_cases_are_resampled(golden_dir: Path) -> None:
    """Case-level weights equal explicitly resampling whole forecasts (members together)."""
    for kind, names in (
        ("ensemble", ["mean_crps_ensemble", "mean_brier_score"]),
        (
            "quantile",
            [
                "mean_pinball_loss",
                "interval_coverage",
                "mean_interval_width",
                "mean_interval_score",
            ],
        ),
    ):
        rows = golden(golden_dir, kind)["rows"]
        obs = np.array([r["obs"] for r in rows], dtype=float)
        pred = np.array([r["pred"] for r in rows], dtype=float)
        case = np.array([r["case"] for r in rows])
        cols = {"_case": case.astype(float)}
        if kind == "quantile":
            cols["quantile"] = np.array([r["quantile"] for r in rows])
        counts = np.array([2, 0, 1, 1])  # case draws
        weights = counts[case][None, :].astype(float)
        idx = np.concatenate(
            [np.flatnonzero(case == c) for c in range(4) for _ in range(counts[c])]
        )
        # Copies of a case get distinct case keys, as resampled forecasts would.
        copy = np.concatenate(
            [
                np.full((case == c).sum(), k)
                for k, c in enumerate(c for c in range(4) for _ in range(counts[c]))
            ]
        )
        gid = np.zeros(obs.size, dtype=np.int64)
        for name in names:
            params = {"threshold": 3.0} if name == "mean_brier_score" else {}
            req = MetricRequest(name, None, metrics.Params(**params))
            got = compute_metrics([req], obs, pred, gid, 1, weights, cols)[req][0, 0]
            want_cols = {k: v[idx] for k, v in cols.items()} | {"_case": copy.astype(float)}
            want = compute_metrics([req], obs[idx], pred[idx], gid[idx], 1, columns=want_cols)[req][
                0, 0
            ]
            assert got == pytest.approx(want, rel=1e-12), name


def test_ensemble_bootstrap_through_the_engine(golden_dir: Path) -> None:
    g = golden(golden_dir, "ensemble")
    spec = {
        "version": 1,
        "name": "p",
        "inputs": {"primary": {"dataset": "obs"}, "secondary": [{"dataset": "fc"}]},
        "pair": {"relation": "xw"},
        "aggregate": [
            {
                "group_by": ["primary_series_key"],
                "metrics": ["mean_crps_ensemble"],
                "bootstrap": {"method": "iid", "reps": 200, "seed": 1, "min_count": 4},
            }
        ],
    }
    row = engine.evaluate(spec, catalog(g["rows"], "ensemble")).table.to_pylist()[0]
    assert (
        row["mean_crps_ensemble_q05"] <= row["mean_crps_ensemble"] <= row["mean_crps_ensemble_q95"]
    )
    # min_count counts forecast cases (4), not rows (15).
    spec["aggregate"][0]["bootstrap"]["min_count"] = 5  # type: ignore[index]
    row = engine.evaluate(spec, catalog(g["rows"], "ensemble")).table.to_pylist()[0]
    assert row["mean_crps_ensemble_q05"] is None


def test_skill_scores(hydro: Catalog) -> None:  # noqa: F811
    base = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())
    ref = "nwm31_medium_range_mem1"
    stage = {
        "group_by": ["primary_series_key", "secondary_dataset"],
        "metrics": ["root_mean_square_error", "mean_crps_ensemble", "kling_gupta_efficiency"],
        "skill": {"reference": ref, "metrics": ["root_mean_square_error", "mean_crps_ensemble"]},
    }
    rows = engine.evaluate({**base, "aggregate": [stage]}, hydro).table.to_pylist()
    by = {(r["primary_series_key"], r["secondary_dataset"]): r for r in rows}
    for (gage, ds), r in by.items():
        reference = by[(gage, ref)]
        assert r["root_mean_square_error_skill"] == pytest.approx(
            1 - r["root_mean_square_error"] / reference["root_mean_square_error"]
        )
        assert r["mean_crps_ensemble_skill"] == pytest.approx(
            1 - r["mean_crps_ensemble"] / reference["mean_crps_ensemble"]
        )
        if ds == ref:
            assert r["root_mean_square_error_skill"] == 0
    stage["skill"]["form"] = "ratio"  # type: ignore[index]
    rows = engine.evaluate({**base, "aggregate": [stage]}, hydro).table.to_pylist()
    r = rows[0]
    assert r["root_mean_square_error_ratio"] == pytest.approx(
        r["root_mean_square_error"] / by[(r["primary_series_key"], ref)]["root_mean_square_error"]
    )


def test_skill_rules() -> None:
    from pydantic import ValidationError

    from teehr_core.spec import load_spec

    base = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())

    def with_stage(**stage: Any) -> dict[str, Any]:
        return {
            **base,
            "aggregate": [
                {"metrics": ["root_mean_square_error", "kling_gupta_efficiency"], **stage}
            ],
        }

    ref = "nwm31_medium_range_mem1"
    with pytest.raises(ValidationError, match="need secondary_dataset in group_by"):
        load_spec(with_stage(skill={"reference": ref, "metrics": ["root_mean_square_error"]}))
    grouped = {"group_by": ["secondary_dataset"]}
    with pytest.raises(ValidationError, match="is not a secondary input"):
        load_spec(
            with_stage(
                **grouped, skill={"reference": "nope", "metrics": ["root_mean_square_error"]}
            )
        )
    with pytest.raises(ValidationError, match="needs an error score"):
        load_spec(
            with_stage(**grouped, skill={"reference": ref, "metrics": ["kling_gupta_efficiency"]})
        )
    ok = load_spec(
        with_stage(**grouped, skill={"reference": ref, "metrics": ["root_mean_square_error"]})
    )
    assert ok.aggregate[0].output_columns() == [
        "secondary_dataset",
        "root_mean_square_error",
        "kling_gupta_efficiency",
        "root_mean_square_error_skill",
    ]


def _rows_of(rows: list[dict[str, Any]], gage: str, dataset: str) -> list[dict[str, Any]]:
    return [
        p for p in rows if p["primary_series_key"] == gage and p["secondary_dataset"] == dataset
    ]


def _rmse_on(rows: list[dict[str, Any]], times: set[Any]) -> float:
    subset = [p for p in rows if p["value_time"] in times]
    value = metrics.rmse(
        [p["primary_value"] for p in subset], [p["secondary_value"] for p in subset]
    )
    assert value is not None
    return value


def test_matched_skill(hydro: Catalog) -> None:  # noqa: F811
    """Matched skill equals scoring both datasets on the observations both forecast."""
    base = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())
    ref = "nwm31_medium_range_mem1"
    stage = {
        "group_by": ["primary_series_key", "secondary_dataset"],
        "metrics": ["root_mean_square_error", "mean_crps_ensemble"],
        "skill": {
            "reference": ref,
            "metrics": ["root_mean_square_error", "mean_crps_ensemble"],
            "matched": True,
        },
    }
    spec = {**base, "aggregate": [stage], "output": {"include_pairs": True}}
    result = engine.evaluate(spec, hydro)
    assert result.pairs is not None
    rows = result.pairs.to_pylist()
    table = result.table.to_pylist()
    by = {(r["primary_series_key"], r["secondary_dataset"]): r for r in table}
    differs = False
    for (gage, ds), r in by.items():
        if ds == ref:
            assert r["root_mean_square_error_skill"] == 0
            continue
        mine, theirs = _rows_of(rows, gage, ds), _rows_of(rows, gage, ref)
        common = {p["value_time"] for p in mine} & {p["value_time"] for p in theirs}
        want = 1 - _rmse_on(mine, common) / _rmse_on(theirs, common)
        assert r["root_mean_square_error_skill"] == pytest.approx(want, rel=1e-12)
        unmatched = 1 - r["root_mean_square_error"] / by[(gage, ref)]["root_mean_square_error"]
        differs |= not math.isclose(want, unmatched)
    assert differs, "fixture should have datasets covering different times"


@pytest.mark.parametrize("kind", ["ensemble", "ensemble_equal"])
def test_standalone_ensemble_diagnostics(golden_dir: Path, kind: str) -> None:
    g = golden(golden_dir, kind)
    cases: dict[int, tuple[float, list[float]]] = {}
    for r in g["rows"]:
        cases.setdefault(r["case"], (r["obs"], []))[1].append(r["pred"])
    obs = [y for y, _ in cases.values()]
    members = [xs for _, xs in cases.values()]
    want = g["expected"]
    for name in ("ensemble_spread", "ensemble_mean_root_mean_square_error", "spread_skill_ratio"):
        assert getattr(metrics, name)(obs, members) == pytest.approx(want[name], rel=1e-12)
    counts = metrics.rank_histogram(obs, members)
    for k in range(4):
        assert counts[k] / len(obs) == pytest.approx(want[f"rank_{k}"])
    reliability = metrics.rank_histogram_reliability(obs, members)
    if want["rank_histogram_reliability"] is None:
        assert reliability is None
    else:
        assert reliability == pytest.approx(want["rank_histogram_reliability"], rel=1e-12)


@pytest.mark.parametrize("kind", ["ensemble", "ensemble_equal", "quantile"])
def test_state_by_hour_equals_the_pairs(golden_dir: Path, kind: str) -> None:
    """Probabilistic metrics kept as additive sums: one state row per case's hour, merged."""
    from teehr_core.engine.duckdb_backend import DuckDBBackend
    from teehr_core.metrics.state import mergeable
    from teehr_core.spec import load_spec

    g = golden(golden_dir, kind)
    entries = [
        e
        for e in g["metric_specs"]
        if mergeable([MetricRequest(spec_of(e).name, None, spec_of(e).to_params())])
    ]
    assert len(entries) >= 4, entries
    spec = load_spec(
        {
            "version": 1,
            "name": "p",
            "inputs": {"primary": {"dataset": "obs"}, "secondary": [{"dataset": "fc"}]},
            "pair": {"relation": "xw"},
            "aggregate": [{"group_by": ["primary_series_key"], "metrics": entries}],
            "output": {"include_pairs": True},
        }
    )
    assert DuckDBBackend.state_blockers(spec) == []
    pairs = engine.evaluate(spec, catalog(g["rows"], kind)).pairs
    assert pairs is not None
    backend = DuckDBBackend()
    want = backend.aggregate(spec, pairs, stages=1)[0].to_pylist()
    state = backend.first_stage_state(spec, pairs, "hour")
    assert state.num_rows > 1  # several parts merged
    got = backend.first_stage_from_state(spec, state).to_pylist()
    for a, b in zip(got, want, strict=True):
        for k, v in b.items():
            assert a[k] == (pytest.approx(v, rel=1e-12) if isinstance(v, float) else v), k
