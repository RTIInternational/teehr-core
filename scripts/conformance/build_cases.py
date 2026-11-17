"""Build the shared conformance cases (teehr-shared-eval-spec ``conformance/cases``).

    uv run python scripts/conformance/build_cases.py ../teehr-shared-eval-spec/conformance

Writes each case's inputs, spec and expected results (computed by teehr-core's engine,
whose metrics are checked against exact golden values in this repo's tests), and the
unit table the cases use (``units.csv``). Re-running rewrites the cases it builds and
leaves others alone.
"""

from __future__ import annotations

import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from teehr_core.conformance import Case, check_case, engine_evaluate, load_periods, load_units

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
SPECS = ROOT / "tests" / "specs"
TIMESTAMP = pa.timestamp("us", tz="UTC")
ADDED = "2026-11-17"


def _write_case(
    out: Path,
    name: str,
    description: str,
    spec: dict[str, Any],
    inputs: dict[str, Any] | Path,
    about: str | None = None,
) -> Path:
    path = out / "cases" / name
    shutil.rmtree(path, ignore_errors=True)
    (path / "inputs").mkdir(parents=True)
    (path / "expected").mkdir()
    meta = {"description": description, "added": ADDED}
    (path / "case.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, width=100))
    spec_file = "invalid-spec.yaml" if about else "spec.yaml"
    (path / spec_file).write_text(yaml.safe_dump(spec, sort_keys=False, width=100))
    if isinstance(inputs, Path):
        for f in inputs.iterdir():
            if f.suffix in (".parquet", ".yaml"):
                shutil.copy(f, path / "inputs" / f.name)
    else:
        catalog: dict[str, Any] = {"datasets": {}}
        for ds, entry in inputs.items():
            entry = dict(entry)
            table = entry.pop("table")
            pq.write_table(table, path / "inputs" / f"{ds}.parquet")
            catalog["datasets"][ds] = {**entry, "file": f"{ds}.parquet"}
        (path / "inputs" / "datasets.yaml").write_text(yaml.safe_dump(catalog, sort_keys=False))
    if about:
        (path / "expected" / "error.yaml").write_text(yaml.safe_dump({"about": about}))
    return path


def _expect(path: Path, units: Any, periods: Any) -> pa.Table:
    """Compute and write a case's expected results; check the case then passes."""
    spec = yaml.safe_load((path / "spec.yaml").read_text())
    table = engine_evaluate(spec, path / "inputs", units, periods)
    pq.write_table(table, path / "expected" / "results.parquet")
    outcome = check_case(Case.load(path), engine_evaluate, units, periods)
    assert outcome.passed, outcome.problems
    return table


# --- small hand-made datasets -----------------------------------------------------------


def _t(hours: int) -> datetime:
    return datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=hours)


def _series(rows: list[tuple[str, int, float | None]]) -> pa.Table:
    return pa.table(
        {
            "series_key": [r[0] for r in rows],
            "value_time": pa.array([_t(r[1]) for r in rows], TIMESTAMP),
            "value": pa.array([r[2] for r in rows], pa.float64()),
        }
    )


#: obs and sim per series: a control, constant obs, one pair, and gaps in sim.
OBS = [
    *(("normal", h, v) for h, v in enumerate([3.0, 5.0, 4.0, 8.0, 6.0, 2.0])),
    *(("constant_obs", h, 4.0) for h in range(6)),
    ("one_pair", 0, 7.0),
    *(("sim_gaps", h, v) for h, v in enumerate([1.0, 2.0, 3.0, 5.0])),
]
SIM = [
    *(("normal", h, v) for h, v in enumerate([2.5, 5.5, 3.0, 9.0, 6.5, 1.0])),
    *(("constant_obs", h, v) for h, v in enumerate([3.0, 5.0, 4.0, 4.5, 3.5, 4.0])),
    ("one_pair", 0, 6.0),
    *(("sim_gaps", h, v) for h, v in enumerate([1.5, None, 2.5, None])),
]
KEYS = ["normal", "constant_obs", "one_pair", "sim_gaps"]


def _small_inputs(sim_unit: str = "m3/s", scale: float = 1.0) -> dict[str, Any]:
    sim = [(k, h, None if v is None else v * scale) for k, h, v in SIM]
    common = {"variable": "streamflow", "statistic": "inst"}
    return {
        "obs": {"role": "observed", "unit": "m3/s", "table": _series(OBS), **common},
        "sim": {"role": "simulated", "unit": sim_unit, "table": _series(sim), **common},
        "xw": {
            "role": "auxiliary",
            "kind": "crosswalk",
            "table": pa.table({"left_key": KEYS, "right_key": KEYS}),
        },
    }


def _small_spec(name: str, metrics: list[Any], **extra: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "name": name,
        "inputs": {"primary": {"dataset": "obs"}, "secondary": {"dataset": "sim"}},
        "pair": {"relation": "xw"},
        "aggregate": [{"group_by": ["primary_series_key"], "metrics": metrics}],
        **extra,
    }


def build(out: Path) -> None:
    """Build every case into ``out`` (the spec repo's ``conformance`` directory)."""
    shutil.copy(FIXTURES / "units.csv", out / "units.csv")
    units = load_units(out / "units.csv")
    periods = load_periods(out.parent / "standards" / "periods.csv")

    for fixture, spec_name in (("hydro", "hydro_by_location"), ("energy", "energy_by_location")):
        spec = yaml.safe_load((SPECS / f"{spec_name}.yaml").read_text())
        path = _write_case(
            out,
            f"{fixture}_metrics",
            f"Every deterministic, signature and probabilistic metric the {fixture} fixture's "
            "spec asks for, by location and forecast: real data with gaps, pairs through a "
            "crosswalk, units converted at pairing.",
            spec,
            FIXTURES / fixture,
        )
        _expect(path, units, periods)

    undefined = [
        "count",
        "root_mean_square_error",
        "nash_sutcliffe_efficiency",
        "kling_gupta_efficiency",
        "pearson_correlation",
        "relative_bias",
    ]
    path = _write_case(
        out,
        "undefined_results",
        "Undefined results are null: constant observations (NSE, KGE, Pearson), one pair "
        "(Pearson, KGE). Missing secondary values leave their pairs out (count 2 of 4). A "
        "control series is defined throughout.",
        _small_spec("undefined", undefined),
        _small_inputs(),
    )
    table = _expect(path, units, periods)
    rows = {r["primary_series_key"]: r for r in table.to_pylist()}
    assert all(v is not None for v in rows["normal"].values())
    assert rows["constant_obs"]["nash_sutcliffe_efficiency"] is None
    assert rows["one_pair"]["pearson_correlation"] is None
    assert rows["sim_gaps"]["count"] == 2

    threshold = [
        {"name": "hits", "threshold": {"value": 4.5, "unit": "m3/s"}, "as": "hits_m3s"},
        {
            "name": "hits",
            "threshold": {"value": 4.5 / 0.028316846592, "unit": "[cft_i]/s"},
            "as": "hits_cfs",
        },
        {"name": "false_alarms", "threshold": {"value": 4.5, "unit": "m3/s"}, "as": "fa_m3s"},
        {
            "name": "false_alarms",
            "threshold": {"value": 4.5 / 0.028316846592, "unit": "[cft_i]/s"},
            "as": "fa_cfs",
        },
    ]
    path = _write_case(
        out,
        "threshold_units",
        "A threshold carries its unit: the same threshold in m3/s and in [cft_i]/s gives "
        "the same counts, with the secondary dataset stored in [cft_i]/s (converted at "
        "pairing).",
        _small_spec("thresholds", threshold),
        _small_inputs("[cft_i]/s", 1 / 0.028316846592),
    )
    table = _expect(path, units, periods)
    assert table["hits_m3s"].equals(table["hits_cfs"]) and table["fa_m3s"].equals(table["fa_cfs"])

    empty = _small_inputs()
    obs = _series([*OBS, *(("sim_empty", h, 2.0 + h) for h in range(3))])
    sim = _series([*SIM, *(("sim_empty", h, None) for h in range(3))])
    empty["obs"]["table"], empty["sim"]["table"] = obs, sim
    empty["xw"]["table"] = pa.table(
        {"left_key": [*KEYS, "sim_empty"], "right_key": [*KEYS, "sim_empty"]}
    )
    path = _write_case(
        out,
        "empty_series",
        "A group exists only if it has at least one complete pair: a series whose secondary "
        "values are all missing gets no row (not a row with count 0).",
        _small_spec("empty", ["count", "root_mean_square_error"]),
        empty,
    )
    table = _expect(path, units, periods)
    assert "sim_empty" not in table["primary_series_key"].to_pylist()

    _write_case(
        out,
        "reject_bare_threshold",
        "A threshold without a unit is refused.",
        _small_spec("bare", [{"name": "hits", "threshold": {"value": 4.5}}]),
        _small_inputs(),
        about="unit",
    )
    _write_case(
        out,
        "reject_bare_log",
        "One-argument log() is refused (base 10 in some engines, natural in others): use "
        "ln() or log10().",
        _small_spec("log", ["count"], filter="log(primary_value) > 0"),
        _small_inputs(),
        about="log",
    )
    for name in ("reject_bare_threshold", "reject_bare_log"):
        outcome = check_case(Case.load(out / "cases" / name), engine_evaluate, units, periods)
        assert outcome.passed, (name, outcome.problems)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    build(Path(sys.argv[1]).resolve())
