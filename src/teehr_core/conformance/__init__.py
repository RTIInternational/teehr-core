"""Conformance cases (teehr-shared-eval-spec ``conformance/``): load, run, compare.

A case is a directory::

    case.yaml        description, added (date), and optionally tolerance, keys
    spec.yaml        the evaluation spec (a rejection case: invalid-spec.yaml)
    inputs/          datasets.yaml (the engine's catalog format) and its Parquet files
    expected/        results.parquet, or error.yaml for a rejection case

A system runs every case with its own ``evaluate`` (spec, inputs directory, units,
periods → the final stage's table; raising for a refused spec) and compares with
:func:`check_case`; :func:`run_cases` does that for a directory of cases, by default
with teehr-core's engine (the ``engine`` extra).

Comparison: rows are matched on ``keys`` (default: the first stage's ``group_by``, plus
``group_by`` of later stages), columns must be the same; nulls (or NaN) must match;
integers and strings exactly; floats within the relative ``tolerance`` (default 1e-12).
A rejection case passes when the spec is refused and the message mentions ``about``.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from teehr_core.model.periods import PeriodRegistry
from teehr_core.model.units import UnitRegistry

__all__ = [
    "Case",
    "Evaluate",
    "Outcome",
    "check_case",
    "compare_tables",
    "engine_evaluate",
    "load_periods",
    "load_units",
    "run_cases",
]

#: How a system runs a case: (spec, inputs directory, units, periods) → the result table.
Evaluate = Callable[[dict[str, Any], Path, UnitRegistry, PeriodRegistry], pa.Table]

DEFAULT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Case:
    """One conformance case, loaded from its directory."""

    name: str
    path: Path
    description: str
    spec: dict[str, Any]
    tolerance: float = DEFAULT_TOLERANCE
    keys: tuple[str, ...] = ()
    #: For a rejection case: what the refusal must mention (case-insensitive).
    about: str | None = None
    expected: pa.Table | None = None

    @property
    def rejects(self) -> bool:
        """Whether the spec must be refused."""
        return self.expected is None

    @property
    def inputs(self) -> Path:
        """The case's inputs directory (``datasets.yaml`` and its files)."""
        return self.path / "inputs"

    @classmethod
    def load(cls, path: str | Path) -> Case:
        """Load the case in directory ``path``."""
        path = Path(path)
        meta = yaml.safe_load((path / "case.yaml").read_text())
        invalid = path / "invalid-spec.yaml"
        spec = yaml.safe_load((invalid if invalid.exists() else path / "spec.yaml").read_text())
        expected, about = None, None
        if invalid.exists():
            about = yaml.safe_load((path / "expected" / "error.yaml").read_text())["about"]
        else:
            expected = pq.read_table(path / "expected" / "results.parquet")
        keys = meta.get("keys")
        if keys is None and not invalid.exists():
            keys = [k for stage in spec.get("aggregate", []) for k in stage.get("group_by", [])]
        return cls(
            name=path.name,
            path=path,
            description=str(meta.get("description", "")).strip(),
            spec=spec,
            tolerance=float(meta.get("tolerance", DEFAULT_TOLERANCE)),
            keys=tuple(dict.fromkeys(keys or ())),
            about=about,
            expected=expected,
        )


@dataclass
class Outcome:
    """A case's result: passed, or the problems found."""

    case: str
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether the case passed."""
        return not self.problems


def _undefined(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _same(a: Any, b: Any, tolerance: float) -> bool:
    if _undefined(a) or _undefined(b):
        return _undefined(a) and _undefined(b)
    if isinstance(a, float) or isinstance(b, float):
        if isinstance(a, bool) or isinstance(b, bool):
            return bool(a == b)
        return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)
    return bool(a == b)


def _row_key(row: dict[str, Any], keys: Iterable[str]) -> tuple[str, ...]:
    return tuple(repr(row[k]) for k in keys)


def compare_tables(
    expected: pa.Table, actual: pa.Table, keys: Iterable[str], tolerance: float
) -> list[str]:
    """Return how ``actual`` differs from ``expected`` (empty when it matches)."""
    keys = list(keys)
    problems = []
    if set(expected.column_names) != set(actual.column_names):
        missing = sorted(set(expected.column_names) - set(actual.column_names))
        extra = sorted(set(actual.column_names) - set(expected.column_names))
        return [f"columns differ: missing {missing}, extra {extra}"]
    if expected.num_rows != actual.num_rows:
        return [f"{actual.num_rows} rows, expected {expected.num_rows}"]
    want = {_row_key(r, keys): r for r in expected.to_pylist()}
    got = {_row_key(r, keys): r for r in actual.to_pylist()}
    if len(want) != expected.num_rows or len(got) != actual.num_rows:
        return [f"rows aren't unique on the keys {keys}"]
    for key, row in want.items():
        other = got.get(key)
        if other is None:
            problems.append(f"no row for { ({k: row[k] for k in keys}) }")
            continue
        for column, value in row.items():
            if not _same(value, other[column], tolerance):
                where = {k: row[k] for k in keys}
                problems.append(f"{column} at {where}: {other[column]!r}, expected {value!r}")
    return problems


def check_case(
    case: Case, evaluate: Evaluate, units: UnitRegistry, periods: PeriodRegistry
) -> Outcome:
    """Run ``case`` with ``evaluate`` and compare with what it expects."""
    outcome = Outcome(case.name)
    try:
        actual = evaluate(case.spec, case.inputs, units, periods)
    except Exception as exc:
        if not case.rejects:
            outcome.problems.append(f"failed: {type(exc).__name__}: {exc}")
        elif case.about and case.about.lower() not in str(exc).lower():
            outcome.problems.append(
                f"refused, but the reason doesn't mention {case.about!r}: {exc}"
            )
        return outcome
    if case.rejects:
        outcome.problems.append("the spec was accepted; it must be refused")
        return outcome
    assert case.expected is not None
    outcome.problems = compare_tables(case.expected, actual, case.keys, case.tolerance)
    return outcome


def engine_evaluate(
    spec: dict[str, Any], inputs: Path, units: UnitRegistry, periods: PeriodRegistry
) -> pa.Table:
    """Run a case with teehr-core's engine (the ``engine`` extra)."""
    from teehr_core import engine

    return engine.evaluate(spec, inputs / "datasets.yaml", units, periods).table


def _csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [{k: v or None for k, v in r.items()} for r in csv.DictReader(f)]


def load_units(path: str | Path) -> UnitRegistry:
    """Load a unit table CSV (``aliases`` separated by ``|``)."""
    rows = _csv_rows(Path(path))
    for r in rows:
        r["aliases"] = r["aliases"].split("|") if r.get("aliases") else []
    return UnitRegistry.from_rows(rows)


def load_periods(path: str | Path) -> PeriodRegistry:
    """Load a period table CSV (the spec's ``standards/periods.csv``)."""
    return PeriodRegistry.from_rows(_csv_rows(Path(path)))


def run_cases(
    cases: str | Path,
    evaluate: Evaluate = engine_evaluate,
    units: UnitRegistry | None = None,
    periods: PeriodRegistry | None = None,
) -> list[Outcome]:
    """Run every case under ``cases`` (the spec repo's ``conformance/cases``).

    Units and periods default to the spec repo's: ``conformance/units.csv`` and
    ``standards/periods.csv``, found from ``cases``.
    """
    cases = Path(cases)
    paths = sorted(p for p in cases.iterdir() if (p / "case.yaml").exists())
    if not paths:
        return []
    units = units or load_units(cases.parent / "units.csv")
    periods = periods or load_periods(cases.parent.parent / "standards" / "periods.csv")
    return [check_case(Case.load(path), evaluate, units, periods) for path in paths]
