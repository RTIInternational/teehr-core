"""Local compute engine: pairing, fields, grouping and metrics.

Ported from VeriRace (``verirace/engine/__init__.py`` at 668ab7d).

Examples
--------
>>> from teehr_core import engine
>>> result = engine.evaluate("spec.yaml", "datasets.yaml")  # doctest: +SKIP
>>> result.table.to_pandas()  # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from teehr_core.engine.catalog import Catalog, Dataset
from teehr_core.engine.duckdb_backend import DuckDBBackend, EvaluationResult
from teehr_core.engine.plan import Plan, PlanError, build_plan
from teehr_core.model.contract import ContractError
from teehr_core.model.periods import PeriodRegistry
from teehr_core.model.units import UnitRegistry
from teehr_core.spec import EvaluationSpec, load_spec

__all__ = [
    "Catalog",
    "ContractError",
    "Dataset",
    "DuckDBBackend",
    "EvaluationResult",
    "Plan",
    "PlanError",
    "build_plan",
    "evaluate",
]


def evaluate(
    spec: EvaluationSpec | Mapping[str, Any] | str | Path,
    catalog: Catalog | str | Path,
    units: UnitRegistry | None = None,
    periods: PeriodRegistry | None = None,
) -> EvaluationResult:
    """Run an evaluation in memory (tests and the run executor's shards).

    Parameters
    ----------
    spec
        An :class:`~teehr_core.spec.EvaluationSpec`, a mapping, a YAML file path, or YAML text.
    catalog
        A :class:`Catalog` or the path to a catalog file (``datasets.yaml``).
    units
        Unit registry for a catalog file (a :class:`Catalog` carries its own).
    periods
        Period registry for a catalog file (likewise).

    Returns
    -------
    EvaluationResult
        ``table`` holds the final stage; ``stages`` every stage; ``pairs`` the paired rows
        when ``output.include_pairs`` is set.
    """
    parsed = load_spec(spec)
    cat = catalog if isinstance(catalog, Catalog) else Catalog.from_yaml(catalog, units, periods)
    return DuckDBBackend().execute(build_plan(parsed, cat))
