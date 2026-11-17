"""Local dataset catalog: named, contract-normalized datasets for the engine.

Ported from VeriRace (``verirace/engine/catalog.py`` at 668ab7d).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from teehr_core.model.contract import (
    ContractError,
    normalize_crosswalk,
    normalize_series_catalog,
    normalize_timeseries,
)
from teehr_core.model.periods import PeriodRegistry
from teehr_core.model.units import UnitRegistry

__all__ = ["Catalog", "Dataset", "DatasetRole", "Statistic"]

DatasetRole = Literal["observed", "simulated", "forecast", "auxiliary"]
Statistic = Literal["inst", "mean", "sum", "min", "max"]
_STATISTICS = ("inst", "mean", "sum", "min", "max")


@dataclass(frozen=True)
class Dataset:
    """A dataset with its contract metadata.

    For time series roles, ``table`` follows the time series contract and ``series`` is
    the optional series catalog. For ``auxiliary`` datasets, ``kind`` says what the table
    is (e.g. ``crosswalk``).
    """

    name: str
    role: DatasetRole
    table: pa.Table
    series: pa.Table | None = None
    variable: str | None = None
    unit: str | None = None
    statistic: Statistic | None = None
    #: The period (ADR-0009) each value covers unless its series names another (a
    #: ``period`` column in ``series``); ``instant`` for ``inst`` values by default.
    period: str | None = None
    kind: str | None = None

    @property
    def is_timeseries(self) -> bool:
        """Whether this dataset holds time series values."""
        return self.role != "auxiliary"

    def validated(self, units: UnitRegistry | None = None) -> Dataset:
        """Return a copy with tables normalized to the contract (raises ContractError)."""
        if not self.is_timeseries:
            if self.kind == "crosswalk":
                return replace(self, table=normalize_crosswalk(self.table, name=self.name))
            return self
        problems = []
        if self.statistic not in _STATISTICS:
            problems.append(f"statistic must be one of {_STATISTICS}, got {self.statistic!r}")
        if problems:
            raise ContractError(self.name, problems)
        role: Any = self.role
        table = normalize_timeseries(
            self.table,
            role=role,
            variable=self.variable,
            unit=self.unit,
            name=self.name,
            units=units,
        )
        series = (
            None
            if self.series is None
            else normalize_series_catalog(self.series, name=f"{self.name} series catalog")
        )
        return replace(self, table=table, series=series)


class Catalog:
    """A set of named datasets plus the unit, period and variable registries.

    Datasets are validated when added. Without a unit registry, pairing works only
    between datasets in the same unit; without a period registry, only ``instant`` is
    known. ``variables`` maps a variable to its canonical unit, the unit an input
    filter's numbers are read in (a variable not in it: the dataset's own unit).
    """

    def __init__(
        self,
        datasets: Iterable[Dataset] = (),
        units: UnitRegistry | None = None,
        periods: PeriodRegistry | None = None,
        variables: Mapping[str, str] | None = None,
    ) -> None:
        self.units = units
        self.periods = periods if periods is not None else PeriodRegistry.default()
        self.variables: Mapping[str, str] = variables or {}
        self._datasets: dict[str, Dataset] = {}
        for ds in datasets:
            self.add(ds)

    def add(self, dataset: Dataset) -> None:
        """Validate and add a dataset, replacing any dataset with the same name."""
        self._datasets[dataset.name] = dataset.validated(self.units)

    def __getitem__(self, name: str) -> Dataset:
        try:
            return self._datasets[name]
        except KeyError:
            raise KeyError(
                f"dataset {name!r} not in catalog; available: {', '.join(self._datasets)}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._datasets

    def __iter__(self) -> Iterator[Dataset]:
        return iter(self._datasets.values())

    def names(self) -> list[str]:
        """Dataset names."""
        return list(self._datasets)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        units: UnitRegistry | None = None,
        periods: PeriodRegistry | None = None,
    ) -> Catalog:
        """Load a catalog file (``datasets.yaml``) whose file paths are relative to it.

        Each entry under ``datasets`` gives ``role``, ``file`` and, as applicable,
        ``series_file``, ``variable``, ``unit``, ``statistic``, ``period`` and
        ``kind``. An optional ``variables`` maps a variable to its canonical unit. Other
        keys are ignored.
        """
        path = Path(path)
        meta = yaml.safe_load(path.read_text())
        base = path.parent
        datasets = []
        for name, entry in meta["datasets"].items():
            series_file = entry.get("series_file")
            datasets.append(
                Dataset(
                    name=name,
                    role=entry["role"],
                    table=pq.read_table(base / entry["file"]),
                    series=pq.read_table(base / series_file) if series_file else None,
                    variable=entry.get("variable"),
                    unit=entry.get("unit"),
                    statistic=entry.get("statistic"),
                    period=entry.get("period"),
                    kind=entry.get("kind"),
                )
            )
        return cls(datasets, units, periods, meta.get("variables"))
