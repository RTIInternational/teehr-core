"""Units: conversion over a unit table each system supplies (shared spec ``units.md``).

Ported from VeriRace (``verirace/units.py`` at 668ab7d).

Units are UCUM symbols (``m3/s``, ``[cft_i]/s``, ``Cel``). Each belongs to a physical
dimension and converts to that dimension's SI unit with ``si = value * to_si + offset``;
the offset makes temperature scales work. Conversion is only allowed within a dimension.
No units are built in: a :class:`UnitRegistry` is built from a system's unit table
(VeriRace's ``verirace_units`` lake table, the control plane's ``units`` domain), with
:meth:`UnitRegistry.from_rows`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

__all__ = [
    "IDENTITY",
    "Conversion",
    "Unit",
    "UnitError",
    "UnitRegistry",
    "conversion_between",
    "merge",
]

#: What :meth:`Conversion.apply` takes and returns.
Values = Any  # float | np.ndarray | pa.Array | pa.ChunkedArray


class UnitError(ValueError):
    """Raised for unknown units or conversions between different dimensions."""


@dataclass(frozen=True)
class Unit:
    """A unit: its dimension, and how to convert it to the dimension's SI unit."""

    symbol: str
    dimension: str
    to_si: float
    offset: float = 0.0
    description: str = ""
    aliases: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class Conversion:
    """``converted = value * scale + offset``."""

    scale: float
    offset: float

    @property
    def is_identity(self) -> bool:
        """Whether the conversion leaves values unchanged."""
        return self.scale == 1.0 and self.offset == 0.0

    @property
    def of_differences(self) -> Conversion:
        """Return the conversion of a difference or magnitude: the scale alone.

        A 5 K ramp is a 5 Cel ramp (``units.md``), though 300 K is 26.85 Cel.
        """
        return Conversion(self.scale, 0.0)

    def apply(self, values: Values) -> Values:
        """Return ``values`` (a number, numpy or pyarrow array) converted."""
        if self.is_identity:
            return values
        if isinstance(values, pa.Array | pa.ChunkedArray):
            as_float = pc.cast(values, pa.float64())
            converted: Values = pc.add(pc.multiply(as_float, self.scale), self.offset)
            return converted
        return values * self.scale + self.offset


IDENTITY = Conversion(1.0, 0.0)


class UnitRegistry:
    """Known units, by symbol and alias."""

    def __init__(self, units: Iterable[Unit]) -> None:
        self._units: dict[str, Unit] = {}
        self._aliases: dict[str, str] = {}
        for unit in units:
            if unit.symbol in self._units:
                raise UnitError(f"unit {unit.symbol!r} is defined twice")
            self._units[unit.symbol] = unit
            for alias in unit.aliases:
                self._aliases[alias] = unit.symbol

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> UnitRegistry:
        """Build from a unit table's rows.

        Each row has ``symbol``, ``dimension``, ``to_si``, and optionally ``offset``,
        ``description``, ``aliases``.
        """
        return cls(
            Unit(
                symbol=r["symbol"],
                dimension=r["dimension"],
                to_si=float(r["to_si"]),
                offset=float(r.get("offset") or 0.0),
                description=r.get("description") or "",
                aliases=tuple(r.get("aliases") or ()),
            )
            for r in rows
        )

    def __contains__(self, symbol: object) -> bool:
        return isinstance(symbol, str) and self.canonical(symbol) in self._units

    def __iter__(self) -> Iterator[Unit]:
        return iter(self._units.values())

    def __len__(self) -> int:
        return len(self._units)

    def canonical(self, symbol: str) -> str:
        """Return the canonical symbol for ``symbol`` (resolving aliases)."""
        return self._aliases.get(symbol, symbol)

    def get(self, symbol: str) -> Unit:
        """Return the unit for ``symbol`` (canonical or alias).

        Raises
        ------
        UnitError
            If the unit is unknown.
        """
        unit = self._units.get(self.canonical(symbol))
        if unit is None:
            raise UnitError(f"unknown unit {symbol!r}; known units: {', '.join(self.symbols())}")
        return unit

    def symbols(self) -> list[str]:
        """Return the canonical unit symbols, sorted."""
        return sorted(self._units)

    def conversion(self, from_symbol: str, to_symbol: str) -> Conversion:
        """Return how to convert values in ``from_symbol`` to ``to_symbol``.

        Raises
        ------
        UnitError
            If either unit is unknown or they measure different dimensions.
        """
        a, b = self.get(from_symbol), self.get(to_symbol)
        if a.dimension != b.dimension:
            raise UnitError(
                f"cannot convert {from_symbol!r} ({a.dimension}) to {to_symbol!r} ({b.dimension})"
            )
        if a.symbol == b.symbol:
            return IDENTITY
        return Conversion(a.to_si / b.to_si, (a.offset - b.offset) / b.to_si)


def conversion_between(
    registry: UnitRegistry | None, from_symbol: str, to_symbol: str
) -> Conversion:
    """Return the conversion between two units; identical symbols need no registry."""
    if from_symbol == to_symbol:
        return IDENTITY
    if registry is None:
        raise UnitError(f"no unit registry to convert {from_symbol!r} to {to_symbol!r}")
    return registry.conversion(from_symbol, to_symbol)


def merge(registries: Sequence[UnitRegistry]) -> UnitRegistry:
    """Combine registries (e.g. public and project units); a symbol may appear only once."""
    return UnitRegistry(u for r in registries for u in r)
