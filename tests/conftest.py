import csv
from pathlib import Path

import pytest

from teehr_core.model.periods import PeriodRegistry
from teehr_core.model.units import UnitRegistry

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    return Path(__file__).parent / "golden"


def _rows(name: str) -> list[dict[str, str | None]]:
    with (FIXTURES / name).open() as f:
        return [{k: v or None for k, v in r.items()} for r in csv.DictReader(f)]


@pytest.fixture(scope="session")
def units() -> UnitRegistry:
    """The units VeriRace's public lake ships (tests/fixtures/units.csv)."""
    rows = _rows("units.csv")
    for r in rows:
        r["aliases"] = (r["aliases"] or "").split("|") if r["aliases"] else []  # type: ignore[assignment]
    return UnitRegistry.from_rows(rows)


@pytest.fixture(scope="session")
def periods() -> PeriodRegistry:
    """The shared periods (teehr-shared-eval-spec standards/periods.csv)."""
    return PeriodRegistry.from_rows(_rows("periods.csv"))
