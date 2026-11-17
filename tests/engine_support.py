"""Fixture paths and helpers for the engine tests (from VeriRace's ``tests/unit/test_runs.py``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import yaml

TESTS = Path(__file__).parent
HYDRO = TESTS / "fixtures" / "hydro"
ENERGY = TESTS / "fixtures" / "energy"
SPECS = TESTS / "specs"


def public_spec(path: Path) -> dict[str, Any]:
    """A fixture spec with hydro dataset references pointed at the public project."""
    spec: dict[str, Any] = yaml.safe_load(path.read_text())
    spec["inputs"]["primary"]["dataset"] = "public." + spec["inputs"]["primary"]["dataset"]
    for p in spec["inputs"]["secondary"]:
        p["dataset"] = "public." + p["dataset"]
    spec["pair"]["relation"] = "public." + spec["pair"]["relation"]
    return spec


def _sorted_rows(t: pa.Table) -> list[dict[str, Any]]:
    return sorted(t.to_pylist(), key=lambda r: tuple(str(v) for v in r.values()))


def _assert_same(a: pa.Table, b: pa.Table) -> None:
    ra, rb = _sorted_rows(a), _sorted_rows(b)
    assert len(ra) == len(rb)
    for x, y in zip(ra, rb, strict=True):
        for k in x:
            if isinstance(x[k], float):
                assert x[k] == pytest.approx(y[k], rel=1e-12)
            else:
                assert x[k] == y[k]
