"""The conformance runner, and the cases scripts/conformance/build_cases.py builds."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pyarrow as pa
import pytest

from teehr_core.conformance import compare_tables, run_cases

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cases(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The shared cases, built as in the spec repo (with its periods table)."""
    spec_repo = tmp_path_factory.mktemp("spec")
    (spec_repo / "standards").mkdir()
    shutil.copy(ROOT / "tests" / "fixtures" / "periods.csv", spec_repo / "standards")
    (spec_repo / "conformance").mkdir()
    spec = importlib.util.spec_from_file_location(
        "build_cases", ROOT / "scripts" / "conformance" / "build_cases.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build(spec_repo / "conformance")
    return spec_repo / "conformance" / "cases"


def test_the_engine_passes_every_case(cases: Path) -> None:
    outcomes = run_cases(cases)
    assert len(outcomes) >= 6
    assert [o.problems for o in outcomes if not o.passed] == []


def test_a_wrong_result_fails_its_case(cases: Path) -> None:
    def off_by_a_bit(spec, inputs, units, periods):  # type: ignore[no-untyped-def]
        from teehr_core.conformance import engine_evaluate

        table = engine_evaluate(spec, inputs, units, periods)
        if "root_mean_square_error" in table.column_names:
            i = table.schema.get_field_index("root_mean_square_error")
            col = pa.compute.multiply(table["root_mean_square_error"], 1 + 1e-9)
            table = table.set_column(i, "root_mean_square_error", col)
        return table

    failed = {o.case for o in run_cases(cases, off_by_a_bit) if not o.passed}
    assert failed == {"hydro_metrics", "energy_metrics", "undefined_results"}


def test_accepting_a_refused_spec_fails(cases: Path) -> None:
    def accepts_everything(spec, inputs, units, periods):  # type: ignore[no-untyped-def]
        return pa.table({"x": [1]})

    outcomes = {o.case: o for o in run_cases(cases, accepts_everything)}
    assert outcomes["reject_bare_log"].problems == ["the spec was accepted; it must be refused"]


def test_compare_tables() -> None:
    a = pa.table({"k": ["x", "y"], "v": [1.0, None], "n": [2, 3]})
    assert compare_tables(a, a.take([1, 0]), ["k"], 1e-12) == []  # row order doesn't matter
    nan = pa.table({"k": ["x", "y"], "v": [1.0, float("nan")], "n": [2, 3]})
    assert compare_tables(a, nan, ["k"], 1e-12) == []  # NaN is null
    off = pa.table({"k": ["x", "y"], "v": [1.0 + 1e-6, None], "n": [2, 3]})
    assert compare_tables(a, off, ["k"], 1e-12) == ["v at {'k': 'x'}: 1.000001, expected 1.0"]
    assert compare_tables(a, off, ["k"], 1e-5) == []
    assert "columns differ" in compare_tables(a, a.drop_columns(["n"]), ["k"], 1e-12)[0]
