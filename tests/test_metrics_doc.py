"""The metrics reference (docs/metrics.md) matches the registry."""

import importlib.util
from pathlib import Path

from teehr_core.metrics.catalog import FAMILIES, PARAMETERS, catalog
from teehr_core.metrics.registry import METRICS

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("metrics_doc", ROOT / "scripts/metrics_doc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_page_is_up_to_date():
    page = (ROOT / "docs/metrics.md").read_text()
    assert page == _script().render(), "run: uv run python scripts/metrics_doc.py"


def test_every_metric_parameter_and_family_is_described():
    entries = catalog()
    assert {e["name"] for e in entries} == set(METRICS)
    assert {p for e in entries for p in e["parameters"]} <= set(PARAMETERS)
    assert {e["family"] for e in entries} <= set(FAMILIES)
