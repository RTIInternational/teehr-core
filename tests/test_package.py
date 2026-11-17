import subprocess
import sys

import teehr_core


def test_version() -> None:
    assert teehr_core.__version__


def test_import_is_light() -> None:
    """The core imports without Spark, DuckDB or pandas (only the engine needs DuckDB)."""
    code = (
        "import sys, teehr_core, teehr_core.metrics, teehr_core.metrics.state, "
        "teehr_core.spec, teehr_core.model.units, teehr_core.model.contract, "
        "teehr_core.model.periods, teehr_core.model.dataset_types; "
        "heavy = {'pyspark', 'duckdb', 'pandas'} & set(sys.modules); "
        "print(','.join(sorted(heavy)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
