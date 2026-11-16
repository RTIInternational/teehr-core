import subprocess
import sys

import teehr_core


def test_version() -> None:
    assert teehr_core.__version__


def test_import_is_light() -> None:
    """The core imports without Spark, DuckDB or pandas."""
    code = (
        "import sys, teehr_core; "
        "heavy = {'pyspark', 'duckdb', 'pandas'} & set(sys.modules); "
        "print(','.join(sorted(heavy)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
