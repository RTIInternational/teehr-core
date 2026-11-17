"""teehr-core: TEEHR's data model, metric kernels, metric state, bootstrap and spec engine.

Imports only numpy, pyarrow, pydantic, pyyaml and sqlglot; the spec engine
(``teehr_core.engine``) needs the ``engine`` extra (DuckDB).
"""

from importlib.metadata import version

__version__ = version("teehr-core")
