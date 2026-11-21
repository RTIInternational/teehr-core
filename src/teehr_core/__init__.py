"""teehr-core: TEEHR's data model, metric kernels, metric state, bootstrap and spec engine.

Imports only numpy, pyarrow, pydantic and sqlglot; the spec engine (``teehr_core.engine``) needs
the ``engine`` extra (DuckDB), and the warehouse rules on Iceberg tables
(``teehr_core.warehouse``) the ``iceberg`` extra (PyIceberg).
"""

__version__ = "0.0.1"
