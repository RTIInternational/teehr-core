"""DuckDB connections for the engine: Arrow in and out, spilling, object storage.

Ported from VeriRace (``verirace/warehouse/connect.py`` and ``config.py`` at 668ab7d):
the parts the engine uses, without VeriRace's lakes.
"""

from __future__ import annotations

import shutil
import tempfile
import weakref
from dataclasses import dataclass

import duckdb
import pyarrow as pa

from teehr_core.sql import lit

__all__ = [
    "S3Config",
    "arrow_reader",
    "arrow_table",
    "memory_connection",
    "register",
    "storage_access",
]


@dataclass(frozen=True)
class S3Config:
    """Object storage settings for a DuckDB S3 secret."""

    endpoint: str | None = None  # host[:port]; None means AWS S3
    key_id: str | None = None
    secret: str | None = None
    session_token: str | None = None
    region: str = "us-east-1"
    url_style: str = "path"
    use_ssl: bool = False


def arrow_table(con: duckdb.DuckDBPyConnection, sql: str, params: object = None) -> pa.Table:
    """Run ``sql`` on ``con`` and return its result as an Arrow table.

    Through a relation (``con.sql``): from DuckDB 2.0, Arrow from an ``execute`` result
    makes an extra copy of the rows (ten times slower for 3M rows) and is typed optional.
    """
    con.complete()  # see register
    table: pa.Table = con.sql(sql, params=params).to_arrow_table()
    return table


def arrow_reader(
    con: duckdb.DuckDBPyConnection, sql: str, batch_rows: int, params: object = None
) -> pa.RecordBatchReader:
    """Run ``sql`` on ``con`` and return its result in batches of ``batch_rows`` (as above)."""
    con.complete()  # see register
    reader: pa.RecordBatchReader = con.sql(sql, params=params).to_arrow_reader(batch_rows)
    return reader


def register(con: duckdb.DuckDBPyConnection, name: str, table: object) -> None:
    """Register ``table`` (Arrow, or anything DuckDB scans) on ``con`` as view ``name``.

    From DuckDB 2.0 a connection refuses ``register`` and ``sql`` while a query's result
    is unfinished (e.g. after ``fetchone``); a new ``execute`` still discards it, as
    before. So finish any such result first (a no-op when there is none).
    """
    con.complete()
    con.register(name, table)


def memory_connection() -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB connection that spills to a directory of its own.

    DuckDB's default spill directory, ``./.tmp``, uses the same file names in every
    process, so parallel workers that spill at once overwrite each other's files. The
    directory goes when the connection does.

    DuckDB 2.0's ``type_pushdown`` optimizer is off: it pushes ``TRY_CAST`` into a CSV
    scan, which then fails on a value the cast would have made NULL (an unreadable time in
    an upload; still so in 2.0.1 dev builds, 2026-11-02).
    """
    spill = tempfile.mkdtemp(prefix="teehr-core-spill-")
    con = duckdb.connect(config={"temp_directory": spill, "disabled_optimizers": "type_pushdown"})
    weakref.finalize(con, shutil.rmtree, spill, True)
    return con


def _s3_secret(con: duckdb.DuckDBPyConnection, s3: S3Config) -> None:
    parts = ["TYPE s3", f"REGION {lit(s3.region)}", f"URL_STYLE {lit(s3.url_style)}"]
    parts.append(f"USE_SSL {'true' if s3.use_ssl else 'false'}")
    if s3.endpoint:
        parts.append(f"ENDPOINT {lit(s3.endpoint)}")
    if s3.key_id:
        parts.append(f"KEY_ID {lit(s3.key_id)}")
    if s3.secret:
        parts.append(f"SECRET {lit(s3.secret)}")
    if s3.session_token:
        parts.append(f"SESSION_TOKEN {lit(s3.session_token)}")
    if not s3.key_id:
        # No explicit keys: the AWS credential chain (instance or pod role, environment,
        # ~/.aws), as on EC2/EKS.
        con.install_extension("aws")
        con.load_extension("aws")
        parts.insert(1, "PROVIDER credential_chain")
    con.execute(f"CREATE OR REPLACE SECRET teehr_core_s3 ({', '.join(parts)})")


def storage_access(con: duckdb.DuckDBPyConnection, s3: S3Config) -> None:
    """Let ``con`` read and write object storage with ``s3`` (before it's restricted)."""
    con.install_extension("httpfs")  # no-op if installed (the platform image bakes it in)
    con.load_extension("httpfs")
    _s3_secret(con, s3)
