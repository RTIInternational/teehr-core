"""Connections that run SQL built from specs can't reach files, URLs or other databases."""

from pathlib import Path

import duckdb
import pytest

from teehr_core.engine.connect import memory_connection
from teehr_core.engine.duckdb_backend import DuckDBBackend
from teehr_core.engine.sandbox import ca_bundles, restrict

ESCAPES = [
    "SELECT * FROM read_text('/etc/hostname')",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_text('/proc/self/environ')",
    "SELECT * FROM read_text('http://example.com')",
    "COPY (SELECT 1) TO '/tmp/verirace-sandbox-escape.csv'",
    "ATTACH '/tmp/verirace-sandbox-escape.db' AS other",
    "LOAD spatial",
]


@pytest.mark.parametrize("sql", ESCAPES)
def test_restricted_connection_reaches_nothing(sql: str) -> None:
    con = duckdb.connect()
    restrict(con)
    with pytest.raises(duckdb.Error, match="disabled"):
        con.execute(sql)


def test_restriction_is_one_way(tmp_path: Path) -> None:
    con = duckdb.connect()
    restrict(con, [str(tmp_path)])
    for sql in [
        "SET enable_external_access = true",
        "RESET enable_external_access",
        "SET allowed_directories = ['/']",
        "SET allowed_paths = ['/etc/hostname']",
    ]:
        with pytest.raises(duckdb.Error):
            con.execute(sql)
    con.execute("SET threads = 2; SET memory_limit = '1GB'")  # tuning still works


@pytest.mark.skipif(not ca_bundles(), reason="no CA certificates on this machine")
def test_https_can_still_read_the_ca_certificates() -> None:
    """DuckDB 2.0 reads them for every HTTPS request; nothing else becomes readable."""
    con = duckdb.connect()
    restrict(con)
    bundle = ca_bundles()[0]
    assert con.execute(
        f"SELECT octet_length(content) > 0 FROM read_blob('{bundle}')"
    ).fetchall() == [(True,)]
    with pytest.raises(duckdb.Error, match="disabled"):
        con.execute("SELECT * FROM read_text('/etc/hostname')")


def test_allowed_directories_still_work(tmp_path: Path) -> None:
    con = duckdb.connect()
    restrict(con, [str(tmp_path)])
    out = tmp_path / "x.parquet"
    con.execute(f"COPY (SELECT 42 AS v) TO '{out}'")
    assert con.execute(f"SELECT v FROM read_parquet('{out}')").fetchall() == [(42,)]
    with pytest.raises(duckdb.Error, match="disabled"):
        con.execute(f"SELECT * FROM read_text('{tmp_path.parent}/elsewhere')")


def test_engine_connections_are_restricted() -> None:
    """The engine pairs and aggregates registered tables; its SQL reaches no files."""
    con = DuckDBBackend._connect()
    with pytest.raises(duckdb.Error, match="disabled"):
        con.execute("SELECT * FROM read_text('/etc/hostname')")


def test_connections_spill_to_their_own_directories() -> None:
    """Parallel workers mustn't share spill files (DuckDB names them alike everywhere)."""
    a, b = memory_connection(), memory_connection()
    spill = [
        Path(c.execute("SELECT current_setting('temp_directory')").fetchone()[0]) for c in (a, b)
    ]
    assert spill[0] != spill[1]
    restrict(a)  # spilling still works in a sandboxed connection
    a.execute("SET memory_limit = '50MB'; SET threads = 1")
    a.execute("CREATE TEMP TABLE t AS SELECT md5(i::VARCHAR) AS x FROM range(4000000) t(i)")
    assert any(spill[0].iterdir())
    assert not any(spill[1].iterdir())
    a.close()
    del a
    assert not spill[0].exists()
