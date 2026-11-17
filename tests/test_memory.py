"""Memory limits for the engine's connections, and a process's peak memory."""

from __future__ import annotations

import sys

import duckdb
import pytest

from teehr_core.engine.duckdb_backend import DuckDBBackend
from teehr_core.engine.memory import RUN_SHARE, default_memory_limit, peak_mb, reset_peak


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_reset_starts_the_peak_again() -> None:
    assert reset_peak()
    block = b"\x01" * (200 * 2**20)  # written, so it is resident
    high = peak_mb()
    del block
    assert reset_peak()
    low = peak_mb()
    assert high is not None and low is not None
    assert high - low > 150


def test_the_engine_connection_gets_the_run_share(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEEHR_CORE_DUCKDB_MEMORY_LIMIT", raising=False)

    def limit(con: duckdb.DuckDBPyConnection) -> str:
        try:
            return str(con.execute("SELECT current_setting('memory_limit')").fetchone()[0])  # type: ignore[index]
        finally:
            con.close()

    engine = limit(DuckDBBackend._connect())
    assert engine == limit(DuckDBBackend._connect(memory_limit=default_memory_limit(RUN_SHARE)))
