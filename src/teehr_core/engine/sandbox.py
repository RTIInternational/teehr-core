"""Locking down DuckDB connections that run SQL built from specs.

Ported from VeriRace (``verirace/sandbox.py`` at 668ab7d).
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import duckdb

from teehr_core.sql import lit

__all__ = ["restrict"]


#: Where DuckDB's httpfs looks for the system's CA certificates (DuckDB 2.0's list).
CA_BUNDLES = (
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/ca-bundle.pem",
    "/etc/ssl/cert.pem",
)


def ca_bundles() -> list[str]:
    """Return the CA certificate files on this machine that HTTPS requests read."""
    candidates = [os.environ.get("SSL_CERT_FILE", ""), *CA_BUNDLES]
    return sorted({c for c in candidates if c and os.path.isfile(c)})


def restrict(con: duckdb.DuckDBPyConnection, allowed: Iterable[str] = ()) -> None:
    """Allow ``con`` no files or URLs except under the ``allowed`` directories.

    For connections that run SQL built from specs: even an expression or name that got
    past validation can't read local files (``/proc/self/environ``), other storage paths
    or other hosts, attach databases or load extensions. Attached lakes and their catalogs
    keep working, and DuckDB's own temporary files still spill. One way: DuckDB refuses to
    re-enable external access or widen the directories on a running connection, while
    tuning settings (threads, memory_limit) can still change.

    The system's CA certificates stay readable: from DuckDB 2.0, HTTPS requests (S3 on
    AWS) read them from disk, and fail without them.
    """
    dirs = sorted({d if d.endswith("/") else d + "/" for d in allowed})
    if dirs:
        con.execute(f"SET allowed_directories = [{', '.join(lit(d) for d in dirs)}]")
    certs = ca_bundles()
    if certs:
        con.execute(f"SET allowed_paths = [{', '.join(lit(c) for c in certs)}]")
    con.execute("SET enable_external_access = false")
