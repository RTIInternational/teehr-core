"""Quoting names and strings into SQL text, for what can't be a bound parameter.

Ported from VeriRace (``verirace/sql.py`` at 668ab7d).
"""

from __future__ import annotations

__all__ = ["lit", "q"]


def q(name: str) -> str:
    """Quote a SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def lit(value: str) -> str:
    """Quote a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"
