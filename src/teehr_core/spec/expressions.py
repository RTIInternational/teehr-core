"""Restricted SQL expressions for ``fields[].expr``, ``filter`` and ``conditions`` levels.

Ported from VeriRace (``verirace/spec/expressions.py`` at 668ab7d).

Expressions are written in an engine-neutral subset of standard SQL, the shared expression
language, and stored as written. They are parsed with sqlglot's neutral dialect and checked
against an allow-list chosen to mean the same in DuckDB, Spark and Trino: column references,
literals, operators, ``CASE``, ``CAST`` to a few types, ``IN``/``BETWEEN``/``LIKE``,
``INTERVAL`` and a fixed set of math, time and string functions. Subqueries, table
functions, unknown functions and multiple statements are refused.

The engine runs :func:`to_duckdb`'s transpilation of the checked tree, never the text
itself.
"""

from __future__ import annotations

from typing import overload

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

__all__ = [
    "ALLOWED_FUNCTIONS",
    "ExpressionError",
    "check_expression",
    "expression_columns",
    "to_duckdb",
]


class ExpressionError(ValueError):
    """An expression uses syntax or functions outside the allowed subset."""


_STRUCTURE = {
    "column", "identifier", "literal", "boolean", "null", "paren", "neg", "not", "and", "or",
    "add", "sub", "mul", "div", "mod", "eq", "neq", "gt", "gte", "lt", "lte", "is", "in",
    "between", "case", "if", "cast", "datatype", "var", "interval", "like", "dpipe",
}  # fmt: skip

# sqlglot expression keys of the allowed functions.
_FUNCTIONS = {
    "abs", "coalesce", "least", "greatest", "nullif", "ceil", "floor",
    "ln", "log", "exp", "sqrt", "pow",
    "year", "quarter", "month", "day", "hour", "minute", "datetrunc", "extract",
    "lower", "upper", "length", "substring", "concat",
}  # fmt: skip

#: Function names as written (``log10`` and ``log(b, x)`` parse to ``log``, ``power`` to
#: ``pow``).
ALLOWED_FUNCTIONS = sorted(
    (_FUNCTIONS - {"datetrunc", "pow", "log"}) | {"date_trunc", "log10", "power", "pow"}
)

#: Types a ``CAST`` may name.
CAST_TYPES = frozenset(
    {"DOUBLE", "BIGINT", "INT", "VARCHAR", "TEXT", "BOOLEAN", "DATE", "TIMESTAMP", "TIMESTAMPTZ"}
)
#: Fields ``EXTRACT`` may take, and units an ``INTERVAL`` may have.
EXTRACT_FIELDS = frozenset({"YEAR", "QUARTER", "MONTH", "DAY", "HOUR", "MINUTE"})
INTERVAL_UNITS = frozenset({"DAY", "HOUR", "MINUTE", "SECOND"})

#: Why some functions are refused, for the error message.
_INSTEAD = {
    "log": "one-argument log differs by engine (base 10 or natural); use ln, log10 or log(base, x)",
    "round": "round's ties differ by engine; use floor or ceil",
    "dayofweek": "engines number weekdays differently",
    "week": "engines number weeks differently",
    "datediff": "date_diff's arguments differ by engine; subtract the times instead",
    "isnan": "engines name and treat NaN differently",
    "ilike": "not in every engine; use lower(x) LIKE ...",
}


def _refuse(text: str, what: str, why: str | None = None) -> ExpressionError:
    reason = f": {why}" if why else ""
    return ExpressionError(
        f"{what!r} is not allowed in expressions ({text!r}){reason}. Allowed functions: "
        + ", ".join(ALLOWED_FUNCTIONS)
    )


def _parse(text: str) -> exp.Expr:
    try:
        return sqlglot.parse_one(text)
    except ParseError as exc:
        raise ExpressionError(f"cannot parse expression {text!r}: {exc}") from exc


def check_expression(text: str) -> str:
    """Validate ``text`` and return it as written.

    Raises
    ------
    ExpressionError
        If the expression can't be parsed or uses anything outside the allowed subset.
    """
    tree = _parse(text)
    for node in tree.walk():
        key = node.key
        if isinstance(node, exp.Log) and node.args.get("expression") is None:
            raise _refuse(text, "log", _INSTEAD["log"])
        if isinstance(node, exp.DataType):
            if node.this.name not in CAST_TYPES:
                raise _refuse(text, f"CAST to {node.sql()}", "use " + ", ".join(sorted(CAST_TYPES)))
            continue
        if isinstance(node, exp.Var):
            allowed = EXTRACT_FIELDS if isinstance(node.parent, exp.Extract) else INTERVAL_UNITS
            if not isinstance(node.parent, exp.Extract | exp.Interval):
                raise _refuse(text, node.name)
            if node.name.upper() not in allowed:
                raise _refuse(text, node.name, "use " + ", ".join(sorted(allowed)))
            continue
        if key in _STRUCTURE or key in _FUNCTIONS:
            continue
        what = node.name if isinstance(node, exp.Anonymous) else key
        raise _refuse(text, what, _INSTEAD.get(key))
    return text


@overload
def to_duckdb(text: str, *, scale: float = 1.0, offset: float = 0.0) -> str: ...
@overload
def to_duckdb(text: None, *, scale: float = 1.0, offset: float = 0.0) -> None: ...
def to_duckdb(text: str | None, *, scale: float = 1.0, offset: float = 0.0) -> str | None:
    """Return a checked expression as DuckDB SQL (None stays None).

    With ``scale`` or ``offset``, the column ``value`` is read as ``value * scale +
    offset``: an input's filter compares its values in its variable's canonical unit.
    """
    if text is None:
        return None
    tree = _parse(text)
    if (scale, offset) != (1.0, 0.0):
        converted = sqlglot.parse_one(f"(value * {scale!r} + {offset!r})")

        def convert(node: exp.Expr) -> exp.Expr:
            if isinstance(node, exp.Column) and node.name == "value" and not node.table:
                return converted.copy()
            return node

        tree = tree.transform(convert)
    return tree.sql(dialect="duckdb")


def expression_columns(text: str) -> set[str]:
    """Return the names of the columns a (checked) expression refers to."""
    return {c.name for c in _parse(text).find_all(exp.Column)}
