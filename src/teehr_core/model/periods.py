"""Periods: the span of time one value covers (ADR-0009), from a system's period table.

Ported from VeriRace (``verirace/periods.py`` at 668ab7d).

A period is a named row with a ``form``:

- ``instant``: the value is a reading at ``value_time``; its window is that instant.
- ``duration``: the value covers ``length`` (ISO 8601, e.g. ``PT1H``), and ``value_time``
  is the window's ``start`` or ``end`` (``value_time_at``).
- ``calendar``: the value covers a span of the year from ``start`` to ``end`` (``--MM-DD``,
  both inclusive); its window is the occurrence of that span containing ``value_time``.
- ``absolute``: each value carries its own window, in ``period_start`` and ``period_end``.

Each dataset names a default period, and a series may name its own in its series table's
``period`` column. Values pair when their resolved windows are equal. The encoding is
shared with the TEEHR team; the common rows (``instant``, ``hour_starting``, ...) live in
the public lake, and a project can add its own. Like units, none are built into the code,
except that ``instant`` is what an ``inst`` value covers when nothing says otherwise.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

from teehr_core.durations import parse_duration
from teehr_core.sql import lit

__all__ = [
    "INSTANT",
    "Period",
    "PeriodError",
    "PeriodRegistry",
    "merge",
    "parse_iso_duration",
    "window_sql",
]

Form = Literal["instant", "duration", "calendar", "absolute"]
FORMS: tuple[str, ...] = ("instant", "duration", "calendar", "absolute")

#: The period an ``inst`` value covers unless its dataset or series names another.
INSTANT = "instant"

_MONTH_DAY = re.compile(r"--(\d{2})-(\d{2})")


class PeriodError(ValueError):
    """An invalid period definition, or an unknown period name."""


def parse_iso_duration(text: str) -> timedelta:
    """Parse an ISO 8601 duration of fixed length (weeks, days, hours, minutes, seconds).

    Years and months are refused: their length varies (a calendar period covers those).

    Raises
    ------
    ValueError
        If ``text`` isn't such a duration, or is zero.
    """
    return parse_duration(text, allow_zero=False)


def _month_day(text: str) -> tuple[int, int]:
    match = _MONTH_DAY.fullmatch(text)
    if match is None:
        raise ValueError(f"invalid month-day {text!r}; expected --MM-DD such as '--04-01'")
    month, day = int(match.group(1)), int(match.group(2))
    try:
        date(2001, month, day)  # a common year: --02-29 isn't in every year
    except ValueError:
        raise ValueError(f"month-day {text!r} isn't a day in every year") from None
    return month, day


@dataclass(frozen=True)
class Period:
    """One period table row."""

    name: str
    form: str
    length: str | None = None
    value_time_at: str | None = None
    start: str | None = None
    end: str | None = None
    description: str = ""

    def problems(self) -> list[str]:
        """What's wrong with this definition (empty: nothing)."""
        out: list[str] = []
        if not self.name:
            out.append("a period needs a name")
        if self.form not in FORMS:
            return [*out, f"form must be one of {', '.join(FORMS)}, got {self.form!r}"]
        used = {
            "instant": set(),
            "duration": {"length", "value_time_at"},
            "calendar": {"start", "end"},
            "absolute": set(),
        }[self.form]
        for column in ("length", "value_time_at", "start", "end"):
            given = getattr(self, column) is not None
            if column in used and not given:
                out.append(f"a {self.form} period needs {column}")
            elif given and column not in used:
                out.append(f"a {self.form} period has no {column}")
        if self.length is not None:
            try:
                parse_iso_duration(self.length)
            except ValueError as exc:
                out.append(f"length: {exc}")
        if self.value_time_at is not None and self.value_time_at not in ("start", "end"):
            out.append(f"value_time_at must be start or end, got {self.value_time_at!r}")
        for column in ("start", "end"):
            text = getattr(self, column)
            if text is not None:
                try:
                    _month_day(text)
                except ValueError as exc:
                    out.append(f"{column}: {exc}")
        return [f"period {self.name!r}: {p}" for p in out]

    def window_sql(self, row: str) -> tuple[str, str]:
        """Return SQL for the window ``[start, end)`` of a value in row alias ``row``.

        ``row`` has ``value_time`` (and, for ``absolute``, ``period_start`` and
        ``period_end``). A calendar value outside every occurrence of its span (e.g. a
        September value of an April-July period) gets a NULL window, so it pairs with
        nothing. Calendar spans are in UTC, the connection's time zone.
        """
        vt = f"{row}.value_time"
        if self.form == "instant":
            return vt, vt
        if self.form == "duration":
            assert self.length is not None
            secs = int(parse_iso_duration(self.length).total_seconds())
            if self.value_time_at == "start":
                return vt, f"{vt} + to_seconds({secs})"
            return f"{vt} - to_seconds({secs})", vt
        if self.form == "absolute":
            return f"{row}.period_start", f"{row}.period_end"
        assert self.start is not None and self.end is not None
        (sm, sd), (em, ed) = _month_day(self.start), _month_day(self.end)
        wraps = 1 if (em, ed) < (sm, sd) else 0  # e.g. --10-01 to --09-30
        y = f"year({vt})"

        def at(year: str, month: int, day: int) -> str:
            return f"make_timestamptz({year}, {month}, {day}, 0, 0, 0.0, 'UTC')"

        # The occurrence starting in the value's year, or else the one before it.
        first = f"(CASE WHEN {vt} >= {at(y, sm, sd)} THEN {y} ELSE {y} - 1 END)"
        start = at(first, sm, sd)
        end = f"({at(f'{first} + {wraps}', em, ed)} + INTERVAL 1 DAY)"
        return (
            f"(CASE WHEN {vt} < {end} THEN {start} END)",
            f"(CASE WHEN {vt} < {end} THEN {end} END)",
        )


class PeriodRegistry:
    """Known periods, by name."""

    def __init__(self, periods: Iterable[Period]) -> None:
        self._periods: dict[str, Period] = {}
        problems: list[str] = []
        for p in periods:
            if p.name in self._periods:
                problems.append(f"period {p.name!r} is defined twice")
            problems += p.problems()
            self._periods[p.name] = p
        if problems:
            raise PeriodError("; ".join(problems))

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> PeriodRegistry:
        """Build from a period table's rows (mappings with the table's columns)."""
        return cls(
            Period(
                name=r["name"],
                form=r["form"],
                length=r.get("length"),
                value_time_at=r.get("value_time_at"),
                start=r.get("start"),
                end=r.get("end"),
                description=r.get("description") or "",
            )
            for r in rows
        )

    @classmethod
    def default(cls) -> PeriodRegistry:
        """Only ``instant``: for catalogs built without the warehouse's periods."""
        return cls([Period(INSTANT, "instant")])

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._periods

    def __iter__(self) -> Any:
        return iter(self._periods.values())

    def __len__(self) -> int:
        return len(self._periods)

    def names(self) -> list[str]:
        """Period names, sorted."""
        return sorted(self._periods)

    def get(self, name: str) -> Period:
        """Return the period called ``name``.

        Raises
        ------
        PeriodError
            If there is none.
        """
        period = self._periods.get(name)
        if period is None:
            raise PeriodError(f"unknown period {name!r}; known periods: {', '.join(self.names())}")
        return period


def merge(registries: Sequence[PeriodRegistry]) -> PeriodRegistry:
    """Combine registries (e.g. public and project periods); a name may appear only once."""
    return PeriodRegistry(p for r in registries for p in r)


def window_sql(periods: Mapping[str, Period], row: str, name: str | None) -> tuple[str, str]:
    """Return SQL for a value's window when its period may differ by series.

    ``periods`` are the periods the values use; with one, its window. With several,
    ``name`` is the SQL expression for each row's period name, and the window is chosen
    by it.
    """
    if len(periods) == 1:
        return next(iter(periods.values())).window_sql(row)
    assert name is not None
    starts, ends = [], []
    for key, period in sorted(periods.items()):
        s, e = period.window_sql(row)
        starts.append(f"WHEN {lit(key)} THEN {s}")
        ends.append(f"WHEN {lit(key)} THEN {e}")
    return (
        f"(CASE {name} {' '.join(starts)} END)",
        f"(CASE {name} {' '.join(ends)} END)",
    )
