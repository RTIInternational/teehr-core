"""Durations as ISO 8601 strings such as ``PT15M``, ``PT6H`` or ``P1D``.

Ported from VeriRace (``verirace/durations.py`` at 668ab7d).

Durations of fixed length only: weeks, days, hours, minutes and seconds. Years and months
vary in length; where a calendar span is meant (a bootstrap block of months, a calendar
period), it's said separately (:func:`parse_months`).
"""

from __future__ import annotations

import re
from datetime import timedelta

__all__ = [
    "DURATION_PATTERN",
    "MONTHS_PATTERN",
    "format_duration",
    "format_hours_label",
    "parse_duration",
    "parse_months",
]

#: A fixed-length ISO 8601 duration (``PT1H``, ``PT15M``, ``P1D``, ``P1DT6H``, ``P2W``).
#: The pattern alone allows ``P`` and ``PT``; :func:`parse_duration` refuses them.
DURATION_PATTERN = r"^P(\d+W)?(\d+D)?(T(\d+H)?(\d+M)?(\d+S)?)?$"
#: A whole number of calendar months as ISO 8601 (``P1M``, ``P3M``).
MONTHS_PATTERN = r"^P\d+M$"

_ISO = re.compile(r"P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?")


def parse_duration(text: str, *, allow_zero: bool = True) -> timedelta:
    """Parse a fixed-length ISO 8601 duration such as ``PT15M``, ``PT6H`` or ``P1D``.

    Raises
    ------
    ValueError
        If ``text`` isn't such a duration (years and months are refused), or is zero
        when ``allow_zero`` is false.
    """
    match = _ISO.fullmatch(text)
    if match is None or text in ("P", "PT") or text.endswith("T"):
        hint = " (a calendar span of months is P1M, where months are allowed)"
        raise ValueError(
            f"invalid duration {text!r}; expected ISO 8601 such as 'PT15M', 'PT6H' or 'P1D'"
            + (hint if re.fullmatch(MONTHS_PATTERN, text) else "")
        )
    w, d, h, m, s = (int(g) if g else 0 for g in match.groups())
    td = timedelta(weeks=w, days=d, hours=h, minutes=m, seconds=s)
    if not allow_zero and td <= timedelta(0):
        raise ValueError(f"duration {text!r} is zero")
    return td


def parse_months(text: str) -> int:
    """Parse a number of calendar months given as ISO 8601 (``P1M``, ``P3M``)."""
    if re.fullmatch(MONTHS_PATTERN, text) is None:
        raise ValueError(f"invalid months {text!r}; expected ISO 8601 such as 'P1M' or 'P3M'")
    return int(text[1:-1])


def format_duration(td: timedelta) -> str:
    """Format a duration as ISO 8601: ``P1D``, ``PT6H``, ``PT90M``, ``P1DT6H``, ``PT0S``."""
    seconds = int(td.total_seconds())
    if seconds == 0:
        return "PT0S"
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    time = "".join(f"{v}{u}" for v, u in ((hours, "H"), (minutes, "M"), (secs, "S")) if v)
    return f"P{f'{days}D' if days else ''}{f'T{time}' if time else ''}"


def format_hours_label(td: timedelta) -> str:
    """Format a duration for labels: whole hours as ``6h``, otherwise minutes as ``90min``."""
    seconds = int(td.total_seconds())
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    return f"{seconds // 60}min"
