"""RFC streamflow forecasts at gauges, from the NWS National Water Prediction Service (NWPS) API.

Each gauge's latest forecast (``/nwps/v1/gauges/<lid>/stageflow/forecast``), as rows of
``series_key`` (``nwpsrfc-<lid>``, the lid in lower case, as legacy keyed them),
``reference_time``, ``value_time`` (UTC) and ``value`` in m3/s (:data:`UNIT`).

As the legacy ingest did:

- the reference time is one step before the forecast's first valid time (the API's
  ``issuedTime`` is when the product went out, after the forecast's start);
- no-data values (:data:`NODATA`) are dropped;
- a gauge whose forecast has no flow (stage only, or nothing) gives no forecast.

Unlike legacy, any step is kept (RFCs forecast at 1, 3 or 6 hours), and the step is the
smallest between the forecast's times, so a forecast with missing times (gaps) still has its
reference time.

    from teehr_core.fetch import nwps

    for lid, got in nwps.forecasts(["ANAW1", "BRKI4"]):
        if isinstance(got, nwps.Forecast):
            rows = got.rows
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Any, NamedTuple

import pyarrow as pa

logger = logging.getLogger(__name__)

#: The API's forecast endpoint for a gauge (its NWS location id, ``lid``).
FORECAST_URL = "https://api.water.noaa.gov/nwps/v1/gauges/{lid}/stageflow/forecast"
#: The unit rows are in (UCUM).
UNIT = "m3/s"
#: The API's flow units, to m3/s.
TO_M3S = {"kcfs": 1000 * 0.3048**3, "cfs": 0.3048**3}
#: Values the API uses for "no data".
NODATA = (-999.0, -9999.0)
#: Prefix of the series keys (legacy's location ids).
PREFIX = "nwpsrfc"
#: Seconds to connect and to read a response.
TIMEOUT = (5, 30)

SCHEMA = pa.schema([
    pa.field("series_key", pa.string(), nullable=False),
    pa.field("reference_time", pa.timestamp("us", "UTC"), nullable=False),
    pa.field("value_time", pa.timestamp("us", "UTC"), nullable=False),
    pa.field("value", pa.float64()),
])  # fmt: skip


class Forecast(NamedTuple):
    """A gauge's forecast: its series, issue (reference) time, step and rows."""

    series_key: str
    reference_time: datetime
    step: timedelta
    rows: pa.Table


def series_key(lid: str) -> str:
    """Return a gauge's series key.

    Parameters
    ----------
    lid : str
        The gauge's NWS location id, any case.

    Returns
    -------
    str
        ``nwpsrfc-<lid>``, the lid in lower case.

    Examples
    --------
    >>> series_key("ANAW1")
    'nwpsrfc-anaw1'
    """
    return f"{PREFIX}-{lid.lower()}"


def lid_of(key: str) -> str:
    """Return the NWS location id (upper case) of a series key (or of a lid).

    Examples
    --------
    >>> lid_of("nwpsrfc-anaw1")
    'ANAW1'
    """
    return key.removeprefix(f"{PREFIX}-").upper()


def resolution(step: timedelta) -> str:
    """Return a step as an ISO 8601 duration, as series tables record it.

    Examples
    --------
    >>> resolution(timedelta(hours=6))
    'PT6H'
    >>> resolution(timedelta(minutes=90))
    'PT90M'
    """
    minutes = int(step.total_seconds() // 60)
    return f"PT{minutes // 60}H" if minutes % 60 == 0 else f"PT{minutes}M"


def to_forecast(lid: str, payload: Mapping[str, Any]) -> Forecast | None:
    """Return the forecast in one API response (None: no flow forecast).

    Parameters
    ----------
    lid : str
        The gauge's NWS location id.
    payload : Mapping
        The parsed JSON of its ``stageflow/forecast`` response.

    Returns
    -------
    Forecast or None
        None when the forecast has no flow (``secondaryName`` isn't ``Flow``), its unit isn't
        one of :data:`TO_M3S`, or it has fewer than two valid values (no step).

    Examples
    --------
    >>> got = to_forecast("ANAW1", {
    ...     "secondaryName": "Flow", "secondaryUnits": "kcfs", "data": [
    ...         {"validTime": "2026-11-27T18:00:00Z", "secondary": 13.5},
    ...         {"validTime": "2026-11-28T00:00:00Z", "secondary": 13.6}]})
    >>> got.series_key, got.reference_time.isoformat(), got.step
    ('nwpsrfc-anaw1', '2026-11-27T12:00:00+00:00', datetime.timedelta(seconds=21600))
    """
    if payload.get("secondaryName") != "Flow":
        return None
    unit = payload.get("secondaryUnits")
    if unit not in TO_M3S:
        logger.warning("NWPS %s: flow in %r, not one of %s: skipped", lid, unit, list(TO_M3S))
        return None
    values: dict[datetime, float] = {}
    for point in payload.get("data") or []:
        value, valid = point.get("secondary"), point.get("validTime")
        if value is None or valid is None or float(value) in NODATA:
            continue
        values[datetime.fromisoformat(valid.replace("Z", "+00:00"))] = float(value)
    times = sorted(values)
    if len(times) < 2:
        return None
    step = min(b - a for a, b in pairwise(times))
    reference = times[0] - step
    key = series_key(lid)
    rows = pa.table({
        "series_key": [key] * len(times),
        "reference_time": [reference] * len(times),
        "value_time": times,
        "value": [values[t] * TO_M3S[unit] for t in times],
    }, schema=SCHEMA)  # fmt: skip
    return Forecast(key, reference, step, rows)


def forecast(lid: str) -> Forecast | None:
    """Fetch a gauge's latest forecast (None: no flow forecast, or no such gauge).

    Parameters
    ----------
    lid : str
        The gauge's NWS location id (or its series key).

    Returns
    -------
    Forecast or None
        See :func:`to_forecast`; also None when the API doesn't know the gauge (404).

    Raises
    ------
    urllib.error.URLError
        Any other failure to get it (5xx, a timeout).

    Examples
    --------
    >>> got = forecast("ANAW1")
    >>> got.rows.column_names
    ['series_key', 'reference_time', 'value_time', 'value']
    """
    lid = lid_of(lid)
    request = urllib.request.Request(FORECAST_URL.format(lid=lid),
                                     headers={"User-Agent": "teehr-core"})  # fmt: skip
    try:
        with urllib.request.urlopen(request, timeout=sum(TIMEOUT)) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise
    return to_forecast(lid, payload)


def forecasts(
    lids: Iterable[str], workers: int = 16
) -> Iterator[tuple[str, Forecast | Exception | None]]:
    """Fetch many gauges' latest forecasts, several at once, in the order given.

    A gauge that fails gives its exception rather than stopping the rest: the caller decides
    how many failures are too many.

    Parameters
    ----------
    lids : Iterable[str]
        NWS location ids (or series keys).
    workers : int, default 16
        Requests at once.

    Yields
    ------
    tuple of (str, Forecast or Exception or None)
        Each lid (as given) and its :func:`forecast`, or the exception fetching it raised.

    Examples
    --------
    >>> for lid, got in forecasts(["ANAW1", "BRKI4"]):
    ...     if isinstance(got, Exception):
    ...         print(lid, "failed:", got)
    """

    def one(lid: str) -> tuple[str, Forecast | Exception | None]:
        try:
            return lid, forecast(lid)
        except Exception as err:  # reported to the caller, per gauge
            return lid, err

    with ThreadPoolExecutor(workers) as pool:
        yield from pool.map(one, lids)
