"""USGS streamflow observations, from the USGS Water Data API (``dataretrieval.waterdata``).

Rows are ``series_key`` (``usgs-<site number>``, the gauge's location key), ``value_time`` (UTC),
``value`` in m3/s (:data:`UNIT`) and ``quality`` (:func:`quality`), a batch of sites at a time.
As the legacy ingest did: no-data values are dropped and, for instantaneous values, only those
on the hour are kept. A dataset without a ``quality`` column takes the rows without it.

Set ``API_USGS_PAT`` (a key from https://api.waterdata.usgs.gov/signup/) for the API's higher
rate limits; without it, small requests still work.

    from teehr_core.fetch import usgs

    for batch in usgs.fetch_streamflow(["usgs-01646500"], start, end):
        ...
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta
from typing import Literal

import pandas as pd
import pyarrow as pa

logger = logging.getLogger(__name__)

#: Discharge, cubic feet per second.
PARAMETER_CODE = "00060"
#: The unit rows are in (UCUM): the source's ft3/s, converted.
UNIT = "m3/s"
CFS_TO_M3S = 0.3048**3
#: Values the USGS uses for "no data".
NODATA = (-999999.0, -99999.0, -9999.0, -999.0)
#: Sites per request.
BATCH_SITES = 100

Service = Literal["iv", "dv"]

SCHEMA = pa.schema([
    pa.field("series_key", pa.string(), nullable=False),
    pa.field("value_time", pa.timestamp("us", "UTC"), nullable=False),
    pa.field("value", pa.float64()),
    pa.field("quality", pa.string()),
])  # fmt: skip


def quality(approval_status: object, qualifier: object) -> str | None:
    """Return a value's quality: its approval status then its qualifiers, comma-separated.

    E.g. ``Provisional``, ``Approved,ESTIMATED`` (the API's own words). None: neither given.
    """
    parts = [approval_status] if isinstance(approval_status, str) and approval_status else []
    if isinstance(qualifier, str):
        parts += [qualifier] if qualifier else []
    elif isinstance(qualifier, Iterable):  # a list or array (None or NaN: none)
        parts += [str(q) for q in qualifier if q]
    return ",".join(parts) or None


def site_number(site: str) -> str:
    """Return the bare site number of ``site``.

    ``site`` is ``usgs-01646500``, ``USGS-01646500`` or ``01646500``.
    """
    return site.split("-", 1)[1] if site.lower().startswith("usgs-") else site


def series_key(site: str) -> str:
    """Return the series (and location) key of ``site``: ``usgs-<site number>``."""
    return f"usgs-{site_number(site)}"


def _request(service: Service, sites: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    """One request to the Water Data API: the sites' discharge from ``start`` up to ``end``."""
    from dataretrieval import waterdata

    ids = [f"USGS-{site_number(s)}" for s in sites]
    last = end - timedelta(minutes=1)  # the API's range includes its end
    if service == "iv":
        time = f"{start:%Y-%m-%dT%H:%M:%S}Z/{last:%Y-%m-%dT%H:%M:%S}Z"
        frame, _ = waterdata.get_continuous(
            monitoring_location_id=ids, parameter_code=PARAMETER_CODE, time=time
        )
    else:
        time = f"{start:%Y-%m-%d}/{last:%Y-%m-%d}"
        frame, _ = waterdata.get_daily(
            monitoring_location_id=ids, parameter_code=PARAMETER_CODE, time=time,
            skip_geometry=True,
        )  # fmt: skip
    return frame


def _one_series_per_site(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one time series per site: the one still reporting, then the one with most values.

    A site may have several, which would repeat its times. Usually an old series and its
    current replacement, overlapping for a while: the one with the latest value is kept.
    """
    if "time_series_id" not in frame or frame.empty:
        return frame
    stats = frame.groupby(["monitoring_location_id", "time_series_id"])["time"].agg(["max", "size"])
    several = stats.groupby(level=0).size()
    several = several[several > 1]
    if several.empty:
        return frame
    ranked = stats.sort_values(["max", "size"], ascending=False)
    kept = ranked.groupby(level=0).head(1).index
    logger.warning("sites with more than one discharge time series (kept the latest): %s",
                   ", ".join(several.index))  # fmt: skip
    keep = frame.set_index(["monitoring_location_id", "time_series_id"]).index.isin(kept)
    return frame[keep]


def to_rows(frame: pd.DataFrame, service: Service = "iv", hourly: bool = True) -> pa.Table:
    """Return the API's rows as v1 timeseries rows (:data:`SCHEMA`, m3/s).

    The API's rows are ``monitoring_location_id``, ``time``, ``value`` in ft3/s,
    ``approval_status`` and ``qualifier``. No-data
    values are dropped and, for instantaneous values with ``hourly``, those not on the hour.
    """
    if frame is None or frame.empty:
        return SCHEMA.empty_table()
    frame = _one_series_per_site(frame)
    times = pd.to_datetime(frame["time"])
    times = times.dt.tz_localize("UTC") if times.dt.tz is None else times.dt.tz_convert("UTC")
    values = frame["value"].astype("float64")
    keep = values.notna() & ~values.isin(NODATA)
    if service == "iv" and hourly:
        keep &= (times.dt.minute == 0) & (times.dt.second == 0)
    status = frame["approval_status"] if "approval_status" in frame else [None] * len(frame)
    qualifiers = frame["qualifier"] if "qualifier" in frame else [None] * len(frame)
    flags = pd.Series([quality(s, q) for s, q in zip(status, qualifiers, strict=True)],
                      index=frame.index, dtype="object")  # fmt: skip
    out = pd.DataFrame({
        "series_key": frame["monitoring_location_id"].map(series_key)[keep],
        "value_time": times[keep],
        "value": values[keep] * CFS_TO_M3S,
        "quality": flags[keep],
    })  # fmt: skip
    out = out.drop_duplicates(["series_key", "value_time"])
    return pa.Table.from_pandas(out, schema=SCHEMA, preserve_index=False)


def fetch_streamflow(
    sites: Iterable[str],
    start: datetime,
    end: datetime,
    *,
    service: Service = "iv",
    hourly: bool = True,
    batch_sites: int = BATCH_SITES,
) -> Iterator[pa.Table]:
    """Yield the sites' streamflow from ``start`` up to (not including) ``end``, UTC.

    As v1 timeseries rows in m3/s (:func:`to_rows`), one batch of ``batch_sites`` sites at a time.

    ``service``: ``iv`` instantaneous values (default; on the hour with ``hourly``), ``dv``
    daily means (at midnight UTC of their day). A batch the API has nothing for is empty.
    """
    sites = list(dict.fromkeys(sites))  # unique, in order
    for i in range(0, len(sites), batch_sites):
        batch = sites[i : i + batch_sites]
        rows = to_rows(_request(service, batch, start, end), service, hourly)
        logger.info("USGS %s: sites %d-%d of %d: %d rows", service, i + 1, i + len(batch),
                    len(sites), rows.num_rows)  # fmt: skip
        yield rows
