"""USGS streamflow observations, from the USGS Water Data API (``dataretrieval.waterdata``).

Rows are ``series_key`` (``usgs-<site number>``, the gauge's location key), ``value_time`` (UTC),
``value`` in m3/s (:data:`UNIT`), ``approval_status`` (``Provisional`` or ``Approved``) and
``qualifier`` (:func:`qualifier`), a batch of sites at a time. As the legacy ingest did: no-data
values are dropped and, for instantaneous values, only those on the hour are kept. For a dataset
without those optional columns, drop them before writing.

Set ``API_USGS_PAT`` (a key from https://api.waterdata.usgs.gov/signup/) for the API's higher
rate limits, which the bulk files share; without it, small requests still work.

    from teehr_core.fetch import usgs

    for batch in usgs.fetch_streamflow(["usgs-01646500"], start, end):
        ...
"""

from __future__ import annotations

import logging
import os
import time
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
#: Daily mean (the statistic daily values are kept for).
DAILY_MEAN = "00003"
#: Sites per request.
BATCH_SITES = 100
#: Tries for a bulk file when rate limited (429), waiting between them: about an hour.
RATE_LIMIT_TRIES = 8
#: A site's whole record, all parameters, as Parquet (``service``: continuous or daily).
BULK_URL = "https://api.waterdata.usgs.gov/stac-files/{service}/USGS-{site}/USGS-{site}.parquet"

Service = Literal["iv", "dv"]

SCHEMA = pa.schema([
    pa.field("series_key", pa.string(), nullable=False),
    pa.field("value_time", pa.timestamp("us", "UTC"), nullable=False),
    pa.field("value", pa.float64()),
    pa.field("approval_status", pa.string()),
    pa.field("qualifier", pa.string()),
])  # fmt: skip


def qualifier(codes: object) -> str | None:
    """Return a value's qualifiers comma-separated, e.g. ``ESTIMATED,ICE`` (None: none).

    The API gives them as a list (in files, an array; None or NaN for none).
    """
    parts: list[str] = []
    if isinstance(codes, str):
        parts += [codes] if codes else []
    elif isinstance(codes, Iterable):
        parts += [str(q) for q in codes if q]
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
            statistic_id=DAILY_MEAN, skip_geometry=True,
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
    values = pd.to_numeric(frame["value"], errors="coerce").astype("float64")  # files: strings
    keep = values.notna() & ~values.isin(NODATA)
    if service == "iv" and hourly:
        keep &= (times.dt.minute == 0) & (times.dt.second == 0)
    none = pd.Series([None] * len(frame), index=frame.index, dtype="object")
    if "approval_status" in frame:
        status = (
            frame["approval_status"].astype("object").where(frame["approval_status"].notna(), None)
        )
    else:
        status = none
    codes = frame["qualifier"].map(qualifier) if "qualifier" in frame else none
    out = pd.DataFrame({
        "series_key": frame["monitoring_location_id"].map(series_key)[keep],
        "value_time": times[keep],
        "value": values[keep] * CFS_TO_M3S,
        "approval_status": status[keep],
        "qualifier": codes[keep],
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


def _bulk_file(service: Service, site: str) -> pd.DataFrame | None:
    """Return a site's discharge from its bulk Parquet file (None: no file)."""
    import io
    import urllib.error
    import urllib.request

    import pyarrow.parquet as pq

    url = BULK_URL.format(service="continuous" if service == "iv" else "daily",
                          site=site_number(site))  # fmt: skip
    headers = {"User-Agent": "teehr-core"}
    if os.environ.get("API_USGS_PAT"):  # the files share the API's rate limits
        headers["X-Api-Key"] = os.environ["API_USGS_PAT"]
    for attempt in range(RATE_LIMIT_TRIES):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=300) as resp:  # fmt: skip
                data = resp.read()
            break
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return None
            if err.code != 429 or attempt == RATE_LIMIT_TRIES - 1:
                raise
            wait = _retry_after(str(err.headers.get("Retry-After") or ""), attempt)
            logger.warning("USGS bulk files: rate limited (%s); waiting %d s", site, wait)
            time.sleep(wait)
    keep = [("parameter_code", "=", PARAMETER_CODE)]
    if service == "dv":
        keep.append(("statistic_id", "=", DAILY_MEAN))
    columns = ["time_series_id", "monitoring_location_id", "time", "value", "approval_status",
               "qualifier"]  # fmt: skip
    return pq.read_table(io.BytesIO(data), columns=columns, filters=keep).to_pandas()


def _retry_after(header: str, attempt: int) -> int:
    """Seconds to wait after a 429: the server's ``Retry-After``, else doubling to 15 min."""
    try:
        return max(1, int(header))
    except ValueError:
        return min(60 << attempt, 900)


def read_bulk(
    sites: Iterable[str],
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    service: Service = "iv",
    hourly: bool = True,
) -> Iterator[pa.Table]:
    """Yield each site's streamflow from its bulk file (:data:`BULK_URL`), one site at a time.

    For backfills and reloads: a site's whole record in one download, rather than paging the
    API. The rows are :func:`fetch_streamflow`'s, from ``start`` up to (not including) ``end``
    (default: the whole file). The files lag the API (continuous values by months), so
    follow a backfill with :func:`fetch_streamflow` from where they end. A site without a
    file yields an empty table.
    """
    sites = list(dict.fromkeys(sites))
    for i, site in enumerate(sites, 1):
        frame = _bulk_file(service, site)
        if frame is not None and not frame.empty and (start or end):
            times = pd.to_datetime(frame["time"], utc=True)
            inside = pd.Series(True, index=frame.index)
            if start:
                inside &= times >= start
            if end:
                inside &= times < end
            frame = frame[inside]
        rows = to_rows(frame, service, hourly)
        if frame is None:
            logger.warning("USGS %s: no bulk file for %s", service, site)
        logger.info("USGS %s bulk: site %d of %d (%s): %d rows", service, i, len(sites),
                    site, rows.num_rows)  # fmt: skip
        yield rows
