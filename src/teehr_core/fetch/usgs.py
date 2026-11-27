"""USGS streamflow observations, from the USGS Water Data API (``dataretrieval.waterdata``).

Rows are ``series_key`` (``usgs-<site number>``, the gauge's location key), ``value_time`` (UTC),
``value`` in m3/s (:data:`UNIT`), ``approval_status`` (``Provisional`` or ``Approved``) and
``qualifier`` (:func:`qualifier`), a batch of sites at a time. As the legacy ingest did: no-data
values are dropped and, for instantaneous values, only those on the hour are kept. For a dataset
without those optional columns, drop them before writing.

A site can have several discharge time series at once (a structure's and a river's, a primary
and a secondary sensor, ...). One is kept per site (:func:`choose_series`): a pinned one, else
by the USGS's own metadata.

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
from collections.abc import Iterable, Iterator, Mapping
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
#: Sites per metadata request (:func:`choose_series`).
METADATA_SITES = 200
#: A series whose last value is within this of its site's latest still reports.
REPORTING = timedelta(days=7)
#: Words in a series' description marking the site's whole flow (vs. a part of it).
WHOLE_FLOW = ("total", "combined")
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


def _series_metadata(sites: list[str] | None, service: Service) -> pd.DataFrame:
    """Return the USGS's metadata of the sites' discharge time series (every site's: None)."""
    from dataretrieval import waterdata

    period, statistic = ("Points", None) if service == "iv" else ("Daily", DAILY_MEAN)
    fields = ["id", "monitoring_location_id", "primary", "sublocation_identifier",
              "web_description", "begin_utc", "end_utc"]  # fmt: skip
    ids = [f"USGS-{site_number(s)}" for s in sites] if sites is not None else []
    groups: list[list[str] | None] = (
        [None]
        if sites is None
        else [ids[i : i + METADATA_SITES] for i in range(0, len(ids), METADATA_SITES)]
    )
    frames = []
    for group in groups:
        frame, _ = waterdata.get_time_series_metadata(
            monitoring_location_id=group,
            parameter_code=PARAMETER_CODE, computation_period_identifier=period,
            statistic_id=statistic, properties=fields, skip_geometry=True,
        )  # fmt: skip
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def choose_series(
    sites: Iterable[str] | None = None,
    *,
    pins: Mapping[str, str] | None = None,
    service: Service = "iv",
) -> dict[str, str]:
    """Choose the discharge time series to keep at each site that has several.

    A site can have several at once: a structure's and a river's, a primary and a secondary
    sensor, an old series and its replacement. In order, the one kept is:

    1. the pinned one (``pins``), if the site has it and it still reports (a pin to a
       series that stopped is warned about, and the rules decide);
    2. the one the USGS marks primary;
    3. one still reporting (its last value within :data:`REPORTING` of the site's latest);
    4. one without a sublocation (e.g. not one of a site's velocity meters);
    5. one whose description says it's the whole flow (:data:`WHOLE_FLOW`: "Total ...",
       "Combined");
    6. the longest record.

    A site still decided only by its record's length is logged as a warning: pin it to be
    sure.

    Parameters
    ----------
    sites : Iterable[str], optional
        ``usgs-01646500``, ``USGS-01646500`` or ``01646500``; every site, by default (two
        requests for the whole country).
    pins : Mapping[str, str], optional
        A site's time series id (any form of site), kept over the rules.
    service : {"iv", "dv"}, default "iv"
        Instantaneous values' series, or daily means'.

    Returns
    -------
    dict[str, str]
        ``USGS-<site number>`` to the time series id to keep, for the sites with several
        series (or a pin): pass it as ``series`` to :func:`fetch_streamflow`,
        :func:`read_bulk` or :func:`to_rows`.

    Examples
    --------
    >>> choose_series(["02430626"])  # a primary series and a newer secondary one
    {'USGS-02430626': 'f86922c5...'}
    >>> choose_series(["02237700"], pins={"usgs-02237700": "f5dc2874291f4b49813a55d8c6dceef5"})
    {'USGS-02237700': 'f5dc2874291f4b49813a55d8c6dceef5'}
    """
    pinned = {f"USGS-{site_number(s)}": ts for s, ts in (pins or {}).items()}
    meta = _series_metadata(None if sites is None else list(sites), service)
    chosen: dict[str, str] = {}
    by_length: list[str] = []
    if not meta.empty:
        for site, rows in meta.groupby("monitoring_location_id", sort=True):
            pin = rows[rows["time_series_id"] == pinned.get(site)]
            if not pin.empty:
                end = pd.to_datetime(rows["end_utc"], utc=True)
                last = pd.to_datetime(pin["end_utc"], utc=True).iloc[0]
                if last >= end.max() - REPORTING:
                    chosen[site] = pinned[site]
                    continue
                logger.warning("%s: pinned time series %s stopped reporting (last %s); "
                               "update the pin. Choosing by the rules.", site, pinned[site],
                               last)  # fmt: skip
                pinned.pop(site)
            if len(rows) < 2:
                continue
            kept, decided = _rank(rows)
            chosen[site] = kept
            if not decided:
                by_length.append(site)
    for site, ts in pinned.items():
        if site not in chosen:
            logger.warning("%s: pinned time series %s isn't one of its series; ignored", site, ts)
    if by_length:
        logger.warning("sites with several discharge time series chosen by record length "
                       "only (pin to be sure): %s", ", ".join(by_length))  # fmt: skip
    return chosen


def _rank(rows: pd.DataFrame) -> tuple[str, bool]:
    """Return a site's series to keep, and whether a rule before record length decided it."""
    end = pd.to_datetime(rows["end_utc"], utc=True)
    begin = pd.to_datetime(rows["begin_utc"], utc=True)
    text = rows["web_description"].fillna("").str.lower()
    rules = [
        rows["primary"].fillna("").eq("Primary"),
        end >= end.max() - REPORTING,
        rows["sublocation_identifier"].isna() | rows["sublocation_identifier"].eq(""),
        text.apply(lambda t: any(w in t for w in WHOLE_FLOW)),
    ]
    left = pd.Series(True, index=rows.index)
    for rule in rules:
        if (left & rule).sum() >= 1:
            left &= rule
    # The longest record (a missing begin or end counts as none), then the id: always the same.
    ranked = pd.DataFrame({"length": (end - begin).fillna(pd.Timedelta(0)),
                           "id": rows["time_series_id"].astype(str)})[left]  # fmt: skip
    ranked = ranked.sort_values(["length", "id"], ascending=[False, True])
    return str(ranked["id"].iloc[0]), int(left.sum()) == 1


def _one_series_per_site(frame: pd.DataFrame, series: Mapping[str, str] | None) -> pd.DataFrame:
    """Keep one time series per site: ``series``' choice, else the latest, then the longest.

    A site with several would repeat its times. Without a choice for it (or when the chosen
    series isn't in ``frame``), the series with the latest value is kept, then the one with
    most values, with a warning.
    """
    if "time_series_id" not in frame or frame.empty:
        return frame
    stats = frame.groupby(["monitoring_location_id", "time_series_id"])["time"].agg(["max", "size"])
    several = stats.groupby(level=0).size()
    several = several[several > 1]
    if several.empty:
        return frame
    present = set(stats.index)
    kept = [(site, series[site]) for site in several.index
            if series and site in series and (site, series[site]) in present]  # fmt: skip
    decided = {site for site, _ in kept}
    rest = stats[~stats.index.get_level_values(0).isin(decided)]
    rest = rest[rest.index.get_level_values(0).isin(several.index)]
    if not rest.empty:
        ranked = rest.sort_values(["max", "size"], ascending=False)
        kept += list(ranked.groupby(level=0).head(1).index)
        logger.warning("sites with more than one discharge time series and no choice for them "
                       "(kept the latest): %s",
                       ", ".join(sorted(set(rest.index.get_level_values(0)))))  # fmt: skip
    index = frame.set_index(["monitoring_location_id", "time_series_id"]).index
    single = ~frame["monitoring_location_id"].isin(several.index).to_numpy()
    return frame[single | index.isin(kept)]


def to_rows(
    frame: pd.DataFrame,
    service: Service = "iv",
    hourly: bool = True,
    series: Mapping[str, str] | None = None,
) -> pa.Table:
    """Return the API's rows as v1 timeseries rows (:data:`SCHEMA`, m3/s).

    The API's rows are ``monitoring_location_id``, ``time``, ``value`` in ft3/s,
    ``approval_status`` and ``qualifier``. No-data
    values are dropped and, for instantaneous values with ``hourly``, those not on the hour.
    At a site with several time series, ``series``' choice (:func:`choose_series`) is kept.
    """
    if frame is None or frame.empty:
        return SCHEMA.empty_table()
    frame = _one_series_per_site(frame, series)
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
    series: Mapping[str, str] | None = None,
) -> Iterator[pa.Table]:
    """Yield the sites' streamflow from ``start`` up to (not including) ``end``, UTC.

    As v1 timeseries rows in m3/s (:func:`to_rows`), one batch of ``batch_sites`` sites at a time.

    ``service``: ``iv`` instantaneous values (default; on the hour with ``hourly``), ``dv``
    daily means (at midnight UTC of their day). A batch the API has nothing for is empty.
    ``series``: the time series to keep at sites with several (:func:`choose_series`, which
    is asked for each batch's sites when it's None; pass it to choose once, or to pin).
    """
    sites = list(dict.fromkeys(sites))  # unique, in order
    for i in range(0, len(sites), batch_sites):
        batch = sites[i : i + batch_sites]
        chosen = series if series is not None else choose_series(batch, service=service)
        rows = to_rows(_request(service, batch, start, end), service, hourly, chosen)
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
    series: Mapping[str, str] | None = None,
) -> Iterator[pa.Table]:
    """Yield each site's streamflow from its bulk file (:data:`BULK_URL`), one site at a time.

    For backfills and reloads: a site's whole record in one download, rather than paging the
    API. The rows are :func:`fetch_streamflow`'s, from ``start`` up to (not including) ``end``
    (default: the whole file). The files lag the API (continuous values by months), so
    follow a backfill with :func:`fetch_streamflow` from where they end. A site without a
    file yields an empty table. ``series`` as for :func:`fetch_streamflow` (asked for all the
    sites at once when None).
    """
    sites = list(dict.fromkeys(sites))
    chosen = series if series is not None else choose_series(sites, service=service)
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
        rows = to_rows(frame, service, hourly, chosen)
        if frame is None:
            logger.warning("USGS %s: no bulk file for %s", service, site)
        logger.info("USGS %s bulk: site %d of %d (%s): %d rows", service, i, len(sites),
                    site, rows.num_rows)  # fmt: skip
        yield rows
