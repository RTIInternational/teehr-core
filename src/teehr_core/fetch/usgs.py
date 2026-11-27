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
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Literal, NamedTuple

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


class Series(NamedTuple):
    """One of a site's time series: its id and the span of its values (None: open)."""

    id: str
    begin: datetime | None
    end: datetime | None

    def covers(self, times: pd.Series) -> pd.Series:
        """Return which of ``times`` are in the series' span."""
        inside = pd.Series(True, index=times.index)
        if self.begin is not None:
            inside &= times >= _utc(self.begin)
        if self.end is not None:
            inside &= times <= _utc(self.end)
        return inside


def _utc(when: datetime) -> pd.Timestamp:
    """``when`` as a UTC timestamp (naive: taken as UTC)."""
    stamp = pd.Timestamp(when)
    return stamp.tz_localize("UTC") if stamp.tz is None else stamp.tz_convert("UTC")


def choose_series(
    sites: Iterable[str] | None = None,
    *,
    pins: Mapping[str, str] | None = None,
    service: Service = "iv",
) -> dict[str, list[Series]]:
    """Rank the discharge time series of each site that has several, best first.

    A site can have several: a structure's and a river's, a primary and a secondary sensor,
    an old series and its replacement. At each time, the best-ranked series whose span
    covers it is kept (:func:`to_rows`), so a replaced series fills the years before its
    successor began, and series reporting at the same time never mix. Ranked by, in order:

    1. a pin (``pins``; a pin to a series that stopped reporting is warned about: it only
       covers the times it has);
    2. the USGS's primary flag (non-primary series are provisional, kept 120 days);
    3. still reporting (its last value within :data:`REPORTING` of the site's latest; its
       span is then open-ended);
    4. no sublocation (e.g. not one of a site's velocity meters);
    5. a description saying it's the whole flow (:data:`WHOLE_FLOW`: "Total ...",
       "Combined");
    6. the longest record, then the id.

    Sites where two series reporting at the same time are ranked by record length only are
    logged as a warning: pin them to be sure.

    Parameters
    ----------
    sites : Iterable[str], optional
        ``usgs-01646500``, ``USGS-01646500`` or ``01646500``; every site, by default (two
        requests for the whole country, about 5 s).
    pins : Mapping[str, str], optional
        A site's time series id (any form of site), ranked over the rules.
    service : {"iv", "dv"}, default "iv"
        Instantaneous values' series, or daily means'.

    Returns
    -------
    dict[str, list[Series]]
        ``USGS-<site number>`` to its series, best first, for the sites with several (or a
        pin): pass it as ``series`` to :func:`fetch_streamflow`, :func:`read_bulk` or
        :func:`to_rows`.

    Examples
    --------
    A primary series and a newer secondary sensor:

    >>> [s.id for s in choose_series(["02430626"])["USGS-02430626"]]
    ['f86922c5...', 'c69d656b...']

    A pin first:

    >>> choose_series(["02237700"], pins={"usgs-02237700": "8ed60838..."})["USGS-02237700"][0].id
    '8ed60838...'
    """
    pinned = {f"USGS-{site_number(s)}": ts for s, ts in (pins or {}).items()}
    meta = _series_metadata(None if sites is None else list(sites), service)
    ranked: dict[str, list[Series]] = {}
    by_length: list[str] = []
    if not meta.empty:
        for site, rows in meta.groupby("monitoring_location_id", sort=True):
            pin = pinned.pop(str(site), None)
            if pin is not None and pin not in set(rows["time_series_id"]):
                logger.warning("%s: pinned time series %s isn't one of its series; ignored",
                               site, pin)  # fmt: skip
                pin = None
            if len(rows) < 2 and pin is None:
                continue
            order, tied = _rank(rows, pin)
            ranked[str(site)] = order
            if tied:
                by_length.append(str(site))
    for site, pin in pinned.items():
        logger.warning("%s: pinned time series %s isn't one of its series; ignored", site, pin)
    if by_length:
        logger.warning("sites with several discharge time series reporting at once, ranked by "
                       "record length only (pin to be sure): %s", ", ".join(by_length))  # fmt: skip
    return ranked


def _rank(rows: pd.DataFrame, pin: str | None) -> tuple[list[Series], bool]:
    """Return a site's series best first, and whether the top two tie on every rule."""
    begin = pd.to_datetime(rows["begin_utc"], utc=True)
    end = pd.to_datetime(rows["end_utc"], utc=True)
    reporting = end >= end.max() - REPORTING
    ids = rows["time_series_id"].astype(str)
    text = rows["web_description"].fillna("").astype(str).str.lower()
    sub = rows["sublocation_identifier"].fillna("").astype(str)
    if pin is not None and not reporting[ids == pin].any():
        logger.warning("%s: pinned time series %s stopped reporting (last %s); update the pin",
                       rows["monitoring_location_id"].iloc[0], pin,
                       end[ids == pin].iloc[0])  # fmt: skip
    keys = pd.DataFrame({
        "pin": ids.ne(pin) if pin is not None else False,
        "primary": rows["primary"].fillna("").ne("Primary"),
        "reporting": ~reporting,
        "sublocation": sub.ne(""),
        "whole": ~text.apply(lambda t: any(w in t for w in WHOLE_FLOW)),
        "length": -(end - begin).fillna(pd.Timedelta(0)).dt.total_seconds(),
        "id": ids,
    })  # fmt: skip
    order = keys.sort_values(list(keys.columns)).index
    series = [Series(ids[i], None if pd.isna(begin[i]) else begin[i].to_pydatetime(),
                     None if reporting[i] or pd.isna(end[i]) else end[i].to_pydatetime())
              for i in order]  # fmt: skip
    rules = ["pin", "primary", "reporting", "sublocation", "whole"]
    first, second = (order[0], order[1]) if len(order) > 1 else (order[0], None)
    tied = (
        second is not None
        and bool(reporting[first])
        and bool(reporting[second])
        and (keys.loc[first, rules].tolist() == keys.loc[second, rules].tolist())
    )
    return series, tied


def _one_series_per_site(
    frame: pd.DataFrame, series: Mapping[str, Sequence[Series]] | None
) -> pd.DataFrame:
    """Keep one time series per site at each time: ``series``' ranking, else the latest.

    At a site with a ranking, each time is the best-ranked series' whose span covers it
    (none: the best-ranked one with a value there). A site with several series and no
    ranking keeps the series with the latest value, then the one with most values, with a
    warning.
    """
    if "time_series_id" not in frame or frame.empty:
        return frame
    stats = frame.groupby(["monitoring_location_id", "time_series_id"])["time"].agg(["max", "size"])
    several = stats.groupby(level=0).size()
    several = set(several[several > 1].index)
    if not several:
        return frame
    keep = pd.Series(~frame["monitoring_location_id"].isin(several), index=frame.index)
    times = pd.to_datetime(frame["time"], utc=True)
    unranked = []
    for site in sorted(several):
        at = frame["monitoring_location_id"] == site
        order = list(series.get(site, ())) if series else []
        if not order:
            unranked.append(site)
            continue
        site_times, ids = times[at], frame.loc[at, "time_series_id"].astype(str)
        owner = pd.Series(None, index=site_times.index, dtype=object)
        for s in order:
            owner[owner.isna() & s.covers(site_times)] = s.id
        rank = {s.id: i for i, s in enumerate(order)}
        priority = ids.map(rank).fillna(len(order))
        free = owner.isna()  # no span covers the time: the best-ranked value there
        best = priority[free].groupby(site_times[free]).transform("min")
        keep[at] = (ids == owner) | (free & (priority == best.reindex(ids.index)))
    if unranked:
        rest = stats.loc[unranked].sort_values(["max", "size"], ascending=False)
        kept = set(rest.groupby(level=0).head(1).index)
        index = frame.set_index(["monitoring_location_id", "time_series_id"]).index
        keep |= pd.Series(index.isin(kept), index=frame.index)
        logger.warning("sites with more than one discharge time series and no ranking for "
                       "them (kept the latest): %s", ", ".join(unranked))  # fmt: skip
    return frame[keep.to_numpy()]


def to_rows(
    frame: pd.DataFrame,
    service: Service = "iv",
    hourly: bool = True,
    series: Mapping[str, Sequence[Series]] | None = None,
) -> pa.Table:
    """Return the API's rows as v1 timeseries rows (:data:`SCHEMA`, m3/s).

    The API's rows are ``monitoring_location_id``, ``time``, ``value`` in ft3/s,
    ``approval_status`` and ``qualifier``. No-data
    values are dropped and, for instantaneous values with ``hourly``, those not on the hour.
    At a site with several time series, each time keeps ``series``' choice
    (:func:`choose_series`).
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
    series: Mapping[str, Sequence[Series]] | None = None,
) -> Iterator[pa.Table]:
    """Yield the sites' streamflow from ``start`` up to (not including) ``end``, UTC.

    As v1 timeseries rows in m3/s (:func:`to_rows`), one batch of ``batch_sites`` sites at a time.

    ``service``: ``iv`` instantaneous values (default; on the hour with ``hourly``), ``dv``
    daily means (at midnight UTC of their day). A batch the API has nothing for is empty.
    ``series``: the ranking of sites' several time series (:func:`choose_series`, which is
    asked for each batch's sites when it's None; pass it to ask once).
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
    series: Mapping[str, Sequence[Series]] | None = None,
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
