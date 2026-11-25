"""NWM operational streamflow at reaches (``channel_rt``), from NOAA's public GCS bucket.

Rows are ``series_key`` (``<version>-<feature_id>``, e.g. ``nwm31-4512772``), ``value_time``
(UTC) and ``value`` in m3/s (:data:`UNIT`), plus, for a forecast, ``reference_time`` (the
cycle: the day and ``tHHz`` in the file's path) and, for an ensemble member, ``member``.

As the legacy ingest did:

- each cycle's files are listed per day under ``nwm.YYYYMMDD/<configuration>/``; files that
  are missing are skipped;
- **analysis** (``analysis_assim*``) keeps one value per ``value_time``, the latest cycle's
  (largest look-back), and has no ``reference_time``; a later run upserting over an earlier
  one keeps that rule across runs;
- missing values (the files' fill value) are dropped.

Values are the files' scaled integers (0.01 m3/s) as float64, exact where legacy's float32
was not.

    from teehr_core.fetch import nwm

    found = nwm.files("short_range", start, end)
    for batch in nwm.read_points(found, ["4512772", "8786017"], version="nwm31"):
        ...
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)

BUCKET = "national-water-model"
FILE_URL = f"https://storage.googleapis.com/{BUCKET}/{{name}}"
LIST_URL = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
#: The unit of ``value``.
UNIT = "m3/s"
#: The configurations legacy ingested (their directory names on GCS).
CONFIGURATIONS = (
    "short_range", "short_range_alaska", "short_range_hawaii", "short_range_puertorico",
    "medium_range_mem1", "medium_range_blend", "medium_range_alaska_mem1",
    "medium_range_blend_alaska",
    "analysis_assim_no_da", "analysis_assim_alaska_no_da", "analysis_assim_extend_no_da",
    "analysis_assim_extend_alaska_no_da", "analysis_assim_hawaii_no_da",
    "analysis_assim_puertorico_no_da",
)  # fmt: skip
#: NWM versions by their files' ``NWM_version_number``, and the cycle each began with.
VERSIONS = {"v3.0": "nwm30", "v3.1": "nwm31"}
STARTS = {"nwm30": datetime(2023, 9, 19, 12, tzinfo=UTC),
          "nwm31": datetime(2026, 8, 18, 0, tzinfo=UTC)}  # fmt: skip
#: Within this long after a version began, files may still say the previous one.
VERSION_GRACE = timedelta(days=1)
TRIES = 4

_NAME = re.compile(
    r"nwm\.(?P<day>\d{8})/(?P<configuration>[a-z_0-9]+)/nwm\.t(?P<hour>\d{2})z\."
    r"[a-z_]+\.channel_rt(?:_(?P<member>\d+))?\.(?P<step>[ft]m?)(?P<lead>\d{2,5})\."
    r"[a-z]+\.nc$"
)


@dataclass(frozen=True)
class File:
    """One NWM file: its GCS object name, cycle, and the time its values are for."""

    name: str
    reference_time: datetime
    value_time: datetime  # from the path; the file's own ``time`` is what's written
    member: str | None
    analysis: bool


def _lead(digits: str) -> timedelta:
    """``HH`` hours, or ``HHMM`` hours and minutes (Hawaii's 15-minute steps)."""
    if len(digits) <= 3:
        return timedelta(hours=int(digits))
    return timedelta(hours=int(digits[:-2]), minutes=int(digits[-2:]))


def parse(name: str) -> File | None:
    """Return a ``channel_rt`` file's times from its object name (None for other files)."""
    match = _NAME.search(name)
    if match is None:
        return None
    cycle = datetime.strptime(match["day"], "%Y%m%d").replace(tzinfo=UTC) + timedelta(
        hours=int(match["hour"])
    )
    analysis = match["step"] == "tm"
    lead = _lead(match["lead"])
    return File(name=name, reference_time=cycle,
                value_time=cycle - lead if analysis else cycle + lead,
                member=match["member"], analysis=analysis)  # fmt: skip


def _get(url: str) -> bytes:
    """GET a public URL, retrying transient failures (a missing object is not one)."""
    for attempt in range(TRIES):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                body: bytes = resp.read()
                return body
        except urllib.error.HTTPError as exc:
            if exc.code == 404 or attempt == TRIES - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == TRIES - 1:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _list(prefix: str) -> list[str]:
    """Every object name under ``prefix`` in the bucket."""
    names: list[str] = []
    token = None
    while True:
        query = {"prefix": prefix, "fields": "items(name),nextPageToken", "maxResults": "5000"}
        if token:
            query["pageToken"] = token
        page = json.loads(_get(f"{LIST_URL}?{urllib.parse.urlencode(query)}"))
        names.extend(item["name"] for item in page.get("items", []))
        token = page.get("nextPageToken")
        if not token:
            return names


def files(configuration: str, start: datetime, end: datetime) -> list[File]:
    """List the configuration's ``channel_rt`` files for cycles from ``start`` up to ``end``.

    Both are UTC; ``end`` is exclusive. For analysis, only the latest cycle's file for each
    ``value_time`` is kept (as legacy did), so superseded files are never read.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be time zone aware (UTC)")
    found: list[File] = []
    day = start.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    while day < end:
        for name in _list(f"nwm.{day:%Y%m%d}/{configuration}/"):
            parsed = parse(name)
            if parsed is not None and start <= parsed.reference_time < end:
                found.append(parsed)
        day += timedelta(days=1)
    if found and found[0].analysis:
        latest: dict[datetime, File] = {}
        for f in sorted(found, key=lambda f: (f.reference_time, f.value_time)):
            latest[f.value_time] = f
        found = list(latest.values())
    return sorted(found, key=lambda f: (f.reference_time, f.value_time, f.name))


def _check_version(f: File, said: str, version: str) -> None:
    """Refuse a file of another NWM version than ``version``.

    Except, as legacy allowed, the previous version within :data:`VERSION_GRACE` of
    ``version`` starting: configurations don't all switch on the same cycle.
    """
    if VERSIONS.get(said) == version:
        return
    started = STARTS.get(version)
    if said in VERSIONS and started and started <= f.reference_time < started + VERSION_GRACE:
        logger.warning("%s: NWM %s, within a day of %s starting", f.name, said, version)
        return
    raise ValueError(f"{f.name}: NWM {said or 'version unknown'}, not {version}")


def _read(f: File, wanted: np.ndarray, version: str) -> pa.Table | None:
    """One file's rows for the ``wanted`` feature ids (sorted); None if it has none or is gone."""
    import h5py

    try:
        body = _get(FILE_URL.format(name=f.name))
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        logger.warning("%s: gone (404), skipped", f.name)
        return None
    with h5py.File(io.BytesIO(body), "r") as h5:
        said = h5.attrs.get("NWM_version_number", b"")
        said = said.decode() if isinstance(said, bytes) else str(said)
        _check_version(f, said, version)
        ids = h5["feature_id"][:]
        flow = h5["streamflow"]
        raw = flow[:]
        scale = float(np.asarray(flow.attrs.get("scale_factor", [1.0]))[0])
        offset = float(np.asarray(flow.attrs.get("add_offset", [0.0]))[0])
        fill = np.asarray(flow.attrs.get("_FillValue", [np.iinfo(raw.dtype).min]))[0]
        minutes = int(h5["time"][0])
    at = np.searchsorted(ids, wanted)
    at = np.clip(at, 0, len(ids) - 1)
    hit = ids[at] == wanted
    at, keys = at[hit], wanted[hit]
    values = raw[at]
    keep = values != fill
    keys, values = keys[keep], values[keep]
    if not len(keys):
        return None
    value_time = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(minutes=minutes)
    n = len(keys)
    columns: dict[str, pa.Array] = {
        "series_key": pa.array([f"{version}-{k}" for k in keys.tolist()], pa.string()),
    }
    if not f.analysis:
        columns["reference_time"] = pa.array([f.reference_time] * n, pa.timestamp("us", "UTC"))
    columns["value_time"] = pa.array([value_time] * n, pa.timestamp("us", "UTC"))
    if f.member is not None:
        columns["member"] = pa.array([f.member] * n, pa.string())
    # The scale is a float32 (0.01 is 0.0099999998): round to its decimals for exact values.
    digits = max(0, round(-np.log10(scale))) if 0 < scale < 1 else 0
    columns["value"] = pa.array(np.round(values * scale + offset, digits), pa.float64())
    return pa.table(columns)


def reach_id(key: str | int) -> int:
    """``nwm31-4512772``, ``4512772`` or 4512772 -> 4512772."""
    return int(str(key).rsplit("-", 1)[-1])


def read_points(
    found: Iterable[File],
    reaches: Iterable[str | int],
    version: str = "nwm31",
    workers: int = 16,
    batch_files: int = 100,
) -> Iterator[pa.Table]:
    """Read the reaches' values from the files, ``batch_files`` files per table.

    ``reaches`` are feature ids (``4512772``) or series keys (``nwm31-4512772``). Files are
    fetched ``workers`` at a time; a file that is gone (404) is skipped with a warning.
    Tables come in the files' order (cycle, then time); a batch's files are held until the
    batch is done.
    """
    ids = np.unique(np.array([reach_id(r) for r in reaches], dtype=np.int64))
    ordered = list(found)
    if not len(ids) or not ordered:
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for begin in range(0, len(ordered), batch_files):
            chunk = ordered[begin : begin + batch_files]
            tables = [t for t in pool.map(lambda f: _read(f, ids, version), chunk) if t is not None]
            if tables:
                yield pa.concat_tables(tables)
