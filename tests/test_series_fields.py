"""Series-aware fields: exceedance probabilities and events, against direct calculations."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any

import pytest
import yaml

from teehr_core import engine
from teehr_core.durations import parse_duration
from teehr_core.model.units import UnitRegistry
from tests.engine_support import SPECS

CATALOG = "tests/fixtures/hydro/datasets.yaml"


def _pairs(units: UnitRegistry, fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spec: dict[str, Any] = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())
    spec["fields"] = fields
    spec["output"] = {"include_pairs": True}
    res = engine.evaluate(spec, CATALOG, units)
    assert res.pairs is not None
    return res.pairs.to_pylist()


def _obs_series(pairs: list[dict[str, Any]]) -> dict[str, list[tuple[Any, float]]]:
    series: dict[str, dict[Any, float]] = defaultdict(dict)
    for r in pairs:
        series[r["primary_series_key"]][r["value_time"]] = r["primary_value"]
    return {k: sorted(v.items()) for k, v in series.items()}


def test_exceedance_obs(units: UnitRegistry) -> None:
    pairs = _pairs(units, [{"name": "p", "fn": "exceedance"}])
    for key, points in _obs_series(pairs).items():
        values = [v for _, v in points]
        n = len(values)
        want = {t: (1 + sum(x > v for x in values)) / (n + 1) for t, v in points}
        for r in pairs:
            if r["primary_series_key"] == key:
                assert r["p"] == pytest.approx(want[r["value_time"]])


def test_exceedance_pred_ranks_each_forecast(units: UnitRegistry) -> None:
    pairs = _pairs(units, [{"name": "p", "fn": "exceedance", "side": "secondary"}])
    traj: dict[tuple[Any, ...], dict[Any, float]] = defaultdict(dict)
    for r in pairs:
        traj[(r["secondary_dataset"], r["secondary_series_key"], r["reference_time"])][
            r["value_time"]
        ] = r["secondary_value"]
    for r in pairs:
        values = list(
            traj[(r["secondary_dataset"], r["secondary_series_key"], r["reference_time"])].values()
        )
        want = (1 + sum(x > r["secondary_value"] for x in values)) / (len(values) + 1)
        assert r["p"] == pytest.approx(want)


def _events(
    points: list[tuple[Any, float]],
    thr: float,
    gap: timedelta | None,
    min_dur: timedelta,
) -> list[tuple[Any, Any]]:
    runs: list[list[Any]] = []
    for t, v in points:
        if v >= thr:
            if runs and runs[-1][2]:
                runs[-1][1] = t
            else:
                runs.append([t, t, True])
        elif runs:
            runs[-1][2] = False
    merged: list[list[Any]] = []
    for start, stop, _ in runs:
        if merged and gap is not None and start - merged[-1][1] <= gap:
            merged[-1][1] = stop
        else:
            merged.append([start, stop])
    return [(a, b) for a, b in merged if b - a >= min_dur]


@pytest.mark.parametrize(
    ("threshold", "gap", "min_dur"),
    [
        ({"quantile": 0.8}, None, None),
        ({"quantile": 0.6}, "PT2H", "PT1H"),
    ],
)
def test_events_obs(
    units: UnitRegistry, threshold: Any, gap: str | None, min_dur: str | None
) -> None:
    field: dict[str, Any] = {"name": "ev", "fn": "events", "threshold": threshold}
    if gap:
        field["merge_gap"] = gap
    if min_dur:
        field["min_duration"] = min_dur
    pairs = _pairs(units, [field])
    import numpy as np

    for key, points in _obs_series(pairs).items():
        thr = float(np.quantile([v for _, v in points], threshold["quantile"]))
        events = _events(
            points,
            thr,
            parse_duration(gap) if gap else None,
            parse_duration(min_dur) if min_dur else timedelta(0),
        )
        assert events, key
        for r in pairs:
            if r["primary_series_key"] != key:
                continue
            inside = [s for s, e in events if s <= r["value_time"] <= e]
            if inside:
                assert r["ev"] == f"{key}@{inside[0]:%Y-%m-%dT%H:%M:%SZ}"
            else:
                assert r["ev"] is None


def test_metrics_per_event(units: UnitRegistry) -> None:
    spec: dict[str, Any] = yaml.safe_load((SPECS / "hydro_by_location.yaml").read_text())
    spec["fields"] = [{"name": "ev", "fn": "events", "threshold": {"quantile": 0.8}}]
    spec["filter"] = "ev IS NOT NULL"
    spec["aggregate"] = [
        {"group_by": ["ev", "secondary_dataset"], "metrics": ["count", "root_mean_square_error"]}
    ]
    rows = engine.evaluate(spec, CATALOG, units).table.to_pylist()
    assert rows and all(r["ev"] and r["count"] > 0 for r in rows)
