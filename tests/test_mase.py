"""MASE: stored in-sample naive scales, the naive_scale field and the mase metric."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any

import pyarrow.parquet as pq
import pytest

from teehr_core import metrics
from tests.engine_support import HYDRO, SPECS, public_spec


def _expected_scales() -> dict[tuple[str, int], float]:
    obs = pq.read_table(HYDRO / "usgs_streamflow.parquet").to_pylist()
    by: dict[str, dict[Any, float]] = defaultdict(dict)
    for r in obs:
        by[r["series_key"]][r["value_time"]] = r["value"]
    out = {}
    for key, values in by.items():
        for lag in (3600, 86400):
            diffs = [
                abs(v - values[t - timedelta(seconds=lag)])
                for t, v in values.items()
                if t - timedelta(seconds=lag) in values
            ]
            if diffs:
                out[(key, lag)] = sum(diffs) / len(diffs)
    return out


def _spec(lag: str = "P1D", **metric: Any) -> dict[str, Any]:
    spec = public_spec(SPECS / "hydro_by_location.yaml")
    spec["fields"] = [
        {
            "name": "scale",
            "fn": "naive_scale",
            "from": "public.usgs_streamflow_naive_scale",
            "lag": lag,
        }
    ]
    spec["aggregate"] = [
        {
            "group_by": ["primary_series_key", "secondary_dataset"],
            "metrics": [
                "mean_absolute_error",
                {"name": "mean_absolute_scaled_error", "scale": {"column": "scale"}, **metric},
            ],
        }
    ]
    spec["output"] = {"include_pairs": True}
    return spec


def test_standalone_mase() -> None:
    assert metrics.mase([1, 2, 3], [1.5, 2, 2], 0.5) == pytest.approx(1.0)
    assert metrics.mase([1, 2, 3], [1.5, 2, 2], [0.5, 1, 0]) == pytest.approx(0.5)  # 0 left out
    with pytest.raises(ValueError, match="scale"):
        metrics.evaluate([1, 2], [1, 2], ["mean_absolute_scaled_error"])
