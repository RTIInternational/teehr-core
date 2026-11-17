"""Compare teehr's metric kernels with teehr-core's on the same inputs.

Run with teehr's Python (it imports ``teehr``), from the teehr-core checkout::

    /path/to/teehr/.venv/bin/python scripts/compare_teehr.py

Prints, per metric and case, where the two differ (beyond 1e-9 relative), including one
being undefined (NaN / None) and the other not. Shared metrics only; teehr's per-metric
defaults (no transform, no epsilon).
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from teehr.metrics.models import deterministic, signature
from teehr.metrics.models.base import MetricsBasemodel

from teehr_core import metrics as core
from teehr_core.metrics.registry import ALIASES, METRICS

SKIP = {"confusion_matrix"}  # a dict in teehr; four counts in teehr-core
THRESHOLD = 50.0


def cases() -> dict[str, tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    rng = np.random.default_rng(42)
    n = 800  # daily, a bit over two years
    t = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    flow = rng.gamma(2.0, 30.0, n)
    out = {
        "flows": (flow, flow * rng.normal(1.05, 0.2, n), t),
        "with_negatives": (rng.normal(0, 5, n), rng.normal(0.5, 6, n), t),
        "constant_obs": (np.full(n, 40.0), flow, t),
        "constant_pred": (flow, np.full(n, 40.0), t),
        "single_pair": (np.array([10.0]), np.array([12.0]), t[:1]),
        "two_pairs": (np.array([10.0, 20.0]), np.array([12.0, 18.0]), t[:2]),
        "obs_sum_zero": (np.array([-1.0, 1.0, -2.0, 2.0]), np.array([0.5, 1.5, -1.0, 2.5]), t[:4]),
        "zeros_in_obs": (np.array([0.0, 5.0, 0.0, 10.0]), np.array([1.0, 4.0, 0.5, 11.0]), t[:4]),
    }
    o, p = flow.copy(), flow * 1.1
    o[[3, 50, 51]] = np.nan
    p[[10, 51]] = np.nan
    out["with_nans"] = (o, p, t)
    return out


def teehr_metrics() -> dict[str, MetricsBasemodel]:
    found = {}
    for mod in (deterministic, signature):
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            name = getattr(cls, "default_output_field_name", None)
            if cls.__module__ == mod.__name__ and issubclass(cls, MetricsBasemodel) and name:
                found[name] = cls()
    return found


def teehr_value(model: MetricsBasemodel, o, p, t):
    fn = model.func(model)
    params = list(inspect.signature(fn).parameters)
    naive = pd.Series(t.tz_convert("UTC").tz_localize(None))  # as teehr's warehouse holds them
    args = {"p": pd.Series(o), "s": pd.Series(p), "value_time": naive,
            "threshold_series": pd.Series(np.full(len(o), THRESHOLD))}  # fmt: skip
    try:
        return fn(*(args[k] for k in params))
    except Exception as exc:  # record what teehr does
        return f"error: {type(exc).__name__}"


def core_value(name: str, o, p, t):
    metric = METRICS[name]
    kwargs = {}
    if "threshold" in metric.params:
        kwargs["threshold"] = THRESHOLD
    try:
        got = core.evaluate(o, p, [name], times=np.asarray(t.to_pydatetime()), **kwargs)
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"[:80]
    return next(iter(got.values()))


def undefined(v) -> bool:
    return (
        v is None
        or (isinstance(v, float) and math.isnan(v))
        or (isinstance(v, (np.floating,)) and np.isnan(v))
    )


def same(a, b) -> bool:
    if undefined(a) and undefined(b):
        return True
    if undefined(a) or undefined(b) or isinstance(a, str) or isinstance(b, str):
        return False
    if isinstance(a, (pd.Timestamp, np.datetime64)) or hasattr(b, "isoformat"):
        a, b = pd.Timestamp(a), pd.Timestamp(b)  # naive UTC (teehr) vs aware (core)
        return (a.tz_convert(None) if a.tz else a) == (b.tz_convert(None) if b.tz else b)
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)


def main() -> int:
    data = cases()
    teehr = teehr_metrics()
    shared = sorted(n for n in teehr if n not in SKIP and ALIASES.get(n, n) in METRICS)
    only_teehr = sorted(n for n in teehr if n not in SKIP and ALIASES.get(n, n) not in METRICS)
    differ = 0
    for name in shared:
        rows = []
        for case, (o, p, t) in data.items():
            a, b = teehr_value(teehr[name], o, p, t), core_value(ALIASES.get(name, name), o, p, t)
            if not same(a, b):
                rows.append(f"    {case:15} teehr={a!r:30.30} core={b!r:.40}")
        if rows:
            differ += 1
            print(f"{name}:")
            print("\n".join(rows))
    print(f"\n{len(shared)} shared metrics, {differ} differ on at least one case")
    print(f"teehr only: {', '.join(only_teehr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
