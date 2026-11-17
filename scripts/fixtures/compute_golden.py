"""Compute golden metric values used to verify the teehr-core metrics.

Ported from VeriRace (``scripts/fixtures/compute_golden.py`` at 668ab7d).

Two kinds of reference values, written to tests/golden/:

1. ``metric_vectors.json``: small hand-made cases computed with exact arithmetic
   (``fractions.Fraction`` and 50-digit ``decimal`` square roots), so they don't
   depend on NumPy or any metric library. Includes edge cases that define the
   missing-data policy (undefined results are ``null``).
2. ``<fixture>_metrics.json``: metrics for each fixture evaluation. Pairing and
   grouping are done here with plain pandas, independent of the engine, and every
   metric is computed by at least two independent implementations that must agree
   before a value is written.

Regenerate deliberately, never to make a failing test pass, and record why each
changed value changed in CHANGELOG.md.

Usage:
    uv run --group fixtures python scripts/fixtures/compute_golden.py
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal, getcontext
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path
from typing import Any

import HydroErr as hr
import hydroeval as he
import numpy as np
import pandas as pd
import scipy.stats as st
import scoringrules as sr
import yaml

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
GOLDEN = ROOT / "tests" / "golden"

# Exact: 1 ft = 0.3048 m, so 1 ft3 = 0.028316846592 m3.
M3S_TO_CFS = 1 / 0.028316846592
UNIT_FACTORS = {("m3/s", "[cft_i]/s"): M3S_TO_CFS, ("MW", "MW"): 1.0}

AGREEMENT_RTOL = 1e-12
LEAD_BIN_HOURS = 6

_NOT_CONSTANT = "spread within 16 eps counts as constant"
METRIC_DEFINITIONS = {
    "count": "number of pairs with finite obs and sim",
    "mean_error": "mean(sim - obs)",
    "mean_absolute_error": "mean(|sim - obs|)",
    "mean_square_error": "mean((sim - obs)^2)",
    "root_mean_square_error": "sqrt(mean((sim - obs)^2))",
    "max_value_delta": "max(sim) - max(obs)",
    "relative_bias": "sum(sim - obs) / sum(obs); null if sum(obs) == 0",
    "mean_absolute_relative_error": "sum(|sim - obs|) / sum(obs); null if sum(obs) == 0",
    "multiplicative_bias": "mean(sim) / mean(obs); null if mean(obs) == 0",
    "relative_median": "median(sim) / median(obs) of quantile-sketch values (each value "
    "snapped to its 0.1% bucket's representative); null if median(obs) == 0",
    "relative_minimum": "min(sim) / min(obs); null if min(obs) == 0",
    "relative_maximum": "max(sim) / max(obs); null if max(obs) == 0",
    "relative_standard_deviation": "std(sim) / std(obs), population std; null if obs is constant",
    "mean_absolute_percentage_error": "100 * mean(|sim - obs| / |obs|) over pairs with obs != 0; null if none",
    "symmetric_mean_absolute_percentage_error": "100 * mean(2|sim - obs| / (|obs| + |sim|)); pairs with both 0 count as 0",
    "normalized_mean_absolute_error": "mean(|sim - obs|) / mean(obs); null if mean(obs) == 0",
    "normalized_root_mean_square_error": "rmse / mean(obs); null if mean(obs) == 0",
    "pearson_correlation": f"Pearson correlation; null if obs or sim is constant ({_NOT_CONSTANT})",
    "spearman_correlation": "Spearman correlation of average ranks; null if obs or sim is constant",
    "r_squared": "pearson_r ** 2; null when pearson_r is",
    "coefficient_of_variation_ratio": (
        "(std(sim)/mean(sim)) / (std(obs)/mean(obs)); null if obs is constant or a mean is 0"
    ),
    "root_mean_standard_deviation_ratio": "rmse / std(obs); null if obs is constant",
    "nash_sutcliffe_efficiency": (
        "1 - sum((sim - obs)^2) / sum((obs - mean(obs))^2); "
        f"null if obs is constant ({_NOT_CONSTANT})"
    ),
    "nash_sutcliffe_efficiency_normalized": "1 / (2 - nse); null when nse is",
    "kling_gupta_efficiency": (
        "KGE 2009 (Gupta et al.): 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2), "
        "alpha = std(sim)/std(obs) (population std), beta = mean(sim)/mean(obs); "
        "null if obs or sim is constant or mean(obs) == 0"
    ),
    "kling_gupta_efficiency_mod1": (
        "KGE' (Kling et al. 2012): as kge with alpha = cv(sim)/cv(obs), cv = std/mean; "
        "null if obs or sim is constant or a mean is 0"
    ),
    "kling_gupta_efficiency_mod2": (
        "KGE'' (Tang et al. 2021): 1 - sqrt((r-1)^2 + (alpha-1)^2 + beta^2), "
        "alpha = std(sim)/std(obs), beta = (mean(sim) - mean(obs))/std(obs); "
        "null if obs or sim is constant"
    ),
}
METRICS = list(METRIC_DEFINITIONS)

# ---------------------------------------------------------------------------
# 1. Exact hand-computed vectors
# ---------------------------------------------------------------------------

getcontext().prec = 50


def _sqrt(x: Fraction) -> Decimal:
    return (Decimal(x.numerator) / Decimal(x.denominator)).sqrt()


def _dec(x: Fraction) -> Decimal:
    return Decimal(x.numerator) / Decimal(x.denominator)


#: Quantile sketches (teehr_core.metrics.sketch): values in buckets of relative width ALPHA.
SKETCH_ALPHA = Fraction(1, 1000)
_GAMMA = (1 + SKETCH_ALPHA) / (1 - SKETCH_ALPHA)


def _snap_exact(x: Any) -> Fraction:
    """A value's quantile-sketch bucket representative, 2 g^k / (g + 1), k = ceil(log_g |x|).

    In 50-digit decimals, independently of the engine's floating-point version.
    """
    x = Fraction(x)
    if abs(x) < Fraction(1, 10**12):
        return Fraction(0)
    dec = lambda f: Decimal(f.numerator) / Decimal(f.denominator)  # noqa: E731
    k = math.ceil(dec(abs(x)).ln() / dec(_GAMMA).ln())
    value = 2 * _GAMMA**k / (_GAMMA + 1)
    return value if x > 0 else -value


def _snap_np(x: np.ndarray) -> np.ndarray:
    """The same in floating point (NumPy), as a second implementation."""
    gamma = 1001 / 999
    a = np.abs(x)
    k = np.ceil(np.log(np.where(a < 1e-12, 1.0, a)) / np.log(gamma))
    return np.where(a < 1e-12, 0.0, np.sign(x) * 2 * gamma**k / (gamma + 1))


def _median(xs: list[Fraction]) -> Fraction:
    v = sorted(xs)
    mid = len(v) // 2
    return v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2


def _ranks(xs: list[Fraction]) -> list[Fraction]:
    """Average ranks (1-based); tied values share the mean of their positions."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [Fraction(0)] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = Fraction(i + j + 2, 2)
        i = j + 1
    return ranks


def _corr(a: list[Fraction], b: list[Fraction]) -> Decimal:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return _dec(cov) / (_sqrt(va) * _sqrt(vb))


def _ratio(num: Fraction | Decimal, den: Fraction | Decimal) -> float | None:
    return None if den == 0 else float(num / den)


def exact_metrics(obs_in: Sequence[float], sim_in: Sequence[float]) -> dict[str, float | None]:
    obs = [Fraction(v) for v in obs_in]
    sim = [Fraction(v) for v in sim_in]
    n = len(obs)
    err = [s - o for s, o in zip(sim, obs, strict=True)]
    mo, ms = sum(obs) / n, sum(sim) / n
    var_o = sum((o - mo) ** 2 for o in obs) / n
    var_s = sum((s - ms) ** 2 for s in sim) / n
    sse = sum(e**2 for e in err)
    sae = sum(abs(e) for e in err)
    if n < 2:
        var_o = var_s = Fraction(0)  # variance-based metrics are undefined for one pair
    sd_o, sd_s = _sqrt(var_o), _sqrt(var_s)
    rmse = _sqrt(sse / n)
    nonzero = [abs(e) / abs(o) for e, o in zip(err, obs, strict=True) if o != 0]
    sym = [
        2 * abs(e) / (abs(o) + abs(s)) if abs(o) + abs(s) != 0 else Fraction(0)
        for e, o, s in zip(err, obs, sim, strict=True)
    ]
    nse = 1 - sse / (n * var_o) if var_o != 0 else None

    out: dict[str, float | None] = {
        "count": n,
        "mean_error": float(sum(err) / n),
        "mean_absolute_error": float(sae / n),
        "mean_square_error": float(sse / n),
        "root_mean_square_error": float(rmse),
        "max_value_delta": float(max(sim) - max(obs)),
        "relative_bias": _ratio(sum(err), sum(obs)),
        "mean_absolute_relative_error": _ratio(sae, sum(obs)),
        "multiplicative_bias": _ratio(ms, mo),
        "relative_median": _ratio(
            _median([_snap_exact(v) for v in sim]), _median([_snap_exact(v) for v in obs])
        ),
        "relative_minimum": _ratio(min(sim), min(obs)),
        "relative_maximum": _ratio(max(sim), max(obs)),
        "relative_standard_deviation": _ratio(sd_s, sd_o) if var_o != 0 else None,
        "mean_absolute_percentage_error": float(100 * sum(nonzero) / len(nonzero))
        if nonzero
        else None,
        "symmetric_mean_absolute_percentage_error": float(100 * sum(sym) / n),
        "normalized_mean_absolute_error": _ratio(sae / n, mo),
        "normalized_root_mean_square_error": _ratio(rmse, _dec(mo)),
        "root_mean_standard_deviation_ratio": _ratio(rmse, sd_o) if var_o != 0 else None,
        "nash_sutcliffe_efficiency": float(nse) if nse is not None else None,
        "nash_sutcliffe_efficiency_normalized": float(1 / (2 - nse)) if nse is not None else None,
        "coefficient_of_variation_ratio": None,
        "pearson_correlation": None,
        "spearman_correlation": None,
        "r_squared": None,
        "kling_gupta_efficiency": None,
        "kling_gupta_efficiency_mod1": None,
        "kling_gupta_efficiency_mod2": None,
    }
    if var_o != 0 and mo != 0 and ms != 0:
        out["coefficient_of_variation_ratio"] = float((sd_s / _dec(ms)) / (sd_o / _dec(mo)))
    if var_o == 0 or var_s == 0:
        return out
    r = _corr(obs, sim)
    out["pearson_correlation"] = float(r)
    out["r_squared"] = float(r * r)
    out["spearman_correlation"] = float(_corr(_ranks(obs), _ranks(sim)))
    alpha = sd_s / sd_o

    def kge(a: Decimal, bias_term: Decimal) -> float:
        return float(1 - ((r - 1) ** 2 + (a - 1) ** 2 + bias_term).sqrt())

    out["kling_gupta_efficiency_mod2"] = kge(alpha, (_dec(ms - mo) / sd_o) ** 2)
    if mo != 0:
        beta = _dec(ms / mo)
        out["kling_gupta_efficiency"] = kge(alpha, (beta - 1) ** 2)
        if ms != 0:
            gamma = (sd_s / _dec(ms)) / (sd_o / _dec(mo))
            out["kling_gupta_efficiency_mod1"] = kge(gamma, (beta - 1) ** 2)
    return out


VECTOR_CASES = [
    ("simple", [1, 2, 3, 4, 5], [1.5, 1.5, 3.5, 3, 6], "small mixed over/under prediction"),
    ("perfect", [2, 4, 6, 8], [2, 4, 6, 8], "sim == obs: NSE = KGE = r = 1, errors 0"),
    ("constant_bias", [10, 20, 30, 40], [12, 22, 32, 42], "sim = obs + 2: r = 1, alpha = 1"),
    ("scaled", [1, 2, 3, 4], [2, 4, 6, 8], "sim = 2 * obs: r = 1, alpha = beta = 2"),
    ("anticorrelated", [1, 2, 3, 4], [4, 3, 2, 1], "r = -1, NSE strongly negative"),
    ("negative_values", [-2, -1, 0, 1, 2], [-1.5, -1, 0.5, 0.5, 2.5], "obs sums to 0"),
    ("constant_obs", [5, 5, 5], [4, 5, 6], "obs has zero variance: NSE, r, KGE undefined"),
    ("constant_sim", [1, 2, 3], [2, 2, 2], "sim has zero variance: r and KGE undefined"),
    ("single_pair", [3], [4], "n = 1: variance-based metrics undefined"),
    (
        "ties",
        [1, 2, 2, 3, 3, 3],
        [1, 1, 2, 2, 3, 4],
        "tied values: Spearman uses average ranks; even count: median of the middle two",
    ),
    (
        "zeros",
        [0, 1, 2, 0],
        [0, 2, 1, 1],
        "zero obs: MAPE uses the non-zero pairs; a pair with both 0 is 0 error in sMAPE",
    ),
]


#: Vector times are hourly from here, so a water year (October) starts after two pairs.
VECTOR_START = datetime(2025, 9, 30, 22, tzinfo=UTC)


def build_vectors() -> dict[str, object]:
    cases = []
    for name, obs, sim, note in VECTOR_CASES:
        times = [VECTOR_START + timedelta(hours=i) for i in range(len(obs))]
        cases.append(
            {
                "name": name,
                "note": note,
                "obs": obs,
                "sim": sim,
                "times": [t.isoformat() for t in times],
                "expected": {**exact_metrics(obs, sim), **exact_variants(obs, sim, times)},
            }
        )
    return {
        "description": "Exact-arithmetic reference values (no NumPy or metric libraries).",
        "metric_definitions": {**METRIC_DEFINITIONS, **VARIANT_DEFINITIONS},
        "metric_specs": [*METRICS, *VARIANT_SPECS],
        "cases": cases,
        "probabilistic": build_probabilistic(),
    }


# ---------------------------------------------------------------------------
# 2. Fixture evaluations, cross-checked between independent implementations
# ---------------------------------------------------------------------------

Impl = Callable[[np.ndarray, np.ndarray], float]


def _numpy_kge(o: np.ndarray, s: np.ndarray) -> float:
    r = np.corrcoef(o, s)[0, 1]
    return float(
        1 - math.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)
    )


def _fsum_mean(x: np.ndarray) -> float:
    return math.fsum(x.tolist()) / x.size


def _py_kge_2021(o: np.ndarray, s: np.ndarray) -> float:
    r = st.pearsonr(o, s).statistic
    so, ss = statistics.pstdev(o.tolist()), statistics.pstdev(s.tolist())
    beta = (statistics.fmean(s.tolist()) - statistics.fmean(o.tolist())) / so
    return float(1 - math.sqrt((r - 1) ** 2 + (ss / so - 1) ** 2 + beta**2))


def _np_kge_2021(o: np.ndarray, s: np.ndarray) -> float:
    r = np.corrcoef(o, s)[0, 1]
    beta = (s.mean() - o.mean()) / o.std()
    return float(1 - math.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + beta**2))


def _cv_ratio(o: np.ndarray, s: np.ndarray) -> float:
    return float((s.std() / s.mean()) / (o.std() / o.mean()))


def _py_cv_ratio(o: np.ndarray, s: np.ndarray) -> float:
    so, ss = statistics.pstdev(o.tolist()), statistics.pstdev(s.tolist())
    return (ss / statistics.fmean(s.tolist())) / (so / statistics.fmean(o.tolist()))


def _py_mape(o: np.ndarray, s: np.ndarray) -> float:
    terms = [abs(b - a) / abs(a) for a, b in zip(o.tolist(), s.tolist(), strict=True) if a != 0]
    return 100 * math.fsum(terms) / len(terms)


def _py_smape(o: np.ndarray, s: np.ndarray) -> float:
    terms = [
        2 * abs(b - a) / (abs(a) + abs(b)) if abs(a) + abs(b) else 0.0
        for a, b in zip(o.tolist(), s.tolist(), strict=True)
    ]
    return 100 * math.fsum(terms) / len(terms)


IMPLEMENTATIONS: dict[str, dict[str, Impl]] = {
    "mean_error": {
        "HydroErr.me": lambda o, s: float(hr.me(s, o)),
        "numpy": lambda o, s: float(np.mean(s - o)),
    },
    "mean_absolute_error": {
        "HydroErr.mae": lambda o, s: float(hr.mae(s, o)),
        "fsum": lambda o, s: _fsum_mean(np.abs(s - o)),
    },
    "mean_square_error": {
        "HydroErr.mse": lambda o, s: float(hr.mse(s, o)),
        "fsum": lambda o, s: _fsum_mean((s - o) ** 2),
    },
    "root_mean_square_error": {
        "hydroeval.rmse": lambda o, s: float(he.evaluator(he.rmse, s, o)[0]),
        "HydroErr.rmse": lambda o, s: float(hr.rmse(s, o)),
    },
    "max_value_delta": {
        "numpy": lambda o, s: float(s.max() - o.max()),
        "builtins": lambda o, s: float(max(s.tolist()) - max(o.tolist())),
    },
    "relative_bias": {
        "hydroeval.pbias": lambda o, s: float(-he.evaluator(he.pbias, s, o)[0] / 100),
        "numpy": lambda o, s: float(np.sum(s - o) / np.sum(o)),
    },
    "mean_absolute_relative_error": {
        "numpy": lambda o, s: float(np.sum(np.abs(s - o)) / np.sum(o)),
        "fsum": lambda o, s: math.fsum(np.abs(s - o).tolist()) / math.fsum(o.tolist()),
    },
    "multiplicative_bias": {
        "numpy": lambda o, s: float(s.mean() / o.mean()),
        "statistics": lambda o, s: statistics.fmean(s.tolist()) / statistics.fmean(o.tolist()),
    },
    "relative_median": {
        "numpy": lambda o, s: float(np.median(_snap_np(s)) / np.median(_snap_np(o))),
        "statistics": lambda o, s: float(
            statistics.median([_snap_exact(v) for v in s.tolist()])
            / statistics.median([_snap_exact(v) for v in o.tolist()])
        ),
    },
    "relative_minimum": {
        "numpy": lambda o, s: float(s.min() / o.min()),
        "builtins": lambda o, s: min(s.tolist()) / min(o.tolist()),
    },
    "relative_maximum": {
        "numpy": lambda o, s: float(s.max() / o.max()),
        "builtins": lambda o, s: max(s.tolist()) / max(o.tolist()),
    },
    "relative_standard_deviation": {
        "numpy": lambda o, s: float(s.std() / o.std()),
        "statistics": lambda o, s: statistics.pstdev(s.tolist()) / statistics.pstdev(o.tolist()),
    },
    "mean_absolute_percentage_error": {
        "numpy": lambda o, s: float(100 * np.mean(np.abs(s - o)[o != 0] / np.abs(o[o != 0]))),
        "fsum": _py_mape,
    },
    "symmetric_mean_absolute_percentage_error": {
        "HydroErr.smape2": lambda o, s: float(hr.smape2(s, o)),
        "fsum": _py_smape,
    },
    "normalized_mean_absolute_error": {
        "numpy": lambda o, s: float(np.mean(np.abs(s - o)) / o.mean()),
        "fsum": lambda o, s: _fsum_mean(np.abs(s - o)) / _fsum_mean(o),
    },
    "normalized_root_mean_square_error": {
        "HydroErr.nrmse_mean": lambda o, s: float(hr.nrmse_mean(s, o)),
        "numpy": lambda o, s: float(np.sqrt(np.mean((s - o) ** 2)) / o.mean()),
    },
    "pearson_correlation": {
        "scipy.stats.pearsonr": lambda o, s: float(st.pearsonr(o, s).statistic),
        "HydroErr.pearson_r": lambda o, s: float(hr.pearson_r(s, o)),
    },
    "spearman_correlation": {
        "scipy.stats.spearmanr": lambda o, s: float(st.spearmanr(o, s).statistic),
        "pandas": lambda o, s: float(pd.Series(o).corr(pd.Series(s), method="spearman")),
    },
    "r_squared": {
        "HydroErr.r_squared": lambda o, s: float(hr.r_squared(s, o)),
        "scipy": lambda o, s: float(st.pearsonr(o, s).statistic ** 2),
    },
    "coefficient_of_variation_ratio": {"numpy": _cv_ratio, "statistics": _py_cv_ratio},
    "root_mean_standard_deviation_ratio": {
        "numpy": lambda o, s: float(np.sqrt(np.mean((s - o) ** 2)) / o.std()),
        "statistics": lambda o, s: (
            math.sqrt(_fsum_mean((s - o) ** 2)) / statistics.pstdev(o.tolist())
        ),
    },
    "nash_sutcliffe_efficiency": {
        "hydroeval.nse": lambda o, s: float(he.evaluator(he.nse, s, o)[0]),
        "HydroErr.nse": lambda o, s: float(hr.nse(s, o)),
    },
    "nash_sutcliffe_efficiency_normalized": {
        "hydroeval.nse": lambda o, s: float(1 / (2 - he.evaluator(he.nse, s, o)[0])),
        "HydroErr.nse": lambda o, s: float(1 / (2 - hr.nse(s, o))),
    },
    "kling_gupta_efficiency": {
        "hydroeval.kge": lambda o, s: float(he.evaluator(he.kge, s, o)[0][0]),
        "HydroErr.kge_2009": lambda o, s: float(hr.kge_2009(s, o)),
        "numpy": _numpy_kge,
    },
    "kling_gupta_efficiency_mod1": {
        "hydroeval.kgeprime": lambda o, s: float(he.evaluator(he.kgeprime, s, o)[0][0]),
        "HydroErr.kge_2012": lambda o, s: float(hr.kge_2012(s, o)),
    },
    "kling_gupta_efficiency_mod2": {"numpy": _np_kge_2021, "statistics": _py_kge_2021},
}


def varies(x: np.ndarray) -> bool:
    """Spread larger than floating-point noise (16 eps relative), as in the engine."""
    scale = max(abs(x.max()), abs(x.min()))
    return bool(x.max() - x.min() > 16 * np.finfo(np.float64).eps * scale)


def is_defined(metric: str, obs: np.ndarray, sim: np.ndarray) -> bool:
    """Missing-data policy: when a metric's value is mathematically undefined."""
    obs_varies = obs.size >= 2 and varies(obs)
    sim_varies = sim.size >= 2 and varies(sim)
    both = obs_varies and sim_varies
    return {
        "relative_bias": obs.sum() != 0,
        "mean_absolute_relative_error": obs.sum() != 0,
        "multiplicative_bias": obs.mean() != 0,
        "relative_median": np.median(_snap_np(obs)) != 0,
        "relative_minimum": obs.min() != 0,
        "relative_maximum": obs.max() != 0,
        "relative_standard_deviation": obs_varies,
        "mean_absolute_percentage_error": bool(np.any(obs != 0)),
        "normalized_mean_absolute_error": obs.mean() != 0,
        "normalized_root_mean_square_error": obs.mean() != 0,
        "root_mean_standard_deviation_ratio": obs_varies,
        "nash_sutcliffe_efficiency": obs_varies,
        "nash_sutcliffe_efficiency_normalized": obs_varies,
        "coefficient_of_variation_ratio": obs_varies and obs.mean() != 0 and sim.mean() != 0,
        "pearson_correlation": both,
        "spearman_correlation": both,
        "r_squared": both,
        "kling_gupta_efficiency": both and obs.mean() != 0,
        "kling_gupta_efficiency_mod1": both and obs.mean() != 0 and sim.mean() != 0,
        "kling_gupta_efficiency_mod2": both,
    }.get(metric, True)


def checked_metrics(obs: np.ndarray, sim: np.ndarray) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {"count": int(obs.size)}
    for metric, impls in IMPLEMENTATIONS.items():
        if not is_defined(metric, obs, sim):
            out[metric] = None
            continue
        values = {name: fn(obs, sim) for name, fn in impls.items()}
        ref = next(iter(values.values()))
        for name, v in values.items():
            if not math.isclose(v, ref, rel_tol=AGREEMENT_RTOL, abs_tol=1e-15):
                raise AssertionError(f"{metric}: {name}={v!r} disagrees with {ref!r}")
        out[metric] = ref
    # The exact-arithmetic path must agree too, including which values are null.
    exact = exact_metrics(obs.tolist(), sim.tolist())
    for metric, v in out.items():
        e = exact[metric]
        if (v is None) != (e is None) or (
            v is not None and not math.isclose(v, e, rel_tol=1e-9, abs_tol=1e-9)  # type: ignore[arg-type]
        ):
            raise AssertionError(f"{metric}: library value {v!r} disagrees with exact {e!r}")
    return out


# ---------------------------------------------------------------------------
# 3. Parameterized metrics (thresholds, normalizers, sides, times)
# ---------------------------------------------------------------------------
# Each variant is an output column with the spec entry that produces it. Values are
# computed exactly (fractions) and, for fixtures, by two independent implementations.

CONTINGENCY = (
    "hits",
    "misses",
    "false_alarms",
    "correct_negatives",
    "probability_of_detection",
    "probability_of_false_detection",
    "false_alarm_ratio",
    "critical_success_index",
    "success_ratio",
    "frequency_bias_index",
    "proportion_correct",
)
SIGNATURES = ("minimum", "maximum", "average", "sum", "variance", "flow_duration_curve_slope")
EXCEEDANCE = (Fraction("0.33"), Fraction("0.66"))
Q75 = Fraction(3, 4)

VARIANT_SPECS: list[dict[str, object]] = [
    *({"name": m, "threshold": {"primary_quantile": 0.75}, "as": f"{m}_q75"} for m in CONTINGENCY),
    {"name": "normalized_mean_absolute_error", "normalize_by": "range_primary", "as": "nmae_range"},
    {
        "name": "normalized_root_mean_square_error",
        "normalize_by": "range_primary",
        "as": "nrmse_range",
    },
    *(
        {"name": m, "side": side}
        for m in (*SIGNATURES, "max_value_time")
        for side in ("primary", "secondary")
    ),
    {"name": "mean_crps_ensemble"},
    {"name": "mean_crps_ensemble", "estimator": "fair", "as": "crps_fair"},
    {
        "name": "mean_brier_score",
        "threshold": {"primary_quantile": 0.75},
        "as": "mean_brier_score_q75",
    },
    {"name": "max_value_time_delta"},
    {"name": "annual_peak_flow_bias"},
    {
        "name": "annual_peak_flow_bias",
        "year_start_month": 10,
        "as": "annual_peak_flow_bias_wy",
    },
]

VARIANT_DEFINITIONS = {
    "mean_crps_ensemble": (
        "mean over cases of mean|x - y| - sum|x_i - x_j| / (2 m^2) (fair: / (2 m (m - 1))); "
        "one member: |x - y|"
    ),
    "mean_brier_score": "mean over cases of (fraction of members >= t - [obs >= t])^2",
    "mean_pinball_loss": "mean over rows of q (y - x) if y >= x else (1 - q)(x - y)",
    "interval_coverage": "fraction of cases with lo <= y <= hi (levels default 0.05, 0.95)",
    "mean_interval_width": "mean of hi - lo",
    "mean_interval_score": "mean of (hi - lo) + (2/alpha) max(lo - y, 0) + (2/alpha) max(y - hi, 0)",
    "mean_absolute_scaled_error": "mean over pairs with scale > 0 of |sim - obs| / scale (scale: a per-pair column)",
    "ensemble_spread": "sqrt(mean over cases of the members' variance, m - 1 denominator)",
    "ensemble_mean_root_mean_square_error": "sqrt(mean over cases of (mean of members - y)^2)",
    "spread_skill_ratio": "sqrt(mean over cases of (m + 1) / m x variance) / ensemble_mean_rmse",
    "rank_frequency": "fraction of cases whose rank (members < y + floor(members == y / 2)) is k",
    "rank_outlier_fraction": "fraction of cases with rank 0 or m",
    "rank_histogram_reliability": "sum over k = 0..m of |rank_frequency(k) - 1 / (m + 1)|; "
    "null for mixed ensemble sizes",
    "hits": "pairs with obs >= t and sim >= t",
    "misses": "pairs with obs >= t and sim < t",
    "false_alarms": "pairs with obs < t and sim >= t",
    "correct_negatives": "pairs with obs < t and sim < t",
    "probability_of_detection": "hits / (hits + misses); null if 0/0",
    "probability_of_false_detection": "false_alarms / (false_alarms + correct_negatives); null if 0/0",
    "false_alarm_ratio": "false_alarms / (hits + false_alarms); null if 0/0",
    "critical_success_index": "hits / (hits + misses + false_alarms); null if 0/0",
    "success_ratio": "hits / (hits + false_alarms); null if 0/0",
    "frequency_bias_index": "(hits + false_alarms) / (hits + misses); null if 0/0",
    "proportion_correct": "(hits + correct_negatives) / count; null if 0/0",
    "minimum": "min of one side",
    "maximum": "max of one side",
    "average": "mean of one side",
    "sum": "sum of one side",
    "variance": "population variance of one side",
    "flow_duration_curve_slope": "(ln Q(0.33) - ln Q(0.66)) / 0.33, Q(e) = quantile 1 - e (type 2) of "
    "quantile-sketch values (0.1% buckets); null if <= 0",
    "max_value_time": "earliest time of the maximum of one side",
    "max_value_time_delta": "seconds from earliest time of max(obs) to earliest time of max(sim)",
    "annual_peak_flow_bias": (
        "sum of (annual max(sim) - annual max(obs)) / sum of annual max(obs); years start in "
        "year_start_month (default 1)"
    ),
}


def variant_output(spec: dict[str, object]) -> str:
    if "as" in spec:
        return str(spec["as"])
    if "side" in spec:
        return f"{spec['name']}_{spec['side']}"
    return str(spec["name"])


def _q2(xs: Sequence[Any], q: Any) -> Any:
    """Quantile, Hyndman-Fan type 2 (numpy 'averaged_inverted_cdf'), exact for fractions."""
    v = sorted(xs)
    pos = q * len(v)
    k = math.ceil(pos)
    if k == pos and 0 < k < len(v):
        return (v[k - 1] + v[k]) / 2
    return v[max(k, 1) - 1]


def _year(t: datetime, start_month: int) -> int:
    return (t.year * 12 + t.month - 1 - (start_month - 1)) // 12


def _div(a: Any, b: Any) -> float | None:
    return None if b == 0 else float(Fraction(a) / Fraction(b))


def _table(obs: Sequence[Any], sim: Sequence[Any], thr: Any) -> dict[str, int]:
    t = {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0}
    for o, s in zip(obs, sim, strict=True):
        key = {
            (True, True): "hits",
            (True, False): "misses",
            (False, True): "false_alarms",
            (False, False): "correct_negatives",
        }[(o >= thr, s >= thr)]
        t[key] += 1
    return t


def _scores(t: dict[str, int]) -> dict[str, float | int | None]:
    h, m, f, c = t["hits"], t["misses"], t["false_alarms"], t["correct_negatives"]
    return {
        **t,
        "probability_of_detection": _div(h, h + m),
        "probability_of_false_detection": _div(f, f + c),
        "false_alarm_ratio": _div(f, h + f),
        "critical_success_index": _div(h, h + m + f),
        "success_ratio": _div(h, h + f),
        "frequency_bias_index": _div(h + f, h + m),
        "proportion_correct": _div(h + c, h + m + f + c),
    }


def _fdc(values: Sequence[Any]) -> float | None:
    snapped = [_snap_exact(v) for v in values]  # quantile-sketch values
    lo, hi = (_q2(snapped, 1 - e) for e in EXCEEDANCE)
    if lo <= 0 or hi <= 0:
        return None
    ln = lambda x: (Decimal(x.numerator) / Decimal(x.denominator)).ln()  # noqa: E731
    return float((ln(Fraction(lo)) - ln(Fraction(hi))) / _dec(EXCEEDANCE[1] - EXCEEDANCE[0]))


def _peak_time(values: Sequence[Any], times: Sequence[datetime]) -> datetime:
    peak = max(values)
    return min(t for v, t in zip(values, times, strict=True) if v == peak)


def _annual(
    obs: Sequence[Any], sim: Sequence[Any], times: Sequence[datetime], start: int
) -> float | None:
    peaks: dict[int, list[Any]] = {}
    for o, s, t in zip(obs, sim, times, strict=True):
        y = _year(t, start)
        po, ps = peaks.get(y, [o, s])
        peaks[y] = [max(po, o), max(ps, s)]
    num = sum(ps - po for po, ps in peaks.values())
    return _div(num, sum(po for po, _ in peaks.values()))


def exact_variants(
    obs_in: Sequence[float], sim_in: Sequence[float], times: Sequence[datetime]
) -> dict[str, object]:
    """Exact values of every variant (``VARIANT_SPECS``)."""
    obs = [Fraction(v) for v in obs_in]
    sim = [Fraction(v) for v in sim_in]
    n = len(obs)
    out: dict[str, object] = {}
    for name, value in _scores(_table(obs, sim, _q2(obs, Q75))).items():
        out[f"{name}_q75"] = value
    err = [abs(s - o) for o, s in zip(obs, sim, strict=True)]
    rng = max(obs) - min(obs)
    out["nmae_range"] = _div(sum(err) / n, rng)
    rmse = _sqrt(sum(e**2 for e in err) / n)
    out["nrmse_range"] = None if rng == 0 else float(rmse / _dec(Fraction(rng)))
    for side, vals in (("primary", obs), ("secondary", sim)):
        mean = sum(vals) / n
        out[f"minimum_{side}"] = float(min(vals))
        out[f"maximum_{side}"] = float(max(vals))
        out[f"average_{side}"] = float(mean)
        out[f"sum_{side}"] = float(sum(vals))
        out[f"variance_{side}"] = float(sum((v - mean) ** 2 for v in vals) / n)
        out[f"flow_duration_curve_slope_{side}"] = _fdc(vals)
        out[f"max_value_time_{side}"] = _peak_time(vals, times).isoformat()
    out["mean_crps_ensemble"] = out["crps_fair"] = float(sum(err) / n)  # one member per case
    q75 = _q2(obs, Q75)
    out["mean_brier_score_q75"] = float(
        sum(((s >= q75) - (o >= q75)) ** 2 for o, s in zip(obs, sim, strict=True)) / n
    )
    delta = _peak_time(sim, times) - _peak_time(obs, times)
    out["max_value_time_delta"] = delta.total_seconds()
    out["annual_peak_flow_bias"] = _annual(obs, sim, times, 1)
    out["annual_peak_flow_bias_wy"] = _annual(obs, sim, times, 10)
    return out


def library_variants(o: np.ndarray, s: np.ndarray, t: pd.Series) -> dict[str, object]:
    """Every variant by two independent implementations that must agree."""
    times = pd.to_datetime(t, utc=True).reset_index(drop=True)
    first: dict[str, object] = {}
    second: dict[str, object] = {}

    # numpy / pandas
    thr = float(np.quantile(o, 0.75, method="averaged_inverted_cdf"))
    oe, se = o >= thr, s >= thr
    t_np = {
        "hits": int(np.sum(oe & se)),
        "misses": int(np.sum(oe & ~se)),
        "false_alarms": int(np.sum(~oe & se)),
        "correct_negatives": int(np.sum(~oe & ~se)),
    }
    first.update({f"{k}_q75": v for k, v in _scores(t_np).items()})
    rng = o.max() - o.min()
    first["nmae_range"] = None if rng == 0 else float(np.mean(np.abs(s - o)) / rng)
    first["nrmse_range"] = None if rng == 0 else float(np.sqrt(np.mean((s - o) ** 2)) / rng)
    for side, v in (("primary", o), ("secondary", s)):
        first[f"minimum_{side}"] = float(v.min())
        first[f"maximum_{side}"] = float(v.max())
        first[f"average_{side}"] = float(v.mean())
        first[f"sum_{side}"] = float(v.sum())
        first[f"variance_{side}"] = float(v.var())
        lo, hi = (
            np.quantile(_snap_np(v), 1 - float(e), method="averaged_inverted_cdf")
            for e in EXCEEDANCE
        )
        first[f"flow_duration_curve_slope_{side}"] = (
            float((np.log(lo) - np.log(hi)) / 0.33) if lo > 0 and hi > 0 else None
        )
        first[f"max_value_time_{side}"] = times[v == v.max()].min().isoformat()
    first["mean_crps_ensemble"] = float(np.mean(sr.crps_ensemble(o, s[:, None], estimator="nrg")))
    first["crps_fair"] = float(np.mean(np.abs(s - o)))
    first["mean_brier_score_q75"] = float(
        np.mean(sr.brier_score(oe.astype(float), se.astype(float)))
    )
    first["max_value_time_delta"] = (
        times[s == s.max()].min() - times[o == o.max()].min()
    ).total_seconds()
    df = pd.DataFrame({"o": o, "s": s, "t": times})
    for start, key in ((1, "annual_peak_flow_bias"), (10, "annual_peak_flow_bias_wy")):
        year = (df.t.dt.year * 12 + df.t.dt.month - 1 - (start - 1)) // 12
        peaks = df.groupby(year)[["o", "s"]].max()
        den = peaks.o.sum()
        first[key] = None if den == 0 else float((peaks.s - peaks.o).sum() / den)

    # pure Python (the exact implementation on floats)
    py_times = [x.to_pydatetime() for x in times]
    second = exact_variants(o.tolist(), s.tolist(), py_times)

    for key, a in first.items():
        b = second[key]
        if isinstance(a, float) and isinstance(b, float):
            if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
                raise AssertionError(f"{key}: {a!r} disagrees with {b!r}")
        elif a != b:
            raise AssertionError(f"{key}: {a!r} disagrees with {b!r}")
    return first


# ---------------------------------------------------------------------------
# 4. Probabilistic forecasts (ensembles, quantiles)
# ---------------------------------------------------------------------------
# Long-format rows, one per member or quantile, as the engine pairs them. Exact values
# (fractions) must agree with scoringrules.

ENSEMBLE_ROWS = [
    # (case, obs, member value): four forecasts, 3-5 members each
    *((0, 3, x) for x in (1, 2, 5)),
    *((1, 10, x) for x in (8, 9, 11, 12)),
    *((2, 0, x) for x in (0, 0, 1, 2, 4)),
    *((3, 7, x) for x in (2, 3, 4)),
]
ENSEMBLE_THRESHOLD = 3
# Equal ensemble sizes (3 members), so the rank histogram's reliability is defined; ties
# and observations outside the ensemble cover every rank.
ENSEMBLE_EQUAL_ROWS = [
    *((0, 0, x) for x in (1, 2, 3)),  # rank 0 (below all)
    *((1, 5, x) for x in (1, 2, 3)),  # rank 3 (above all)
    *((2, 2, x) for x in (1, 3, 4)),  # rank 1
    *((3, 3, x) for x in (1, 2, 4)),  # rank 2
    *((4, 2, x) for x in (1, 2, 2)),  # tie: 1 below + floor(2 / 2) = rank 2
    *((5, 2, x) for x in (2, 3, 4)),  # tie: 0 below + floor(1 / 2) = rank 0
]
QUANTILE_LEVELS = (
    Fraction("0.05"),
    Fraction("0.25"),
    Fraction("0.5"),
    Fraction("0.75"),
    Fraction("0.95"),
)
QUANTILE_FORECASTS = [
    # (case, obs, values at QUANTILE_LEVELS)
    (0, 5, (2, 4, 5, 6, 8)),
    (1, 1, (2, 3, 4, 5, 7)),  # obs below the interval
    (2, 9, (1, 2, 3, 4, 6)),  # obs above the interval
    (3, Fraction("4.5"), (3, 4, Fraction("4.5"), 5, 6)),
]
ENSEMBLE_DIAGNOSTICS = [
    "ensemble_spread",
    "ensemble_mean_root_mean_square_error",
    "spread_skill_ratio",
    "rank_outlier_fraction",
    "rank_histogram_reliability",
    *({"name": "rank_frequency", "rank": k, "as": f"rank_{k}"} for k in range(4)),
]
PROBABILISTIC_SPECS = {
    "ensemble": [
        "count",
        "mean_crps_ensemble",
        {"name": "mean_crps_ensemble", "estimator": "fair", "as": "crps_fair"},
        {"name": "mean_brier_score", "threshold": {"value": ENSEMBLE_THRESHOLD, "unit": "m"}},
        *ENSEMBLE_DIAGNOSTICS,
    ],
    "ensemble_equal": ["count", "mean_crps_ensemble", *ENSEMBLE_DIAGNOSTICS],
    "quantile": [
        "count",
        "mean_pinball_loss",
        "interval_coverage",
        "mean_interval_width",
        "mean_interval_score",
        {"name": "interval_coverage", "interval": [0.25, 0.75], "as": "coverage_50"},
        {"name": "mean_interval_score", "interval": [0.25, 0.75], "as": "winkler_50"},
    ],
}


def _crps(y: Fraction, xs: list[Fraction], fair: bool) -> Fraction:
    m = len(xs)
    first = sum(abs(x - y) for x in xs) / m
    spread = sum(abs(a - b) for a in xs for b in xs)
    if m == 1:
        return first
    return first - spread / (2 * m * (m - 1) if fair else 2 * m * m)


def _ensemble_cases() -> dict[int, tuple[Fraction, list[Fraction]]]:
    cases: dict[int, tuple[Fraction, list[Fraction]]] = {}
    for c, y, x in ENSEMBLE_ROWS:
        cases.setdefault(c, (Fraction(y), []))[1].append(Fraction(x))
    return cases


def _ensemble_diagnostics(
    cases: dict[int, tuple[Fraction, list[Fraction]]],
) -> dict[str, float | None]:
    """Spread-skill and rank histogram metrics, exactly, from the definitions."""
    n = len(cases)
    var, sq_err, scaled_var, ranks, sizes = [], [], [], [], []
    for y, xs in cases.values():
        m = len(xs)
        mean = sum(xs) / m
        v = sum((x - mean) ** 2 for x in xs) / (m - 1)
        var.append(v)
        scaled_var.append(Fraction(m + 1, m) * v)
        sq_err.append((mean - y) ** 2)
        below = sum(x < y for x in xs)
        ties = sum(x == y for x in xs)
        ranks.append(below + ties // 2)
        sizes.append(m)
    rmse = _sqrt(sum(sq_err) / n)
    out: dict[str, float | None] = {
        "ensemble_spread": float(_sqrt(sum(var) / n)),
        "ensemble_mean_root_mean_square_error": float(rmse),
        "spread_skill_ratio": float(_sqrt(sum(scaled_var) / n) / rmse),
        "rank_outlier_fraction": float(
            Fraction(sum(r in (0, m) for r, m in zip(ranks, sizes, strict=True)), n)
        ),
    }
    freq = {k: Fraction(sum(r == k for r in ranks), n) for k in range(4)}
    out |= {f"rank_{k}": float(f) for k, f in freq.items()}
    if len(set(sizes)) == 1:
        m = sizes[0]
        out["rank_histogram_reliability"] = float(
            sum(
                abs(Fraction(sum(r == k for r in ranks), n) - Fraction(1, m + 1))
                for k in range(m + 1)
            )
        )
    else:
        out["rank_histogram_reliability"] = None
    return out


def _winkler(y: Fraction, lo: Fraction, hi: Fraction, alpha: Fraction) -> Fraction:
    return (hi - lo) + (2 / alpha) * max(lo - y, 0) + (2 / alpha) * max(y - hi, 0)


def build_probabilistic() -> dict[str, object]:
    cases = _ensemble_cases()
    n = len(cases)
    ens = {
        "count": len(ENSEMBLE_ROWS),
        "mean_crps_ensemble": float(sum(_crps(y, xs, False) for y, xs in cases.values()) / n),
        "crps_fair": float(sum(_crps(y, xs, True) for y, xs in cases.values()) / n),
        "mean_brier_score": float(
            sum(
                (
                    Fraction(sum(x >= ENSEMBLE_THRESHOLD for x in xs), len(xs))
                    - (y >= ENSEMBLE_THRESHOLD)
                )
                ** 2
                for y, xs in cases.values()
            )
            / n
        ),
    }
    # scoringrules: ensembles padded with NaN to equal size are not supported, so score
    # each case separately.
    sr_crps = np.mean(
        [
            float(sr.crps_ensemble(float(y), np.array([float(x) for x in xs]), estimator="nrg"))
            for y, xs in cases.values()
        ]
    )
    sr_fair = np.mean(
        [
            float(sr.crps_ensemble(float(y), np.array([float(x) for x in xs]), estimator="fair"))
            for y, xs in cases.values()
        ]
    )
    sr_brier = np.mean(
        [
            float(
                sr.brier_score(
                    float(y >= ENSEMBLE_THRESHOLD), np.mean([x >= ENSEMBLE_THRESHOLD for x in xs])
                )
            )
            for y, xs in cases.values()
        ]
    )
    for key, value in (
        ("mean_crps_ensemble", sr_crps),
        ("crps_fair", sr_fair),
        ("mean_brier_score", sr_brier),
    ):
        if not math.isclose(ens[key], value, rel_tol=1e-12):  # type: ignore[arg-type]
            raise AssertionError(f"{key}: exact {ens[key]!r} disagrees with scoringrules {value!r}")
    ens |= _ensemble_diagnostics(cases)
    equal_cases: dict[int, tuple[Fraction, list[Fraction]]] = {}
    for c, y, x in ENSEMBLE_EQUAL_ROWS:
        equal_cases.setdefault(c, (Fraction(y), []))[1].append(Fraction(x))
    ens_equal = {
        "count": len(ENSEMBLE_EQUAL_ROWS),
        "mean_crps_ensemble": float(
            sum(_crps(y, xs, False) for y, xs in equal_cases.values()) / len(equal_cases)
        ),
        **_ensemble_diagnostics(equal_cases),
    }

    rows_q = []
    for c, y, values in QUANTILE_FORECASTS:
        for q, x in zip(QUANTILE_LEVELS, values, strict=True):
            rows_q.append((c, Fraction(y), Fraction(x), q))
    pinball = sum(max(q * (y - x), (q - 1) * (y - x)) for _, y, x, q in rows_q) / len(rows_q)
    sr_pinball = np.mean(
        [float(sr.quantile_score(float(y), float(x), float(q))) for _, y, x, q in rows_q]
    )

    def interval(lo_i: int, hi_i: int) -> tuple[float, float, float]:
        alpha = 1 - (QUANTILE_LEVELS[hi_i] - QUANTILE_LEVELS[lo_i])
        cov, width, wink, wink_sr = [], [], [], []
        for _, y, values in QUANTILE_FORECASTS:
            y = Fraction(y)
            lo, hi = Fraction(values[lo_i]), Fraction(values[hi_i])
            cov.append(Fraction(int(lo <= y <= hi)))
            width.append(hi - lo)
            wink.append(_winkler(y, lo, hi, alpha))
            wink_sr.append(float(sr.interval_score(float(y), float(lo), float(hi), float(alpha))))
        w = float(sum(wink) / len(wink))
        if not math.isclose(w, float(np.mean(wink_sr)), rel_tol=1e-12):
            raise AssertionError(f"winkler: exact {w!r} disagrees with scoringrules")
        return float(sum(cov) / len(cov)), float(sum(width) / len(width)), w

    cov90, width90, wink90 = interval(0, 4)
    cov50, _, wink50 = interval(1, 3)
    if not math.isclose(float(pinball), float(sr_pinball), rel_tol=1e-12):
        raise AssertionError("pinball: exact disagrees with scoringrules")
    quant = {
        "count": len(rows_q),
        "mean_pinball_loss": float(pinball),
        "interval_coverage": cov90,
        "mean_interval_width": width90,
        "mean_interval_score": wink90,
        "coverage_50": cov50,
        "winkler_50": wink50,
    }
    return {
        "description": (
            "Ensemble and quantile forecasts in long format (one row per member or quantile; "
            "rows of a case share obs). Exact values, checked against scoringrules."
        ),
        "ensemble": {
            "rows": [{"case": c, "obs": y, "pred": x} for c, y, x in ENSEMBLE_ROWS],
            "metric_specs": PROBABILISTIC_SPECS["ensemble"],
            "expected": ens,
        },
        "ensemble_equal": {
            "rows": [{"case": c, "obs": y, "pred": x} for c, y, x in ENSEMBLE_EQUAL_ROWS],
            "metric_specs": PROBABILISTIC_SPECS["ensemble_equal"],
            "expected": ens_equal,
        },
        "quantile": {
            "rows": [
                {"case": c, "obs": float(y), "pred": float(x), "quantile": float(q)}
                for c, y, x, q in rows_q
            ],
            "metric_specs": PROBABILISTIC_SPECS["quantile"],
            "expected": quant,
        },
    }


def lead_bin_label(lead_hours: float) -> str:
    start = int(lead_hours // LEAD_BIN_HOURS) * LEAD_BIN_HOURS
    return f"[{start}h, {start + LEAD_BIN_HOURS}h)"


def load_fixture(name: str) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    meta = yaml.safe_load((FIXTURES / name / "datasets.yaml").read_text())
    frames = {
        ds: pd.read_parquet(FIXTURES / name / spec["file"]) for ds, spec in meta["datasets"].items()
    }
    series = {
        f"{ds}__series": pd.read_parquet(FIXTURES / name / spec["series_file"])
        for ds, spec in meta["datasets"].items()
        if "series_file" in spec
    }
    return meta, {**frames, **series}


def pair(
    meta: dict[str, object], frames: dict[str, pd.DataFrame], obs_ds: str, fc_ds: str
) -> pd.DataFrame:
    datasets = meta["datasets"]  # type: ignore[index]
    obs = frames[obs_ds].rename(
        columns={"series_key": "primary_series_key", "value": "primary_value"}
    )
    fc = frames[fc_ds].rename(
        columns={"series_key": "secondary_series_key", "value": "secondary_value"}
    )
    if "usgs_nwm31_crosswalk" in frames:
        xw = frames["usgs_nwm31_crosswalk"].rename(
            columns={"left_key": "primary_series_key", "right_key": "secondary_series_key"}
        )
        fc = fc.merge(xw[["primary_series_key", "secondary_series_key"]], on="secondary_series_key")
    else:
        attr = meta["pairing"]["on_attribute"]  # type: ignore[index]
        so = frames[f"{obs_ds}__series"][["series_key", attr]].rename(
            columns={"series_key": "primary_series_key"}
        )
        sp = frames[f"{fc_ds}__series"][["series_key", attr]].rename(
            columns={"series_key": "secondary_series_key"}
        )
        fc = fc.merge(sp, on="secondary_series_key").merge(so, on=attr).drop(columns=attr)
    factor = UNIT_FACTORS[(datasets[fc_ds]["unit"], datasets[obs_ds]["unit"])]
    fc["secondary_value"] = fc["secondary_value"] * factor
    paired = fc.merge(
        obs[["primary_series_key", "value_time", "primary_value"]],
        on=["primary_series_key", "value_time"],
    )
    paired = paired[
        np.isfinite(paired["primary_value"]) & np.isfinite(paired["secondary_value"])
    ].copy()
    lead_h = (paired["value_time"] - paired["reference_time"]).dt.total_seconds() / 3600
    paired["lead_time_bin"] = lead_h.map(lead_bin_label)
    paired["secondary_dataset"] = fc_ds
    return paired


def evaluate(paired: pd.DataFrame, group_by: list[str]) -> list[dict[str, object]]:
    groups = []
    for keys, g in paired.groupby(group_by, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        groups.append(
            {
                "keys": dict(zip(group_by, keys, strict=True)),
                "metrics": {
                    **checked_metrics(
                        g["primary_value"].to_numpy(), g["secondary_value"].to_numpy()
                    ),
                    **library_variants(
                        g["primary_value"].to_numpy(),
                        g["secondary_value"].to_numpy(),
                        g["value_time"],
                    ),
                },
            }
        )
    return groups


FIXTURE_EVALUATIONS = {
    "hydro": {
        "primary": "usgs_streamflow",
        "forecasts": ["nwm31_short_range", "nwm31_medium_range_mem1"],
        "pairing": (
            "crosswalk usgs_nwm31_crosswalk; exact value_time; predicted converted to observed unit"
        ),
        "groupings": {
            "by_location": ["primary_series_key", "secondary_dataset"],
            "by_lead_time_bin": ["primary_series_key", "secondary_dataset", "lead_time_bin"],
        },
    },
    "energy": {
        "primary": "load_actual",
        "forecasts": ["load_dayahead"],
        "pairing": "series attribute 'zone'; exact value_time",
        "groupings": {
            "by_location": ["primary_series_key", "secondary_dataset"],
            "by_lead_time_bin": ["primary_series_key", "secondary_dataset", "lead_time_bin"],
        },
    },
}


def build_fixture_golden(name: str) -> dict[str, object]:
    cfg = FIXTURE_EVALUATIONS[name]
    meta, frames = load_fixture(name)
    paired = pd.concat([pair(meta, frames, cfg["primary"], fc) for fc in cfg["forecasts"]])
    return {
        "description": (
            f"Golden metrics for the {name} fixture, cross-checked between implementations."
        ),
        "pairing": cfg["pairing"],
        "lead_time_bins": (
            f"{LEAD_BIN_HOURS}h bins labelled '[start h, end h)', "
            "lead = value_time - reference_time"
        ),
        "agreement_rtol": AGREEMENT_RTOL,
        "implementations": {m: list(impls) for m, impls in IMPLEMENTATIONS.items()},
        "pair_count": len(paired),
        "metric_specs": [*METRICS, *VARIANT_SPECS],
        "evaluations": {
            gname: {"group_by": group_by, "groups": evaluate(paired, group_by)}
            for gname, group_by in cfg["groupings"].items()
        },
    }


def main() -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    provenance = {
        lib: version(lib)
        for lib in ("hydroeval", "HydroErr", "numpy", "scipy", "pandas", "scoringrules")
    }
    outputs = {"metric_vectors": build_vectors()}
    for name in FIXTURE_EVALUATIONS:
        outputs[f"{name}_metrics"] = build_fixture_golden(name)
    for fname, payload in outputs.items():
        payload["generated_with"] = provenance  # type: ignore[index]
        path = GOLDEN / f"{fname}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
