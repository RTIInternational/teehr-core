"""The metric catalog: every registered metric, described for the people choosing one.

:func:`catalog` is the one source for metric references: the generated
``docs/metrics.md``, APIs that list metrics, and CLIs. Each entry has the metric's
canonical name and aliases, family, definition, spec parameters (with which are
required), the pair columns it reads besides the paired values, the unit its result
is in, and whether it merges from state (so runs of it can be incremental).
"""

from __future__ import annotations

from typing import Any

from teehr_core.metrics.registry import ALIASES, METRICS, MetricRequest, Params
from teehr_core.metrics.state import mergeable
from teehr_core.metrics.units import KINDS, UnitKind

#: Each spec parameter: what it is and its default ("required" where a metric needs it).
PARAMETERS: dict[str, tuple[str, str]] = {
    "threshold": (
        "The event threshold (a value >= it is an event): `{value, unit}`, or "
        "`{column: <name>}` for a per-pair threshold",
        "required",
    ),
    "normalize_by": (
        "What normalizes the error: `mean_primary`, `range_primary`, `{value, unit}` or "
        "`{column: <name>}`",
        "`mean_primary`",
    ),
    "side": ("Which side the signature describes: `primary` or `secondary`", "`primary`"),
    "year_start_month": (
        "The month a year starts in (1-12), for annual and timing signatures",
        "1; 10 (a water year) for timing",
    ),
    "missing_day_threshold": (
        "Timing: leave out a year missing more than this share of its days",
        "0.1",
    ),
    "exceedance": ("Flow duration curve slope: the two exceedance probabilities", "`[0.33, 0.66]`"),
    "log": ("Flow duration curve slope: on log values (`true`) or linear", "`true`"),
    "estimator": ("CRPS: `standard` or `fair` (for small ensembles)", "`standard`"),
    "interval": ("The central prediction interval, as two quantile levels", "`[0.05, 0.95]`"),
    "rank": (
        "Rank histogram: which observation rank among the members to count (0 = below all)",
        "0",
    ),
    "scale": (
        "MASE: the pair column with each pair's naive scale (`{column: <name>}`, from a "
        "`naive_scale` field)",
        "required",
    ),
    "sr": ("KGE: the weight on the correlation term", "1"),
    "sa": ("KGE: the weight on the variability term", "1"),
    "sb": ("KGE: the weight on the bias term", "1"),
}

#: What each unit kind means for a result, in a reader's words.
UNIT_KINDS: dict[UnitKind, str] = {
    UnitKind.VALUE: "the evaluation unit (a level)",
    UnitKind.SUM: "the evaluation unit (a sum)",
    UnitKind.DIFFERENCE: "the evaluation unit (a difference)",
    UnitKind.DIFFERENCE2: "the evaluation unit, squared",
    UnitKind.INVARIANT: "none (dimensionless)",
    UnitKind.RATIO: "none (a ratio; not with an offset unit such as °C)",
    UnitKind.TIME: "a time or duration",
}

#: Family order and headings in references.
FAMILIES: dict[str, str] = {
    "count": "Count",
    "error": "Error",
    "relative": "Relative error and bias",
    "efficiency": "Efficiency",
    "correlation": "Correlation",
    "contingency": "Contingency (events above a threshold)",
    "signature": "Hydrologic signatures",
    "timing": "Timing",
    "probabilistic": "Probabilistic (ensembles and quantiles)",
}

#: Example values for required parameters, to probe whether a metric merges.
_PROBE: dict[str, dict[str, Any]] = {
    "threshold": {"threshold": 1.0},
    "scale": {"scale_column": "_scale"},
    "rank": {"rank": 1},
}


def _merges(name: str, required: frozenset[str]) -> bool:
    kwargs: dict[str, Any] = {}
    for p in required:
        kwargs.update(_PROBE.get(p, {}))
    return mergeable([MetricRequest(name, params=Params(**kwargs))])


def catalog() -> list[dict[str, Any]]:
    """Every registered metric, in family order, then by name."""
    order = list(FAMILIES)
    out = []
    for name, m in METRICS.items():
        out.append({
            "name": name,
            "aliases": sorted(a for a, c in ALIASES.items() if c == name),
            "family": m.family,
            "description": m.description,
            "parameters": sorted(m.params),
            "required": sorted(m.required),
            "reads": sorted(n for n in m.needs if not n.startswith("_")),
            "output": m.output,
            "unit": KINDS[name].value,
            "merges": _merges(name, m.required),
            "skill": m.skill,
            "transforms": m.transforms,
        })  # fmt: skip
    rank = {f: i for i, f in enumerate(order)}
    return sorted(out, key=lambda e: (rank.get(e["family"], len(order)), e["name"]))
