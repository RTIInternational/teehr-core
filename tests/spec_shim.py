"""A stand-in for VeriRace's ``MetricSpec`` until the spec models are ported (step 4).

Turns a spec's metric entry (a dict) into a :class:`MetricRequest` and its output column
name, as ``MetricSpec.to_params()`` / ``output_name()`` do, for the forms the tests use.
"""

from typing import Any

from teehr_core.metrics.registry import ALIASES, METRICS, MetricRequest, Params, Transform

_NORMALIZERS = {"mean_primary": "mean_obs", "range_primary": "range_obs"}
_SIDES = {"primary": "obs", "secondary": "pred"}
_TRANSFORMS = {"inv": "reciprocal"}


def _name(spec: dict[str, Any]) -> str:
    name = str(spec["name"])
    return ALIASES.get(name, name)


def to_request(spec: dict[str, Any]) -> MetricRequest:
    p: dict[str, Any] = {}
    t = spec.get("threshold")
    if isinstance(t, dict):
        if "column" in t:
            p["threshold_column"] = t["column"]
        elif "primary_quantile" in t:
            p["threshold_quantile"] = t["primary_quantile"]
        else:
            p["threshold"] = t["value"]
    nb = spec.get("normalize_by")
    if isinstance(nb, dict) and "column" in nb:
        p["normalize_by"], p["normalize_column"] = "column", nb["column"]
    elif isinstance(nb, str):
        p["normalize_by"] = _NORMALIZERS[nb]
    elif isinstance(nb, dict):
        p["normalize_by"], p["normalize_value"] = "value", nb["value"]
    if spec.get("side") is not None:
        p["side"] = _SIDES[spec["side"]]
    for key in ("year_start_month", "estimator", "rank"):
        if spec.get(key) is not None:
            p[key] = spec[key]
    for key in ("exceedance", "interval", "kge_weights"):
        if spec.get(key) is not None:
            p[key] = tuple(spec[key])
    if spec.get("scale") is not None:
        p["scale_column"] = spec["scale"]["column"]
    transform = spec.get("transform")
    if transform is not None:
        transform = Transform(_TRANSFORMS.get(transform, transform))
    return MetricRequest(_name(spec), transform, Params(**p))


def output_name(spec: dict[str, Any]) -> str:
    if spec.get("as") is not None:
        return str(spec["as"])
    name = _name(spec)
    if spec.get("of") is not None:
        return f"{name}_{spec['of']}"
    metric = METRICS.get(name)
    if metric is not None and "side" in metric.params:
        name = f"{name}_{spec.get('side') or 'primary'}"
    if spec.get("values") is not None:
        name = f"{name}_{spec['values']['primary']}"
    transform = spec.get("transform")
    if transform is not None:
        return f"{name}_{_TRANSFORMS.get(transform, transform)}"
    return name
