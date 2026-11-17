"""A transform's offset (shared spec: ``add_epsilon`` dropped; ``{name, offset}`` instead)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from teehr_core import metrics
from teehr_core.metrics.registry import MetricRequest, OffsetTransform, Transform
from teehr_core.metrics.units import result_factors
from teehr_core.model.units import Conversion
from teehr_core.spec import load_spec
from teehr_core.spec.models import MetricSpec

OBS = [0.0, 2.0, 5.0, 0.0, 9.0, 4.0]
SIM = [0.5, 1.0, 6.0, 0.0, 7.0, 4.5]
CFS_TO_CMS = Conversion(0.028316846592, 0.0)


def test_offset_keeps_zeros_a_plain_log_drops() -> None:
    plain = metrics.evaluate(OBS, SIM, ["count", "rmse"], transform="log")
    shifted = metrics.evaluate(OBS, SIM, ["count", "rmse"], transform="log", transform_offset=1.0)
    assert plain["count"] == 4 and shifted["count"] == 6  # two pairs have a zero
    want = np.sqrt(np.mean((np.log(np.add(SIM, 1)) - np.log(np.add(OBS, 1))) ** 2))
    assert shifted["root_mean_square_error"] == pytest.approx(want, rel=1e-12)


def test_an_offset_needs_a_transform() -> None:
    with pytest.raises(ValueError, match="needs a transform"):
        metrics.evaluate(OBS, SIM, ["rmse"], transform_offset=1.0)


def test_spec_form() -> None:
    m = MetricSpec.model_validate(
        {"name": "nse", "transform": {"name": "inv", "offset": {"value": 0.01, "unit": "m3/s"}}}
    )
    assert m.transform_name is Transform.RECIPROCAL
    assert m.output_name() == "nash_sutcliffe_efficiency_reciprocal"
    req = m.to_request()
    assert req == MetricRequest(
        "nash_sutcliffe_efficiency", OffsetTransform(Transform.RECIPROCAL, 0.01), m.to_params()
    )
    plain = MetricSpec.model_validate({"name": "nse", "transform": "log"})
    assert plain.to_request().transform is Transform.LOG


def test_offset_needs_its_unit() -> None:
    with pytest.raises(ValidationError, match="needs its unit"):
        MetricSpec.model_validate({"name": "nse", "transform": {"name": "log", "offset": 0.01}})


def test_offset_metrics_convert_like_their_transform() -> None:
    """log(a x + a c) = log a + log(x + c): differences of logs don't change with the unit."""
    spec = load_spec(
        {
            "version": 1,
            "name": "t",
            "inputs": {"primary": {"dataset": "obs"}, "secondary": {"dataset": "fc"}},
            "pair": {"relation": "xw"},
            "aggregate": [
                {
                    "group_by": ["primary_series_key"],
                    "metrics": [
                        {
                            "name": "rmse",
                            "transform": {"name": "log", "offset": {"value": 1, "unit": "m3/s"}},
                        }
                    ],
                }
            ],
        }
    )
    assert result_factors(spec, CFS_TO_CMS)["root_mean_square_error_log"] == (1.0, 0.0)
    # Computed in each unit, with the offset converted by scale alone: the same result.
    a = CFS_TO_CMS.scale
    in_cfs = metrics.evaluate(OBS, SIM, ["rmse"], transform="log", transform_offset=1.0)
    in_cms = metrics.evaluate(
        np.multiply(OBS, a), np.multiply(SIM, a), ["rmse"], transform="log", transform_offset=a
    )
    assert in_cfs["root_mean_square_error"] == pytest.approx(
        in_cms["root_mean_square_error"], rel=1e-12
    )
