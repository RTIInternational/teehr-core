"""Metrics and options teehr has that VeriRace didn't (teehr-core design, step 2)."""

import numpy as np
import pytest

from teehr_core import metrics


def test_proportion_correct() -> None:
    obs = [10.0, 60.0, 70.0, 20.0, 55.0]
    pred = [12.0, 65.0, 40.0, 51.0, 58.0]
    # threshold 50: hits 2 (60/65, 55/58), miss 1 (70/40), false alarm 1 (20/51), cn 1
    assert metrics.proportion_correct(obs, pred, threshold=50.0) == pytest.approx(3 / 5)
    got = metrics.evaluate(obs, pred, ["accuracy"], threshold=50.0)
    assert got["proportion_correct"] == pytest.approx(3 / 5)


def test_fdc_slope_linear_option() -> None:
    values = np.random.default_rng(3).gamma(2.0, 30.0, 500)
    log_form = metrics.flow_duration_curve_slope(values)
    linear = metrics.flow_duration_curve_slope(values, log=False)
    q33, q66 = np.quantile(values, [0.67, 0.34], method="averaged_inverted_cdf")
    assert linear == pytest.approx((q33 - q66) / 0.33, rel=2e-3)  # sketch-snapped
    assert log_form == pytest.approx((np.log(q33) - np.log(q66)) / 0.33, rel=2e-2)
    # The linear form is defined for values <= 0; the log form isn't.
    shifted = values - 100.0
    assert metrics.flow_duration_curve_slope(shifted, log=False) is not None
    assert metrics.flow_duration_curve_slope(shifted) is None
