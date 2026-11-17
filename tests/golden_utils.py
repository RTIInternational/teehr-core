"""Comparing engine results with golden values (numbers, nulls and ISO timestamps)."""

from datetime import datetime
from typing import Any

import pytest


def assert_matches(got: Any, want: Any, where: Any) -> None:
    """``got`` (engine) equals ``want`` (golden): null, ISO timestamp or number."""
    if want is None:
        assert got is None, where
    elif isinstance(want, str):
        assert isinstance(got, datetime) and got.isoformat() == want, (where, got, want)
    else:
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12), where
