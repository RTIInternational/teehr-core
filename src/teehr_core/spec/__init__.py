"""Evaluation spec models, loading and JSON Schema.

Ported from VeriRace (``verirace/spec/__init__.py`` at 668ab7d).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from teehr_core.spec.expressions import ExpressionError, check_expression
from teehr_core.spec.models import (
    SUMMARIES,
    AttributeField,
    CalendarField,
    ChangeField,
    ConditionsField,
    EvaluationSpec,
    EventsField,
    ExceedanceField,
    ExprField,
    GeneratedInput,
    InputRef,
    Inputs,
    LeadTimeBinsField,
    MetricSpec,
    NaiveScaleField,
    Output,
    Pairing,
    PercentileField,
    Quantity,
    Reduce,
    Stage,
    TimeMatch,
    WithinField,
)

__all__ = [
    "SUMMARIES",
    "AttributeField",
    "CalendarField",
    "ChangeField",
    "ConditionsField",
    "EvaluationSpec",
    "EventsField",
    "ExceedanceField",
    "ExprField",
    "ExpressionError",
    "GeneratedInput",
    "InputRef",
    "Inputs",
    "LeadTimeBinsField",
    "MetricSpec",
    "NaiveScaleField",
    "Output",
    "Pairing",
    "PercentileField",
    "Quantity",
    "Reduce",
    "Stage",
    "TimeMatch",
    "WithinField",
    "check_expression",
    "json_schema",
    "load_spec",
    "spec_warnings",
]


def load_spec(source: EvaluationSpec | Mapping[str, Any] | str | Path) -> EvaluationSpec:
    """Load a spec from a model, a mapping, a YAML/JSON file path, or YAML text.

    A ``str`` containing a newline is YAML text; otherwise it is a file path.
    """
    if isinstance(source, EvaluationSpec):
        return source
    if isinstance(source, Mapping):
        return EvaluationSpec.model_validate(source)
    text = source if isinstance(source, str) and "\n" in source else Path(source).read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ValueError("a spec must be a YAML/JSON mapping")
    return EvaluationSpec.model_validate(data)


def json_schema() -> dict[str, Any]:
    """Return the JSON Schema for evaluation specs (version 1)."""
    return EvaluationSpec.model_json_schema(by_alias=True)


def spec_warnings(spec: EvaluationSpec) -> list[str]:
    """Return what a valid spec probably shouldn't do: costly or surprising choices."""
    warnings = []
    for ref in spec.inputs.secondary:
        if isinstance(ref, GeneratedInput) and ref.generate != "climatology" and not ref.max_age:
            name = ref.name or f"{ref.generate}:{ref.like}"
            warnings.append(
                f"{name!r}: {ref.generate} without max_age may use an observation from any "
                "time before each forecast, so the run reads every observed series' whole "
                "record, not just the forecasts' span: often many times the data and memory "
                "needed. Set max_age (e.g. PT2H for hourly forecasts)."
            )
    return warnings
