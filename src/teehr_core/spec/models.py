"""Evaluation spec, version 1.

Ported from VeriRace (``verirace/spec/models.py`` at 668ab7d), with teehr's metric options
(``log``, ``missing_day_threshold``).

A spec describes one evaluation as a fixed sequence of steps:
``inputs → pair → fields → filter → aggregate → output``. See design doc §6.
"""

from __future__ import annotations

import re
from datetime import timedelta
from itertools import pairwise
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    WithJsonSchema,
    field_validator,
    model_validator,
)

from teehr_core.durations import DURATION_PATTERN, MONTHS_PATTERN, parse_duration, parse_months
from teehr_core.metrics.registry import (
    ALIASES,
    METRICS,
    SPEC_NORMALIZERS,
    SPEC_SIDES,
    TRANSFORM_ALIASES,
    MetricRequest,
    OffsetTransform,
    Params,
    Transform,
    TransformKey,
    canonical_name,
)
from teehr_core.spec.expressions import check_expression

__all__ = [
    "SUMMARIES",
    "AttributeField",
    "ChangeField",
    "EvaluationSpec",
    "EventsField",
    "ExceedanceField",
    "ExprField",
    "GeneratedInput",
    "Guard",
    "InputRef",
    "Inputs",
    "LeadTimeBinsField",
    "MetricSpec",
    "NaiveScaleField",
    "Output",
    "Pairing",
    "PercentileField",
    "Quantity",
    "Stage",
    "TimeMatch",
    "ValuesRef",
    "WithinField",
]

Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z_][a-z0-9_]*$")]
_DATASET_REF = r"^(public\.)?[a-z][a-z0-9_]{0,59}$"


def _check_dataset_ref(value: str) -> str:
    if not re.fullmatch(_DATASET_REF, value):
        raise ValueError(
            f"{value!r} isn't a dataset reference: give a dataset's name (lowercase letters, "
            "digits and '_', starting with a letter) or public.<name> for the public lake"
        )
    return value


#: A dataset in the project's lake (``name``) or the public lake (``public.name``). Names
#: follow the lakes' own rule, so a reference can't carry anything but a name into SQL.
DatasetRef = Annotated[
    str,
    AfterValidator(_check_dataset_ref),
    WithJsonSchema({"type": "string", "pattern": _DATASET_REF}),
]


def _check_duration(v: str) -> str:
    parse_duration(v)
    return v


#: A fixed-length ISO 8601 duration, e.g. ``PT15M``, ``PT6H``, ``P1D``.
Duration = Annotated[
    str, StringConstraints(pattern=DURATION_PATTERN), AfterValidator(_check_duration)
]
#: Whole calendar months as ISO 8601, e.g. ``P1M``, ``P3M`` (a bootstrap block).
MonthsBlock = Annotated[str, StringConstraints(pattern=MONTHS_PATTERN)]
#: A column of a series table or layer, as uploaded (case kept; quoted in SQL).
ColumnName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")]

#: Summary functions for aggregate stages after the first.
SUMMARIES = ("mean", "median", "min", "max", "std", "sum", "count")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class InputRef(_Model):
    """A dataset used as input, optionally filtered."""

    dataset: DatasetRef = Field(description="Dataset name in the catalog.")
    filter: str | None = Field(
        default=None,
        description="Expression applied before pairing; numbers compared with value are in "
        "the variable's canonical unit.",
    )

    @field_validator("filter")
    @classmethod
    def _check_filter(cls, v: str | None) -> str | None:
        return None if v is None else check_expression(v)

    @property
    def source(self) -> str:
        """The dataset read from the catalog."""
        return self.dataset

    @property
    def label(self) -> str:
        """Its name in the ``secondary_dataset`` column (and as a skill reference)."""
        return self.dataset


class GeneratedInput(_Model):
    """A secondary dataset generated from the observations: a reference for skill scores.

    - ``persistence`` (``like``): for every forecast in ``like`` (series, reference time,
      value time), the last observation at or before the reference time, for the
      observed series linked to that forecast series. Computed within the run from the
      observations it reads.
    - ``climatology`` (``from``): a stored climatology dataset (one value per series, day
      of year and hour, built over the period of record) looked up at each observation's
      day of year and hour. No reference times, so it can't be grouped by lead time.
    - ``blend`` (``like`` and ``from``): the optimal combination of persistence and
      climatology (Yang et al. 2020), on the grid of ``like``: climatology at the value
      time plus the persistence anomaly (observation minus climatology at the reference
      time) damped by the anomalies' autocorrelation at that lead, fitted per observed
      series and whole-hour lead from the observations the run reads.

    Persistence (and so the blend) uses observations complete by the reference time: an
    instantaneous value at or before it, or a value over a period whose window has ended.
    Pairs of generated inputs are keyed by the observed series.
    """

    generate: Literal["persistence", "climatology", "blend"] = Field(
        description="What to generate."
    )
    like: DatasetRef | None = Field(
        default=None,
        description="persistence: a forecast dataset whose grid (series, reference times, "
        "value times) the persistence forecast copies.",
    )
    from_: DatasetRef | None = Field(
        default=None,
        alias="from",
        description="climatology: a climatology dataset (kind 'climatology').",
    )
    name: str | None = Field(
        default=None, description="Its name in secondary_dataset; default '<generate>:<source>'."
    )
    max_age: Duration | None = Field(
        default=None,
        description="persistence (and blend): use only an observation at most this old at "
        "the reference time, e.g. PT2H for hourly forecasts. Set it: without it a run can't "
        "limit the observations it reads to the forecasts' span (an observation from any "
        "time before may be used), so it reads every series' whole record, which can be "
        "many times the data and memory the evaluation needs.",
    )
    filter: str | None = Field(
        default=None,
        description="Expression applied before pairing; numbers compared with value are in "
        "the variable's canonical unit.",
    )

    @field_validator("filter")
    @classmethod
    def _check_filter(cls, v: str | None) -> str | None:
        return None if v is None else check_expression(v)

    @model_validator(mode="after")
    def _check_source(self) -> GeneratedInput:
        if self.generate == "blend" and (self.like is None or self.from_ is None):
            raise ValueError("blend takes 'like' (a forecast dataset) and 'from' (a climatology)")
        if self.generate == "persistence" and (self.like is None or self.from_ is not None):
            raise ValueError("persistence takes 'like' (a forecast dataset), not 'from'")
        if self.generate == "climatology" and (self.from_ is None or self.like is not None):
            raise ValueError("climatology takes 'from' (a climatology dataset), not 'like'")
        if self.generate == "climatology" and self.max_age is not None:
            raise ValueError("max_age applies to persistence only")
        return self

    @property
    def source(self) -> str:
        """The dataset read from the catalog (the forecast grid or the climatology)."""
        source = self.from_ if self.generate == "climatology" else self.like
        assert source is not None
        return source

    @property
    def label(self) -> str:
        """Its name in the ``secondary_dataset`` column (and as a skill reference)."""
        return self.name or f"{self.generate}:{self.source}"

    @property
    def dataset(self) -> str:
        """Same as :attr:`label`: secondary inputs are named by ``dataset``."""
        return self.label


class Inputs(_Model):
    """The primary dataset (usually observations) and one or more secondary datasets.

    Pairing sides are named ``primary`` and ``secondary`` (paired columns ``primary_*`` and
    ``secondary_*``), shared with the TEEHR team. A dataset's role (observed, simulated,
    forecast) still says what its data is.
    """

    primary: InputRef = Field(description="The dataset scored against: usually observations.")
    secondary: list[InputRef | GeneratedInput] = Field(
        min_length=1,
        description=(
            "Datasets scored against the primary one (simulations or forecasts). Each becomes "
            "a value of the 'secondary_dataset' column in pairs."
        ),
    )

    @field_validator("secondary", mode="before")
    @classmethod
    def _single_to_list(cls, v: Any) -> Any:
        return [v] if isinstance(v, dict | InputRef | GeneratedInput) else v


class TimeMatch(_Model):
    """How predicted values are aligned with observed values in time."""

    method: Literal["exact"] = "exact"


class Pairing(_Model):
    """How predicted series are matched to observed series."""

    relation: DatasetRef | None = Field(
        default=None,
        description="Crosswalk dataset mapping primary (left) to secondary (right) keys.",
    )
    on_attribute: str | None = Field(
        default=None,
        description="Series catalog attribute shared by primary and secondary series.",
    )
    time: TimeMatch = TimeMatch()
    common_sample: bool = Field(
        default=False,
        description="Score every secondary input on the same sample: keep only observations "
        "(primary series and value time) that every secondary input paired with, after "
        "filters. A forecast missing on hard days then can't look better for it.",
    )

    @model_validator(mode="after")
    def _one_method(self) -> Pairing:
        if (self.relation is None) == (self.on_attribute is None):
            raise ValueError("pair needs exactly one of 'relation' or 'on_attribute'")
        return self


class ExprField(_Model):
    """A row-level field computed from a SQL expression."""

    name: Identifier
    expr: str

    @field_validator("expr")
    @classmethod
    def _check(cls, v: str) -> str:
        return check_expression(v)


class ColumnRef(_Model):
    """A pair column holding a per-pair value (e.g. a flood threshold or a capacity)."""

    column: Identifier


class Quantity(_Model):
    """A value with its unit (a UCUM symbol), e.g. ``{value: 1000, unit: "[cft_i]/s"}``.

    Building the plan converts it to the evaluation unit (``output.unit``, else the
    primary dataset's), so a spec means the same thing whatever unit its data are in.
    """

    value: float
    unit: Annotated[str, StringConstraints(min_length=1)]


def _require_unit(v: Any) -> Any:
    """Refuse a bare number where a spec gives a value: it needs its unit."""
    if isinstance(v, int | float) and not isinstance(v, bool):
        raise ValueError(
            f"{v!r} needs its unit: give {{value: {v}, unit: <UCUM symbol>}}, "
            f'e.g. {{value: {v}, unit: "[cft_i]/s"}}'
        )
    return v


class SeriesQuantile(_Model):
    """A quantile of each series' own values (in the evaluated window)."""

    quantile: Annotated[float, Field(gt=0, lt=1)]


#: A pairing side: the primary dataset's values or the secondary's.
Side = Literal["primary", "secondary"]
#: The metrics' name for each side's values (teehr_core.metrics: obs and pred).
SIDE_COLUMNS = SPEC_SIDES
_NORMALIZERS = SPEC_NORMALIZERS


class ExceedanceField(_Model):
    """Each value's exceedance probability within its series.

    Weibull plotting position: rank (largest first, ties share the lower rank) / (n + 1)
    over the series' distinct times. ``side: primary`` ranks each primary series;
    ``side: secondary`` ranks each forecast trajectory (dataset, series, reference time and
    member or quantile). Computed over the pairs being evaluated (the evaluated window).
    """

    name: Identifier
    fn: Literal["exceedance"]
    side: Side = "primary"


class EventsField(_Model):
    """Events: runs of values at or above a threshold, labelled per series.

    Each pair inside an event gets the event's id (``<series>@<start>``); others get
    null, so ``group_by`` the field scores each event and ``filter`` keeps event pairs.
    ``threshold`` is a value with its unit (``{value: v, unit: u}``), ``{quantile: q}`` of
    each series' own values (in the evaluated window), or ``{column: c}``: a pair column
    such as a ``percentile`` field, for a threshold from the series' whole record. Runs
    separated by less than ``merge_gap`` join into one event (the gap included); events
    shorter than ``min_duration`` are dropped.
    """

    name: Identifier
    fn: Literal["events"]
    side: Side = "primary"
    threshold: Quantity | SeriesQuantile | ColumnRef
    min_duration: Duration | None = None
    merge_gap: Duration | None = None

    _units = field_validator("threshold", mode="before")(_require_unit)


class ChangeField(_Model):
    """Each value's change over a window within its series: ``value(t) - value(t - window)``.

    ``side: primary`` uses each primary series; ``side: secondary`` each forecast trajectory
    (dataset, series, reference time and member or quantile). Null when the series has no
    value exactly ``window`` earlier. With the metrics' ``values`` option, contingency
    metrics then score ramps: a change of at least the threshold.
    """

    name: Identifier
    fn: Literal["change"]
    side: Side = "primary"
    window: Duration

    @model_validator(mode="after")
    def _check(self) -> ChangeField:
        if parse_duration(self.window) <= timedelta(0):
            raise ValueError("change window must be positive")
        return self


class NaiveScaleField(_Model):
    """Each pair's in-sample naive scale, for MASE.

    The observed series' stored mean ``|y(t) - y(t - lag)|`` over a training period, from
    a ``naive_scale`` dataset, converted to the observed unit.
    """

    name: Identifier
    fn: Literal["naive_scale"]
    from_: DatasetRef = Field(alias="from", description="A naive_scale dataset.")
    lag: Duration = Field(description="The naive forecast's lag, as built (e.g. PT1H, P1D).")


class PercentileField(_Model):
    """Each pair's stored percentile of its observed series, over its period of record.

    From a ``percentiles`` dataset (built over the whole record, so not limited to the
    evaluated window), converted to the observed unit; null for series without one (too
    short a record). Non-exceedance: the 90th is the value not exceeded 90% of the time.
    Use it as a threshold: ``{column: <name>}``.
    """

    name: Identifier
    fn: Literal["percentile"]
    from_: DatasetRef = Field(alias="from", description="A percentiles dataset.")
    percentile: Annotated[float, Field(gt=0, lt=100)] = Field(
        description="Which percentile (0 to 100), as built (e.g. 90, 99.5)."
    )


class LeadTimeBinsField(_Model):
    """Lead time (``value_time - reference_time``) binned into labelled intervals.

    Labels look like ``[0h, 6h)``. Give either a fixed ``width`` or explicit ``edges``;
    with edges, lead times past the last edge fall in ``[last, inf)``.
    """

    name: Identifier
    fn: Literal["lead_time_bins"]
    width: Duration | None = None
    edges: list[Duration] | None = None

    @model_validator(mode="after")
    def _check(self) -> LeadTimeBinsField:
        if (self.width is None) == (self.edges is None):
            raise ValueError("lead_time_bins needs exactly one of 'width' or 'edges'")
        if self.width is not None and parse_duration(self.width) <= timedelta(0):
            raise ValueError("lead_time_bins width must be positive")
        if self.edges is not None:
            parsed = [parse_duration(e) for e in self.edges]
            if len(parsed) < 2 or any(b <= a for a, b in pairwise(parsed)):
                raise ValueError("lead_time_bins edges must be at least two increasing durations")
        return self


class CalendarField(_Model):
    """A calendar label of each pair's time, to group by (water year, quarter, month...).

    ``of`` is the time labelled: ``value_time`` (default) or ``reference_time`` (when a
    forecast was issued; null for pairs without one). Times are UTC. Labels are strings:

    - ``year``: ``2025``; ``month``: ``2025-03``;
    - ``water_year``: the year ``year_start_month`` (default 10, October) starts, named for
      the calendar year it ends in (October 2024 is water year 2025);
    - ``quarter``: calendar quarters, ``2025-Q1`` (January to March); with
      ``year_start_month``, quarters of that year instead, named by its water year
      (``2025-Q1`` is October to December 2024 for 10);
    - ``season``: meteorological seasons ``DJF``, ``MAM``, ``JJA``, ``SON`` (across years;
      group by a year field too for one season per year).
    """

    name: Identifier
    fn: Literal["calendar"]
    part: Literal["year", "month", "water_year", "quarter", "season"]
    of: Literal["value_time", "reference_time"] = "value_time"
    year_start_month: Annotated[int, Field(ge=1, le=12)] | None = Field(
        default=None,
        description="The month a year starts: water_year default 10, quarter default 1.",
    )

    @model_validator(mode="after")
    def _check(self) -> CalendarField:
        if self.year_start_month is not None and self.part not in ("water_year", "quarter"):
            raise ValueError(f"calendar part {self.part!r} doesn't take year_start_month")
        return self

    def start_month(self) -> int:
        """Return the month this label's year starts (1: calendar years)."""
        if self.year_start_month is not None:
            return self.year_start_month
        return 10 if self.part == "water_year" else 1


class ConditionsField(_Model):
    """Overlapping subsets: each pair is copied into every level whose condition it meets.

    ``levels`` maps a label to a SQL condition (checked like ``filter``), e.g.
    ``above_q95: "primary_value > q95"``; with ``all`` (default true) every pair also counts
    in a level labelled ``all``. Unlike other fields a pair can be in several levels, so
    grouping by the field scores each level over its own pairs (a pair above q95 counts
    in ``above_q85`` too). Conditions may use any other field: the copies are made after
    every other field and before ``filter``, which can keep a level (``flow =
    'above_q95'``). Several conditions fields multiply (each pair in every combination).
    """

    name: Identifier
    fn: Literal["conditions"]
    levels: dict[Identifier, str] = Field(min_length=1)
    all: bool = Field(default=True, description="Also an 'all' level: every pair.")

    @field_validator("levels")
    @classmethod
    def _check_levels(cls, v: dict[str, str]) -> dict[str, str]:
        if "all" in v:
            raise ValueError("'all' is the level every pair is in: name the others otherwise")
        return {label: check_expression(expr) for label, expr in v.items()}

    def labels(self) -> list[str]:
        """Return the level labels, ``all`` first when it's on."""
        return (["all"] if self.all else []) + list(self.levels)


def _no_geometry_label(attribute: str) -> str:
    # A geometry label would ride on every pair; output.geometry adds it per result row.
    if attribute == "geometry":
        raise ValueError(
            "a field can't label pairs with a geometry: group by the key and set "
            "output.geometry to give each result row its geometry"
        )
    return attribute


class AttributeField(_Model):
    """An attribute of each pair's observed series, to group or filter by (e.g. a state).

    ``of: series`` (default) reads it from the observed dataset's series table; ``of:
    location`` from the series' location in the dataset's locations layer. Constant per
    series; null where the series or location has none.
    """

    name: Identifier
    fn: Literal["attribute"]
    attribute: ColumnName = Field(description="The column in the series table or layer.")
    of: Literal["series", "location"] = "series"

    @field_validator("attribute")
    @classmethod
    def _not_geometry(cls, v: str) -> str:
        return _no_geometry_label(v)


class WithinField(_Model):
    """The polygon of a layer each pair's observed series is in, to group by (e.g. a basin).

    Each series' location (the observed dataset's locations layer; a line or polygon by
    its centroid) is matched to the polygons of ``layer`` that contain it, boundaries
    included. The value is that polygon's ``attribute`` (default its key); where polygons
    overlap, the one with the smallest key; null when no polygon contains the location.
    """

    name: Identifier
    fn: Literal["within"]
    layer: DatasetRef = Field(description="A polygon layer: name, or public.<name>.")
    attribute: ColumnName = Field(
        default="location_key", description="The polygon attribute to label with (its key)."
    )

    @field_validator("attribute")
    @classmethod
    def _not_geometry(cls, v: str) -> str:
        return _no_geometry_label(v)


#: Fields with one value per observed series: a reduce carries them into its windows.
SERIES_CONSTANT = (NaiveScaleField, PercentileField, AttributeField, WithinField)

FieldSpec = (
    ExprField
    | LeadTimeBinsField
    | CalendarField
    | ConditionsField
    | ExceedanceField
    | EventsField
    | ChangeField
    | NaiveScaleField
    | PercentileField
    | AttributeField
    | WithinField
)


class ObsQuantile(_Model):
    """A quantile of each group's observations."""

    primary_quantile: Annotated[float, Field(gt=0, lt=1)]


#: Metric options; each metric accepts only some of them (see the metric registry).
_PARAM_FIELDS = (
    "threshold",
    "normalize_by",
    "side",
    "year_start_month",
    "exceedance",
    "estimator",
    "interval",
    "rank",
    "scale",
    "sr",
    "sa",
    "sb",
    "log",
    "missing_day_threshold",
)


class TransformWithOffset(_Model):
    """A transform of the values shifted first: ``name(x + offset)``.

    E.g. ``log`` of flows with zero days, which a plain ``log`` leaves out.
    """

    name: Transform
    offset: Quantity = Field(
        description="Added to both sides before the transform: a value with its unit "
        '({value: 0.01, unit: "m3/s"}), converted to the evaluation unit by scale alone.'
    )

    @field_validator("name", mode="before")
    @classmethod
    def _alias(cls, v: Any) -> Any:
        return TRANSFORM_ALIASES.get(v, v) if isinstance(v, str) else v

    _units = field_validator("offset", mode="before")(_require_unit)


class ValuesRef(_Model):
    """Pair columns a metric reads instead of ``primary_value`` and ``secondary_value``."""

    primary: Identifier
    secondary: Identifier


class MetricSpec(_Model):
    """A metric (first stage) or summary (later stages) with options."""

    name: str = Field(
        description="A metric's canonical name or one of its aliases (e.g. rmse for "
        "root_mean_square_error); results carry the canonical name. Later stages: a summary."
    )
    as_: Identifier | None = Field(default=None, alias="as", description="Output column name.")

    @field_validator("name")
    @classmethod
    def _canonical(cls, v: str) -> str:
        return canonical_name(v)

    @field_validator("of")
    @classmethod
    def _canonical_of(cls, v: str | None) -> str | None:
        return None if v is None else canonical_column(v)

    @field_validator("as_")
    @classmethod
    def _not_an_alias(cls, v: str | None) -> str | None:
        if v is not None and canonical_column(v) != v:
            raise ValueError(
                f"'as: {v}' would read as the metric alias for {canonical_column(v)!r}; "
                "choose another name"
            )
        return v

    transform: Transform | TransformWithOffset | None = Field(
        default=None,
        description="First stage only: transform primary and secondary values before the "
        "metric (log is natural; inv is reciprocal), by name or as {name, offset} to add "
        "an offset first.",
    )

    @field_validator("transform", mode="before")
    @classmethod
    def _transform_alias(cls, v: Any) -> Any:
        return TRANSFORM_ALIASES.get(v, v) if isinstance(v, str) else v

    of: str | None = Field(default=None, description="Later stages only: column to summarize.")
    threshold: Quantity | ColumnRef | ObsQuantile | None = Field(
        default=None,
        description="Contingency metrics: event threshold (value >= threshold): a value "
        'with its unit ({value: 1000, unit: "[cft_i]/s"}), {column: <pair column>} or '
        "{primary_quantile: 0.9} (per group).",
    )
    normalize_by: Literal["mean_primary", "range_primary"] | Quantity | ColumnRef | None = Field(
        default=None,
        description="nmae/nrmse: mean_primary (default), range_primary, a value with its "
        'unit (e.g. installed capacity, {value: 500, unit: "MW"}) or {column: <pair '
        "column>} (its group mean).",
    )

    _units = field_validator("threshold", "normalize_by", mode="before")(_require_unit)
    side: Side | None = Field(
        default=None, description="Signatures: which values (default primary)."
    )
    year_start_month: Annotated[int, Field(ge=1, le=12)] | None = Field(
        default=None,
        description="Annual metrics: first month of the year (1 = calendar, 10 = water year).",
    )
    exceedance: tuple[float, float] | None = Field(
        default=None, description="fdc_slope: exceedance probabilities [e1, e2], 0 < e1 < e2 < 1."
    )
    estimator: Literal["standard", "fair"] | None = Field(
        default=None,
        description="crps: standard (default) or fair (unbiased for the ensemble size).",
    )
    interval: tuple[float, float] | None = Field(
        default=None,
        description="Interval metrics: the central interval's quantile levels, default "
        "[0.05, 0.95].",
    )
    scale: ColumnRef | None = Field(
        default=None,
        description="mase: the column holding each pair's naive scale (a naive_scale field).",
    )
    rank: Annotated[int, Field(ge=0)] | None = Field(
        default=None,
        description="rank_frequency: the observation's rank among ensemble members "
        "(0 = below all members).",
    )
    sr: Annotated[float, Field(ge=0)] | None = Field(
        default=None, description="KGE family: weight on the correlation term (default 1)."
    )
    sa: Annotated[float, Field(ge=0)] | None = Field(
        default=None, description="KGE family: weight on the variability term (default 1)."
    )
    sb: Annotated[float, Field(ge=0)] | None = Field(
        default=None, description="KGE family: weight on the bias term (default 1)."
    )
    log: bool | None = Field(
        default=None,
        description="fdc_slope: the log form (default true) or the linear slope (false).",
    )
    missing_day_threshold: Annotated[float, Field(ge=0, le=1)] | None = Field(
        default=None,
        description="Timing metrics: the fraction of a year's days that may be missing "
        "before the year is left out (default 0.1).",
    )
    values: ValuesRef | None = Field(
        default=None,
        description="First stage only: compute the metric on these pair columns instead of "
        "primary_value and secondary_value, e.g. change fields for ramps.",
    )

    @model_validator(mode="after")
    def _check_probabilities(self) -> MetricSpec:
        for name in ("exceedance", "interval"):
            pair = getattr(self, name)
            if pair is not None and not 0 < pair[0] < pair[1] < 1:
                raise ValueError(f"{name} must be two probabilities with 0 < lo < hi < 1")
        return self

    def given_params(self) -> set[str]:
        """Names of the metric options set in the spec."""
        return {f for f in _PARAM_FIELDS if getattr(self, f) is not None}

    def to_params(self) -> Params:
        """Return the metric's parameters for the engine."""
        p: dict[str, Any] = {}
        t = self.threshold
        if isinstance(t, ColumnRef):
            p["threshold_column"] = t.column
        elif isinstance(t, ObsQuantile):
            p["threshold_quantile"] = t.primary_quantile
        elif t is not None:  # in the evaluation unit once the plan is built
            p["threshold"] = t.value
        nb = self.normalize_by
        if isinstance(nb, ColumnRef):
            p["normalize_by"], p["normalize_column"] = "column", nb.column
        elif isinstance(nb, str):
            p["normalize_by"] = _NORMALIZERS[nb]
        elif nb is not None:
            p["normalize_by"], p["normalize_value"] = "value", nb.value
        if self.side is not None:
            p["side"] = SIDE_COLUMNS[self.side]
        if self.year_start_month is not None:
            p["year_start_month"] = self.year_start_month
        if self.exceedance is not None:
            p["exceedance"] = tuple(self.exceedance)
        if self.estimator is not None:
            p["estimator"] = self.estimator
        if self.interval is not None:
            p["interval"] = tuple(self.interval)
        if self.rank is not None:
            p["rank"] = self.rank
        if self.scale is not None:
            p["scale_column"] = self.scale.column
        if self.log is not None:
            p["log"] = self.log
        if self.missing_day_threshold is not None:
            p["missing_day_threshold"] = self.missing_day_threshold
        if any(w is not None for w in (self.sr, self.sa, self.sb)):
            p["kge_weights"] = tuple(1.0 if w is None else w for w in (self.sr, self.sa, self.sb))
        return Params(**p)

    @property
    def transform_name(self) -> Transform | None:
        """Return the transform, without its offset."""
        t = self.transform
        return t.name if isinstance(t, TransformWithOffset) else t

    def transform_key(self) -> TransformKey | None:
        """Return the transform for the engine (an offset in the evaluation unit)."""
        t = self.transform
        if isinstance(t, TransformWithOffset):
            return OffsetTransform(t.name, t.offset.value)
        return t

    def to_request(self) -> MetricRequest:
        """Return the metric request for the engine (values in the evaluation unit)."""
        return MetricRequest(self.name, self.transform_key(), self.to_params())

    def output_name(self) -> str:
        """Column name for this metric's result."""
        if self.as_ is not None:
            return self.as_
        if self.of is not None:
            return f"{self.name}_{self.of}"
        name = self.name
        metric = METRICS.get(name)
        if metric is not None and "side" in metric.params:
            name = f"{name}_{self.side or 'primary'}"
        if self.values is not None:
            name = f"{name}_{self.values.primary}"
        if self.transform_name is not None:
            return f"{name}_{self.transform_name.value}"
        return name


def quantile_suffix(q: float) -> str:
    """Column suffix for a bootstrap quantile: 0.05 -> ``q05``, 0.975 -> ``q97_5``."""
    pct = round(q * 100, 6)
    if pct.is_integer():
        return f"q{int(pct):02d}"
    return "q" + str(pct).replace(".", "_")


class Bootstrap(_Model):
    """Bootstrap confidence intervals for a stage's metrics (shared resamples).

    Every group is resampled ``reps`` times, the same way for every metric in the
    stage. Results depend only on ``seed`` and the group's key, never on how the work
    was split up.
    """

    method: Literal["iid", "circular_block", "stationary", "year_block"] = Field(
        default="circular_block",
        description="iid resamples pairs; circular_block resamples fixed-length blocks "
        "(wrapping around); stationary uses random block lengths with mean `block`; "
        "year_block (alias gumboot) resamples whole (water) years, keeping each year's "
        "seasonality.",
    )

    @field_validator("method", mode="before")
    @classmethod
    def _method_alias(cls, v: Any) -> Any:
        return "year_block" if v == "gumboot" else v

    block: Duration | MonthsBlock | Annotated[int, Field(ge=1)] | None = Field(
        default=None,
        description="Block length for block methods. Whole days (e.g. P7D) or calendar "
        "months (e.g. P1M, P3M) resample those periods of value_time, empty ones included, "
        "and can come from stored state; another duration (e.g. PT12H) is converted to "
        "rows from each group's time span; a number is rows.",
    )
    reps: Annotated[int, Field(ge=10, le=10_000)] = 1000
    seed: Annotated[int, Field(ge=0)] = 0
    quantiles: list[Annotated[float, Field(gt=0, lt=1)]] = Field(
        default_factory=lambda: [0.05, 0.95], min_length=1
    )
    sort_by: list[Identifier] = Field(
        default_factory=lambda: ["value_time"],
        description="Pair columns that order each group before resampling blocks.",
    )
    min_count: Annotated[int, Field(ge=2)] = Field(
        default=30,
        description="Groups with fewer forecast cases (pairs, for deterministic forecasts) "
        "get null intervals.",
    )
    year_start_month: Annotated[int, Field(ge=1, le=12)] = Field(
        default=10,
        description="year_block: the month years start in (10 = US water year; 1 = calendar).",
    )
    min_years: Annotated[int, Field(ge=2)] = Field(
        default=5, description="year_block: groups spanning fewer years get null intervals."
    )

    @model_validator(mode="after")
    def _check(self) -> Bootstrap:
        if self.method == "year_block":
            if self.block is not None:
                raise ValueError("year_block resamples whole years; it takes no 'block'")
        elif self.method != "iid" and self.block is None:
            raise ValueError(f"bootstrap method {self.method!r} needs 'block'")
        if isinstance(self.block, str):
            months = re.fullmatch(MONTHS_PATTERN, self.block) is not None
            if (
                parse_months(self.block) if months else parse_duration(self.block).total_seconds()
            ) <= 0:
                raise ValueError("bootstrap block must be positive")
        suffixes = [quantile_suffix(q) for q in self.quantiles]
        if len(suffixes) != len(set(suffixes)):
            raise ValueError(f"bootstrap quantiles must be distinct: {self.quantiles}")
        return self

    def block_periods(self) -> tuple[Literal["day", "month"], int] | None:
        """Return the block as calendar periods of value_time: ``("day", 7)``, ``("month", 1)``.

        None for iid and year_block, a block of rows, or a duration that isn't whole days.
        """
        block = self.block
        if not isinstance(block, str) or self.method in ("iid", "year_block"):
            return None
        if re.fullmatch(MONTHS_PATTERN, block):
            return "month", parse_months(block)
        seconds = parse_duration(block).total_seconds()
        if seconds % 86400:
            return None
        return "day", int(seconds // 86400)

    def columns(self, metric_column: str) -> list[str]:
        """Interval columns for one metric, e.g. ``kge_q05``, ``kge_q95``."""
        return [f"{metric_column}_{quantile_suffix(q)}" for q in self.quantiles]


class Skill(_Model):
    """Skill scores of each secondary dataset against a reference dataset.

    For each listed metric, ``<metric>_skill = 1 - score / score_of_reference`` (or
    ``<metric>_ratio = score / score_of_reference`` with ``form: ratio``), comparing rows
    that match on every group_by column except ``secondary_dataset``.
    """

    reference: str = Field(description="A secondary input dataset, e.g. a persistence forecast.")
    metrics: list[Identifier] = Field(
        min_length=1,
        description="Output columns of error metrics in this stage (a metric's alias names "
        "its canonical column).",
    )

    @field_validator("metrics")
    @classmethod
    def _canonical(cls, v: list[str]) -> list[str]:
        return [canonical_column(c) for c in v]

    form: Literal["skill", "ratio"] = "skill"
    matched: bool = Field(
        default=False,
        description="Compare each dataset with the reference only on observations both "
        "forecast (same observed series and value_time within a group), recomputing both "
        "scores on those; the metric columns themselves stay unmatched.",
    )
    test: Literal["dm"] | None = Field(
        default=None,
        description="dm: Diebold-Mariano test of each dataset's loss against the reference's "
        "on the observations both forecast; adds dm_statistic (negative: the dataset's "
        "errors are smaller) and dm_p_value (two-sided).",
    )
    loss: Literal["squared", "absolute"] = Field(
        default="squared", description="The test's loss: squared or absolute error."
    )
    lag: Annotated[int, Field(ge=0)] | None = Field(
        default=None,
        description="Autocovariance lags in the test's (Newey-West) variance, in time steps; "
        "default floor(4 (n / 100)^(2/9)).",
    )

    def column(self, metric_column: str) -> str:
        """Return the skill column for one metric column."""
        return f"{metric_column}_{self.form}"

    def test_columns(self) -> list[str]:
        """Columns the test adds (none without one)."""
        return ["dm_statistic", "dm_p_value"] if self.test else []


#: Column a first stage with guards adds, explaining nulled metrics.
REASON_COLUMN = "null_reason"


#: The column labelling a reduced pair's statistic (``reduce``).
STAT = "stat"
#: Pair columns a reduce keeps in its windows' keys when present, so it never merges
#: different forecasts: the predicted dataset and series, and an ensemble's members.
REDUCE_KEYS = ("secondary_dataset", "secondary_series_key", "member", "quantile")


class Reduce(_Model):
    """Reduce each window of pairs to a statistic before the metrics (e.g. its peak).

    The pairs are grouped by ``by`` (a forecast window: series, reference time, lead-time
    bin...), and each window becomes one pair per statistic in ``stats``: its mean, min,
    max or sum of observed and of predicted values, labelled in a ``stat`` column. Metrics
    then score those reduced pairs. Applied after ``filter`` and conditions, before a
    rollup's copies.
    """

    by: list[Identifier] = Field(
        min_length=1,
        description="Columns that make a window; must include primary_series_key. The predicted "
        "dataset and series, members, quantiles and conditions fields are added.",
    )
    stats: list[Literal["mean", "min", "max", "sum"]] = Field(min_length=1)

    @field_validator("stats")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("reduce stats repeat a statistic")
        return v


class Guard(_Model):
    """Null metrics where a group's data is too thin to trust them, and say why.

    Conditions apply to each group's observations; ``metrics`` (output names) limits
    which metrics a guard nulls (default: all of the stage's). A tripped guard nulls the
    metrics and their bootstrap intervals, and the stage's ``null_reason`` column says
    why, e.g. ``nse, kge: obs std 0.004 < 0.01``.
    """

    metrics: list[str] | None = Field(default=None, description="Metrics it guards (default all).")

    @field_validator("metrics")
    @classmethod
    def _canonical(cls, v: list[str] | None) -> list[str] | None:
        return None if v is None else [canonical_column(c) for c in v]

    min_count: Annotated[int, Field(ge=1)] | None = Field(
        default=None, description="At least this many pairs."
    )
    min_obs_mean: float | None = Field(
        default=None, description="Observed mean at least this (e.g. for relative metrics)."
    )
    min_obs_std: Annotated[float, Field(ge=0)] | None = Field(
        default=None,
        description="Observed standard deviation at least this (e.g. for NSE, KGE, "
        "correlation, which divide by the observations' variability).",
    )

    @model_validator(mode="after")
    def _check(self) -> Guard:
        if self.min_count is None and self.min_obs_mean is None and self.min_obs_std is None:
            raise ValueError("a guard needs min_count, min_obs_mean or min_obs_std")
        return self


#: The column naming each row's grouping set, in a stage with ``rollup`` or ``grouping_sets``.
GROUPING = "grouping"
#: The label of a rolled-up column in rows aggregated across it.
ALL_LABEL = "(all)"
#: Calendar parts whose labels name their year, and the parts that are those years.
_YEAR_QUALIFIED = ("quarter", "month")
_YEARS = ("year", "water_year")


class Stage(_Model):
    """One aggregation stage."""

    group_by: list[str] = Field(default_factory=list)
    rollup: list[Identifier] | None = Field(
        default=None,
        description=(
            "First stage only: also aggregate across these columns, coarsest first, as SQL "
            "ROLLUP: rollup [water_year, quarter] gives rows by quarter, by water year and "
            "for all. Each row's set is named in a 'grouping' column, and a column it's "
            "aggregated across reads '(all)'. Folded into grouping_sets and group_by."
        ),
    )
    grouping_sets: list[list[Identifier]] | None = Field(
        default=None,
        description=(
            "First stage only: the general form of rollup, each set the columns a row is "
            "grouped by (besides group_by); [] aggregates across all of them."
        ),
    )
    metrics: list[MetricSpec] = Field(min_length=1)
    bootstrap: Bootstrap | None = Field(
        default=None, description="First stage only: bootstrap intervals for its metrics."
    )
    skill: Skill | None = Field(
        default=None, description="First stage only: skill scores against a reference dataset."
    )
    guards: list[Guard] = Field(
        default_factory=list,
        description="First stage only: null metrics on too little or too flat data, with "
        "a null_reason column.",
    )

    @model_validator(mode="before")
    @classmethod
    def _fold_rollup(cls, data: Any) -> Any:
        """Fold ``rollup`` into ``grouping_sets``, and the sets' columns into ``group_by``.

        Everything downstream then sees a stage's full keys in ``group_by`` (its own
        columns, the sets' columns and ``grouping``) and the sets in ``grouping_sets``.
        """
        if not isinstance(data, dict):
            return data
        rollup, sets = data.get("rollup"), data.get("grouping_sets")
        if rollup is not None:
            if sets is not None:
                raise ValueError("give rollup or grouping_sets, not both")
            if not isinstance(rollup, list) or not rollup:
                raise ValueError("rollup needs at least one column")
            sets = [rollup[:i] for i in range(len(rollup), -1, -1)]
            data = {**data, "rollup": None, "grouping_sets": sets}
        if not sets or not isinstance(sets, list):
            return data
        group_by = list(data.get("group_by") or [])
        folded = GROUPING in group_by  # already folded (a dumped spec read again)
        columns = list(dict.fromkeys(c for s in sets if isinstance(s, list) for c in s))
        if not folded:
            both = [c for c in columns if c in group_by]
            if both:
                raise ValueError(
                    f"{both} in both group_by and the rollup: a rolled-up column goes only "
                    "in the rollup"
                )
            group_by += [*columns, GROUPING]
        return {**data, "group_by": group_by}

    @model_validator(mode="after")
    def _check_sets(self) -> Stage:
        if self.grouping_sets is None:
            return self
        sets = [tuple(s) for s in self.grouping_sets]
        if len(set(sets)) != len(sets):
            raise ValueError("grouping_sets repeat a set")
        if any(len(set(s)) != len(s) for s in sets):
            raise ValueError("a grouping set repeats a column")
        return self

    def rolled_up(self) -> list[str]:
        """Return the columns some grouping set aggregates across (in group_by order)."""
        if not self.grouping_sets:
            return []
        columns = {c for s in self.grouping_sets for c in s}
        return [c for c in self.group_by if c in columns]

    def set_names(self) -> list[str]:
        """Return each grouping set's name, for the ``grouping`` column.

        Its finest column (the last), or ``all`` for the empty set; when two sets share a
        last column, their columns joined with ``+``.
        """
        sets = self.grouping_sets or []
        last = [s[-1] if s else "all" for s in sets]
        if len(set(last)) == len(last):
            return last
        return ["+".join(s) if s else "all" for s in sets]

    def finest_set(self) -> list[str] | None:
        """Return the set holding every rolled-up column (None: no such set, or no sets).

        Rows of every coarser set can be merged from its rows (mergeable state).
        """
        columns = self.rolled_up()
        sets = self.grouping_sets or []
        return next((s for s in sets if set(s) == set(columns)), None)

    def rolls_up_from_state(self) -> bool:
        """Whether a run merging this stage from state can build its coarser sets there.

        It needs a finest set (holding every rolled-up column) and no DM test, whose state
        follows each group's time series.
        """
        test = self.skill is not None and self.skill.test is not None
        return self.finest_set() is not None and not test

    def per_series(self) -> bool:
        """Whether every group is within one observed series (so shards own their groups)."""
        return (
            "primary_series_key" in self.group_by and "primary_series_key" not in self.rolled_up()
        )

    @field_validator("metrics", mode="before")
    @classmethod
    def _names_to_specs(cls, v: Any) -> Any:
        return [{"name": m} if isinstance(m, str) else m for m in v] if isinstance(v, list) else v

    def output_columns(self) -> list[str]:
        """Columns this stage produces (metrics, then their bootstrap intervals)."""
        metrics = [m.output_name() for m in self.metrics]
        intervals = (
            [c for m in self.bootstrapped() for c in self.bootstrap.columns(m.output_name())]
            if self.bootstrap
            else []
        )
        skills = [self.skill.column(m) for m in self.skill.metrics] if self.skill else []
        skills += self.skill.test_columns() if self.skill else []
        reason = [REASON_COLUMN] if self.guards else []
        return [*self.group_by, *metrics, *intervals, *skills, *reason]

    def bootstrapped(self) -> list[MetricSpec]:
        """Metrics that get bootstrap intervals (all but ``count``, which can't vary)."""
        return [m for m in self.metrics if m.name != "count"] if self.bootstrap else []


class Output(_Model):
    """What the evaluation returns."""

    include_pairs: bool = Field(default=False, description="Also return the paired rows.")
    unit: str | None = Field(
        default=None,
        description="Unit to evaluate in (e.g. 'm3/s'): primary and secondary values are "
        "converted to it before fields and metrics, and thresholds and normalizers (which "
        "carry their own unit) are converted to it too. Default: the observed dataset's "
        "unit. An input filter's numbers are in its variable's canonical unit.",
    )
    pairs: Literal["all", "none"] | None = Field(
        default=None,
        description=(
            "Whether a saved evaluation's runs keep their pairs in the project lake: "
            "'all' or 'none'. Unset uses the platform default. First-stage rows and "
            "results are always kept; without pairs, drill-down re-pairs on request."
        ),
    )
    geometry: Identifier | None = Field(
        default=None,
        description=(
            "Give each result row a geometry, by a column of the final stage's group_by: a "
            "within field (its polygon) or primary_series_key (the observed series' location). "
            "Looked up once per result row; results then come as GeoJSON on request."
        ),
    )
    attributes: list[ColumnName] = Field(
        default_factory=list,
        description=(
            "With geometry: attributes of the same place to add as result columns, e.g. a "
            "basin's name, from the layer the geometry comes from."
        ),
    )


class EvaluationSpec(_Model):
    """An evaluation spec (version 1)."""

    version: Literal[1]
    name: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
    inputs: Inputs
    pair: Pairing
    fields: list[FieldSpec] = Field(default_factory=list)
    filter: str | None = Field(default=None, description="SQL expression applied to pairs.")
    reduce: Reduce | None = Field(
        default=None, description="Reduce each window of pairs to statistics before metrics."
    )
    aggregate: list[Stage] = Field(min_length=1)
    output: Output = Output()

    @field_validator("filter")
    @classmethod
    def _check_filter(cls, v: str | None) -> str | None:
        return None if v is None else check_expression(v)

    def reduce_keys(self) -> list[str]:
        """Return a reduce's window keys: ``by``, then conditions fields (not the rest).

        The pair columns in :data:`REDUCE_KEYS` are added by the engine when present.
        """
        assert self.reduce is not None
        conditions = [f.name for f in self.fields if isinstance(f, ConditionsField)]
        return list(dict.fromkeys([*self.reduce.by, *conditions]))

    def _check_reduce(self) -> None:
        if self.reduce is None:
            return
        if "primary_series_key" not in self.reduce.by:
            raise ValueError(
                "reduce.by must include primary_series_key: windows stay within one observed "
                "series (runs split by series)"
            )
        if STAT in [f.name for f in self.fields]:
            raise ValueError(f"no field may be named {STAT!r}: a reduce labels pairs with it")
        carried = {f.name for f in self.fields if isinstance(f, SERIES_CONSTANT)}
        kept = {*self.reduce_keys(), *REDUCE_KEYS, STAT, *carried}
        first = self.aggregate[0]
        lost = [c for c in first.group_by if c not in kept and c != GROUPING]
        if lost:
            raise ValueError(
                f"aggregate[0] groups by {lost}, which a reduce doesn't keep: add them to "
                "reduce.by (it keeps by, stat, the forecast's keys and per-series fields)"
            )
        timed = [m.output_name() for m in first.metrics if METRICS[m.name].needs_time]
        if timed:
            raise ValueError(
                f"aggregate[0]: {timed} read each pair's time, which a reduced pair doesn't "
                "have (it takes its window's first)"
            )
        if first.skill is not None and first.skill.test is not None:
            raise ValueError("a reduce can't be combined with a skill test")

    def _check_calendar_sets(self) -> None:
        """Refuse sets that keep a quarter or month but drop its year: they repeat rows.

        Quarter and month labels name their year (``2025-Q1``), so a set with ``quarter``
        but not the ``water_year`` beside it in the rollup gives the same groups as the set
        with both: e.g. rollup [quarter, water_year] in the wrong order.
        """
        stage = self.aggregate[0]
        if not stage.grouping_sets:
            return
        calendar = {f.name: f for f in self.fields if isinstance(f, CalendarField)}
        rolled = set(stage.rolled_up())
        years = [c for c in rolled if c in calendar and calendar[c].part in _YEARS]
        for s in stage.grouping_sets:
            fine = [c for c in s if c in calendar and calendar[c].part in _YEAR_QUALIFIED]
            dropped = [y for y in years if y not in s]
            if fine and dropped:
                raise ValueError(
                    f"aggregate[0]: the set {s} keeps {fine[0]} but not {dropped[0]}; "
                    f"{fine[0]} labels already name their year, so it repeats the set with "
                    f"both (list the rollup coarsest first: [{dropped[0]}, {fine[0]}])"
                )

    def _check_skill(self, where: str, i: int, stage: Stage) -> None:
        skill = stage.skill
        assert skill is not None
        if i > 0:
            raise ValueError(f"{where}: 'skill' is only allowed in the first stage")
        if "secondary_dataset" not in stage.group_by:
            raise ValueError(f"{where}: skill scores need secondary_dataset in group_by")
        predicted = [p.label for p in self.inputs.secondary]
        if skill.reference not in predicted:
            raise ValueError(
                f"{where}: skill reference {skill.reference!r} is not a secondary input "
                f"({', '.join(predicted)})"
            )
        by_column = {m.output_name(): m for m in stage.metrics}
        for col in skill.metrics:
            m = by_column.get(col)
            if m is None:
                raise ValueError(
                    f"{where}: skill metric {col!r} is not a column of this stage"
                    + _alias_hint(col)
                )
            if not METRICS[m.name].skill:
                raise ValueError(
                    f"{where}: skill needs an error score (lower is better, 0 is perfect); "
                    f"{m.name!r} isn't one"
                )

    @model_validator(mode="after")
    def _check_stages(self) -> EvaluationSpec:
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise ValueError(f"field names must be unique: {names}")
        if GROUPING in names:
            raise ValueError(f"no field may be named {GROUPING!r}: rollups name their sets in it")
        for i, stage in enumerate(self.aggregate[1:], start=1):
            if stage.grouping_sets is not None:
                raise ValueError(f"aggregate[{i}]: rollup and grouping_sets are first stage only")
        first = self.aggregate[0]
        if first.grouping_sets is None and GROUPING in first.group_by:
            raise ValueError(f"aggregate[0]: {GROUPING!r} comes from rollup or grouping_sets")
        self._check_calendar_sets()
        self._check_reduce()
        for i, stage in enumerate(self.aggregate):
            where = f"aggregate[{i}]"
            for m in stage.metrics:
                if i == 0:
                    if m.name not in METRICS:
                        raise ValueError(
                            f"{where}: unknown metric {m.name!r}; available: {', '.join(METRICS)}"
                        )
                    if m.of is not None:
                        raise ValueError(f"{where}: 'of' is only allowed in later stages")
                    metric = METRICS[m.name]
                    extra = m.given_params() - metric.params
                    if extra:
                        raise ValueError(
                            f"{where}: {m.name!r} doesn't take {sorted(extra)}"
                            + (f"; it takes {sorted(metric.params)}" if metric.params else "")
                        )
                    missing = metric.required - m.given_params()
                    if missing:
                        raise ValueError(f"{where}: {m.name!r} needs {sorted(missing)}")
                    if m.transform is not None and not metric.transforms:
                        raise ValueError(f"{where}: {m.name!r} doesn't take a transform")
                else:
                    if m.name not in SUMMARIES:
                        raise ValueError(
                            f"{where}: unknown summary {m.name!r}; "
                            f"available: {', '.join(SUMMARIES)}"
                        )
                    if m.of is None:
                        raise ValueError(f"{where}: summary {m.name!r} needs 'of'")
                    if m.transform is not None:
                        raise ValueError(f"{where}: 'transform' is only allowed in the first stage")
                    if m.values is not None:
                        raise ValueError(f"{where}: 'values' is only allowed in the first stage")
                    if m.given_params():
                        raise ValueError(
                            f"{where}: metric options {sorted(m.given_params())} are only "
                            "allowed in the first stage"
                        )
            if i > 0 and stage.bootstrap is not None:
                raise ValueError(f"{where}: 'bootstrap' is only allowed in the first stage")
            if i > 0 and stage.guards:
                raise ValueError(f"{where}: 'guards' are only allowed in the first stage")
            stage_metrics = {m.output_name() for m in stage.metrics}
            for guard in stage.guards:
                unknown = sorted(set(guard.metrics or []) - stage_metrics)
                if unknown:
                    raise ValueError(
                        f"{where}: guard metrics {unknown} aren't in this stage"
                        + "".join(_alias_hint(c) for c in unknown)
                    )
            if stage.skill is not None:
                self._check_skill(where, i, stage)
            cols = stage.output_columns()
            if len(cols) != len(set(cols)):
                raise ValueError(f"{where}: duplicate output columns {cols}")
            if i > 0:
                prev = self.aggregate[i - 1]
                if not set(stage.group_by) <= set(prev.group_by):
                    raise ValueError(
                        f"{where}: group_by must be a subset of the previous stage's group_by"
                    )
                prev_cols = set(prev.output_columns())
                for m in stage.metrics:
                    if m.of not in prev_cols:
                        raise ValueError(
                            f"{where}: {m.of!r} is not a column of the previous stage"
                            + _alias_hint(m.of)
                        )
        layers: dict[str, list[str]] = {}
        for f in self.fields:
            if isinstance(f, WithinField):
                layers.setdefault(f.layer, []).append(f.name)
        for layer, names in layers.items():
            if len(names) > 1:
                raise ValueError(
                    f"fields {names} both look up layer {layer!r}: use one within field per "
                    "layer, labelled by its key, and add the polygon's other attributes to "
                    "the results with output.geometry and output.attributes"
                )
        geometry = self.output.geometry
        if geometry is not None:
            within = {f.name for f in self.fields if isinstance(f, WithinField)}
            if geometry != "primary_series_key" and geometry not in within:
                raise ValueError(
                    f"output.geometry: {geometry!r} must be primary_series_key or a within field"
                )
            if geometry not in self.aggregate[-1].group_by:
                raise ValueError(
                    f"output.geometry: {geometry!r} must be in the final stage's group_by"
                )
        if self.output.attributes and geometry is None:
            raise ValueError("output.attributes needs output.geometry (the place they're about)")
        return self


def _alias_hint(column: str | None) -> str:
    """Return a hint for a column named by a metric alias (columns carry canonical names)."""
    if column is None or canonical_column(column) == column:
        return ""
    return f" (results use canonical names: {canonical_column(column)!r}?)"


#: What may follow a metric's name in its result columns: a transform, a summary's
#: bootstrap quantile, skill, a side, or a Diebold-Mariano column.
_COLUMN_SUFFIX = re.compile(
    r"_(?:log|sqrt|square|cube|exp|reciprocal|abs|skill|ratio|primary|secondary|q\d+(?:_\d+)?"
    r"|dm_[a-z_]+)(?:_.*)?"
)


def canonical_column(column: str) -> str:
    """Return ``column`` under its metric's canonical name if it uses an alias.

    ``rmse``, ``kge_q05`` and ``nse_log`` become ``root_mean_square_error``,
    ``kling_gupta_efficiency_q05`` and ``nash_sutcliffe_efficiency_log``; other names are
    unchanged.
    """
    if column in ALIASES:
        return ALIASES[column]
    for alias in sorted(ALIASES, key=len, reverse=True):
        rest = column[len(alias) :]
        if column.startswith(alias) and _COLUMN_SUFFIX.fullmatch(rest):
            return ALIASES[alias] + rest
    return column
