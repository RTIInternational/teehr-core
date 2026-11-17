"""Logical plan: a spec resolved against a catalog and checked before execution.

Ported from VeriRace (``verirace/engine/plan.py`` at 668ab7d).

Planning catches every problem that doesn't need the data itself (missing datasets,
wrong roles, unit mismatches, incompatible statistics) so backends can assume a valid
plan. Backends turn a :class:`Plan` into Arrow results.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from teehr_core.durations import parse_duration
from teehr_core.engine.catalog import Catalog, Dataset
from teehr_core.model.periods import INSTANT, Period, PeriodError, window_sql
from teehr_core.model.units import IDENTITY, Conversion, UnitError, UnitRegistry, conversion_between
from teehr_core.spec import (
    AttributeField,
    CalendarField,
    ChangeField,
    EvaluationSpec,
    EventsField,
    GeneratedInput,
    LeadTimeBinsField,
    NaiveScaleField,
    PercentileField,
    Quantity,
    WithinField,
)
from teehr_core.spec.expressions import to_duckdb
from teehr_core.spec.models import TransformWithOffset

__all__ = [
    "Plan",
    "PlanError",
    "PlannedInput",
    "SidePeriods",
    "build_plan",
    "input_filter_sql",
    "labels_name",
    "quantities_in",
]


class PlanError(ValueError):
    """The spec can't be executed against this catalog."""


@dataclass(frozen=True)
class SidePeriods:
    """The periods (ADR-0009) one side's values cover: what its windows are made of."""

    #: Every period the side's values use, by name.
    periods: dict[str, Period]
    #: The dataset's period: values of series that name none cover it.
    default: str | None

    @property
    def by_series(self) -> bool:
        """Whether the period differs by series (windows then look up each series')."""
        return len(self.periods) > 1

    @property
    def windowed(self) -> bool:
        """Whether some values cover more than an instant."""
        return any(p.form != "instant" for p in self.periods.values())

    def window_sql(self, row: str, name: str | None = None) -> tuple[str, str]:
        """Return SQL for the window of a value in row alias ``row`` (periods.window_sql).

        ``name`` is the SQL for the value's period name, needed when :attr:`by_series`.
        """
        return window_sql(self.periods, row, name)

    def same_by_value_time(self, other: SidePeriods) -> bool:
        """Whether values pair with ``other``'s exactly when their value times are equal.

        True when both sides cover one and the same instant or duration period: equal
        value times then mean equal windows, so pairing joins on the stored value time,
        which keeps file pruning. A calendar value may sit anywhere in its window, so
        calendar periods always pair on windows.
        """
        return (
            len(self.periods) == 1
            and self.periods.keys() == other.periods.keys()
            and next(iter(self.periods.values())).form in ("instant", "duration")
        )


@dataclass(frozen=True)
class PlannedInput:
    """A predicted dataset with what's needed to pair it."""

    dataset: Dataset
    filter: str | None
    conversion: Conversion
    #: The periods its values cover; a generated input's are the observations'.
    periods: SidePeriods
    #: For a generated input: how, and the forecast whose grid it copies. Its ``dataset``
    #: describes the generated values (observed units and statistic); the backend builds
    #: its table.
    generate: GeneratedInput | None = None
    like: Dataset | None = None
    #: A blend's climatology, and the conversion of its values to the observed unit.
    climatology: Dataset | None = None
    climatology_conversion: Conversion = IDENTITY


@dataclass(frozen=True)
class Plan:
    """A validated evaluation ready for a backend."""

    spec: EvaluationSpec
    observed: Dataset
    observed_filter: str | None
    predicted: tuple[PlannedInput, ...]
    relation: Dataset | None
    on_attribute: str | None
    #: Datasets fields look up, by field name (e.g. naive scales, in the evaluation unit).
    aux: dict[str, Dataset] = field(default_factory=dict)
    #: The unit pairs and metrics are in: the spec's ``output.unit``, else the observed
    #: dataset's; ``observed_conversion`` takes observed values there.
    unit: str | None = None
    observed_conversion: Conversion = IDENTITY
    #: The periods the observations cover (persistence uses an observation once its
    #: window has ended).
    observed_periods: SidePeriods = field(
        default_factory=lambda: SidePeriods({INSTANT: Period(INSTANT, "instant")}, INSTANT)
    )


def _timeseries(catalog: Catalog, name: str, roles: tuple[str, ...], what: str) -> Dataset:
    if name not in catalog:
        raise PlanError(
            f"{what} dataset {name!r} is not in the catalog ({', '.join(catalog.names())})"
        )
    ds = catalog[name]
    if ds.role not in roles:
        raise PlanError(f"{what} dataset {name!r} has role {ds.role!r}; expected one of {roles}")
    return ds


def build_plan(spec: EvaluationSpec, catalog: Catalog) -> Plan:
    """Resolve ``spec`` against ``catalog``.

    Raises
    ------
    PlanError
        If the evaluation can't run against this catalog.
    """
    observed = _timeseries(catalog, spec.inputs.primary.dataset, ("observed",), "observed")
    assert observed.unit is not None
    unit = spec.output.unit or observed.unit
    try:
        observed_conversion = conversion_between(catalog.units, observed.unit, unit)
    except UnitError as exc:
        raise PlanError(f"output.unit {unit!r}: {exc}") from exc
    spec = quantities_in(spec, catalog.units, unit)

    relation = None
    if spec.pair.relation is not None:
        if spec.pair.relation not in catalog:
            raise PlanError(f"relation {spec.pair.relation!r} is not in the catalog")
        relation = catalog[spec.pair.relation]
        if relation.kind != "crosswalk":
            raise PlanError(f"relation {relation.name!r} must be an auxiliary crosswalk dataset")

    needs_reference_time = any(
        isinstance(f, LeadTimeBinsField)
        or (isinstance(f, CalendarField) and f.of == "reference_time")
        for f in spec.fields
    )
    observed_periods = _side_periods(catalog, observed)
    predicted = []
    for ref in spec.inputs.secondary:
        if isinstance(ref, GeneratedInput):
            predicted.append(
                _generated(catalog, ref, observed, observed_periods, needs_reference_time, unit)
            )
            continue
        ds = _timeseries(catalog, ref.dataset, ("simulated", "forecast"), "predicted")
        assert ds.unit is not None
        try:
            conversion_between(catalog.units, ds.unit, observed.unit)  # same dimension
            conversion = conversion_between(catalog.units, ds.unit, unit)
        except UnitError as exc:
            raise PlanError(f"cannot pair {ds.name!r} with {observed.name!r}: {exc}") from exc
        _same_variable(ds, observed, ds.name)
        if ds.statistic != observed.statistic:
            raise PlanError(
                f"cannot pair {ds.name!r} ({ds.statistic}) with {observed.name!r} "
                f"({observed.statistic}): statistics differ"
            )
        if spec.pair.on_attribute is not None:
            for side in (observed, ds):
                if side.series is None or spec.pair.on_attribute not in side.series.column_names:
                    raise PlanError(
                        f"{side.name!r} has no series attribute "
                        f"{spec.pair.on_attribute!r} to pair on"
                    )
        if needs_reference_time and "reference_time" not in ds.table.column_names:
            raise PlanError(f"lead_time_bins needs reference_time, which {ds.name!r} doesn't have")
        predicted.append(
            PlannedInput(
                ds, _filter_sql(catalog, ds, ref.filter), conversion, _side_periods(catalog, ds)
            )
        )

    aux = {
        f.name: _naive_scale(catalog, f, observed, unit)
        for f in spec.fields
        if isinstance(f, NaiveScaleField)
    }
    for f in spec.fields:
        if isinstance(f, AttributeField | WithinField):
            aux[f.name] = _labels(catalog, f, observed)
        elif isinstance(f, PercentileField):
            aux[f.name] = _percentile(catalog, f, observed, unit)
    labels = [p.dataset.name for p in predicted]
    if len(set(labels)) != len(labels):
        raise PlanError(f"predicted inputs must have distinct names, got {labels}")
    return Plan(
        spec=spec,
        observed=observed,
        observed_filter=_filter_sql(catalog, observed, spec.inputs.primary.filter),
        predicted=tuple(predicted),
        relation=relation,
        on_attribute=spec.pair.on_attribute,
        aux=aux,
        unit=unit,
        observed_conversion=observed_conversion,
        observed_periods=observed_periods,
    )


def input_filter_sql(
    text: str | None,
    units: UnitRegistry | None,
    variables: Mapping[str, str],
    variable: str | None,
    unit: str | None,
) -> str | None:
    """Return an input's ``filter`` as DuckDB SQL, its numbers read in the canonical unit.

    A number compared with ``value`` is in the unit of the dataset's variable (from the
    variables registry), whatever unit the dataset was stored in, so the same filter keeps
    the same rows of the same data stored in cfs or m3/s. A variable not in ``variables``
    (or no unit registry): the dataset's own unit.
    """
    canonical = variables.get(variable) if variable else None
    if text is None or canonical is None or unit is None or units is None or canonical == unit:
        return to_duckdb(text)
    try:
        conversion = conversion_between(units, unit, canonical)
    except UnitError as exc:
        raise PlanError(f"filter {text!r}: {exc}") from exc
    return to_duckdb(text, scale=conversion.scale, offset=conversion.offset)


def _filter_sql(catalog: Catalog, ds: Dataset, text: str | None) -> str | None:
    return input_filter_sql(text, catalog.units, catalog.variables, ds.variable, ds.unit)


def quantities_in(spec: EvaluationSpec, units: UnitRegistry | None, unit: str) -> EvaluationSpec:
    """Return ``spec`` with every value it gives in ``unit``.

    Thresholds, normalizers and transform offsets. A threshold on change fields (a ramp), a
    normalizer and an offset are differences or magnitudes, so they convert by scale alone:
    a 5 K ramp is a 5 Cel ramp, not a -268.15 Cel one.
    """
    changes = {f.name for f in spec.fields if isinstance(f, ChangeField)}

    def convert(q: Quantity, what: str, difference: bool = False) -> Quantity:
        try:
            c = conversion_between(units, q.unit, unit)
        except UnitError as exc:
            raise PlanError(f"{what} ({q.value} {q.unit}): {exc}") from exc
        return Quantity(value=q.value * c.scale + (0.0 if difference else c.offset), unit=unit)

    fields = [
        f.model_copy(update={"threshold": convert(f.threshold, f"field {f.name!r} threshold")})
        if isinstance(f, EventsField) and isinstance(f.threshold, Quantity)
        else f
        for f in spec.fields
    ]
    stages = []
    for i, stage in enumerate(spec.aggregate):
        metrics = []
        for m in stage.metrics:
            what = f"aggregate[{i}] metric {m.as_ or m.name!r}"
            update: dict[str, Any] = {}
            if isinstance(m.threshold, Quantity):
                ramp = m.values is not None and {m.values.primary, m.values.secondary} <= changes
                update["threshold"] = convert(m.threshold, f"{what} threshold", ramp)
            if isinstance(m.normalize_by, Quantity):
                update["normalize_by"] = convert(m.normalize_by, f"{what} normalize_by", True)
            if isinstance(m.transform, TransformWithOffset):
                offset = convert(m.transform.offset, f"{what} transform offset", True)
                update["transform"] = m.transform.model_copy(update={"offset": offset})
            metrics.append(m.model_copy(update=update) if update else m)
        stages.append(stage.model_copy(update={"metrics": metrics}))
    return spec.model_copy(update={"fields": fields, "aggregate": stages})


def _generated(
    catalog: Catalog,
    ref: GeneratedInput,
    observed: Dataset,
    observed_periods: SidePeriods,
    needs_reference_time: bool,
    unit: str,
) -> PlannedInput:
    """Plan a generated input, keyed by the observed series.

    Persistence comes from the observations; climatology from a stored climatology.
    """
    assert observed.unit is not None
    if ref.generate == "climatology":
        clim, conversion = _climatology(catalog, ref.source, observed, ref.label, unit)
        if needs_reference_time:
            raise PlanError(f"{ref.label!r} has no reference times, so no lead_time_bins")
        generated = replace(
            clim,
            name=ref.label,
            role="simulated",
            kind=None,
            variable=observed.variable,
            statistic=observed.statistic,
        )
        return PlannedInput(
            generated,
            _filter_sql(catalog, generated, ref.filter),
            conversion,
            observed_periods,
            generate=ref,
            like=clim,
        )
    like = _timeseries(catalog, ref.source, ("forecast",), "the 'like' of a generated")
    if "reference_time" not in like.table.column_names:
        raise PlanError(f"{ref.label!r} needs reference times, which {like.name!r} doesn't have")
    climatology, clim_conversion = None, IDENTITY
    if ref.generate == "blend":
        assert ref.from_ is not None
        climatology, clim_conversion = _climatology(
            catalog, ref.from_, observed, ref.label, observed.unit
        )
    generated = replace(
        like,
        name=ref.label,
        series=None,
        variable=observed.variable,
        unit=observed.unit,
        statistic=observed.statistic,
        period=observed.period,
    )
    # Persistence (and blend) values are in the observed unit: converted like observations.
    conversion = conversion_between(catalog.units, observed.unit, unit)
    return PlannedInput(
        generated,
        _filter_sql(catalog, generated, ref.filter),
        conversion,
        observed_periods,
        generate=ref,
        like=like,
        climatology=climatology,
        climatology_conversion=clim_conversion,
    )


def _side_periods(catalog: Catalog, ds: Dataset) -> SidePeriods:
    """Resolve the periods ``ds``'s values cover: its default and its series' own.

    Raises
    ------
    PlanError
        If a period is unknown, doesn't fit the statistic, or values would have none.
    """
    default = ds.period or (INSTANT if ds.statistic == "inst" else None)
    named: set[str] = set()
    unnamed = True  # some values take the default (series without a name, or not listed)
    if ds.series is not None and "period" in ds.series.column_names:
        column = ds.series["period"]
        named = {v for v in pc.unique(column).to_pylist() if v is not None}
        unnamed = column.null_count > 0 or not named
    if default is None and unnamed:
        raise PlanError(
            f"{ds.name!r} has {ds.statistic!r} values but no period saying what span each "
            "covers: give the dataset a period (e.g. hour_starting), or each series one"
        )
    names = named | ({default} if default is not None and unnamed else set())
    periods = {}
    for name in sorted(names):
        try:
            period = catalog.periods.get(name)
        except PeriodError as exc:
            raise PlanError(f"{ds.name!r}: {exc}") from exc
        if (period.form == "instant") != (ds.statistic == "inst"):
            raise PlanError(
                f"{ds.name!r}: {ds.statistic!r} values can't cover period {name!r} "
                f"({period.form}); only inst values cover an instant"
            )
        if period.form == "absolute":
            missing = {"period_start", "period_end"} - set(ds.table.column_names)
            if missing:
                raise PlanError(
                    f"{ds.name!r}: period {name!r} is absolute, so values need "
                    f"{', '.join(sorted(missing))}"
                )
        periods[name] = period
    return SidePeriods(periods, default)


def _climatology(
    catalog: Catalog, name: str, observed: Dataset, label: str, unit: str
) -> tuple[Dataset, Conversion]:
    """Return a generated input's climatology dataset and its conversion to ``unit``."""
    if name not in catalog:
        raise PlanError(f"climatology {name!r} is not in the catalog")
    clim = catalog[name]
    if clim.role != "auxiliary" or clim.kind != "climatology":
        raise PlanError(f"{name!r} is not a climatology dataset")
    assert clim.unit is not None
    _same_variable(clim, observed, label)
    try:
        return clim, conversion_between(catalog.units, clim.unit, unit)
    except UnitError as exc:
        raise PlanError(f"cannot pair {label!r} with {observed.name!r}: {exc}") from exc


def _same_variable(ds: Dataset, observed: Dataset, label: str) -> None:
    """Datasets pair only on the same variable (each dataset has one)."""
    if ds.variable != observed.variable:
        raise PlanError(
            f"{label}: {ds.name!r} is {ds.variable!r}, {observed.name!r} is "
            f"{observed.variable!r}; only the same variable pairs"
        )


def labels_name(field_name: str) -> str:
    """Catalog name of a field's per-series labels (``series_key``, ``value``)."""
    return f"__labels_{field_name}"


def _labels(catalog: Catalog, f: AttributeField | WithinField, observed: Dataset) -> Dataset:
    """Each observed series' label for an attribute or within field.

    Runs on the platform put them in the catalog (they read layers to make them). An
    attribute of the series themselves also comes straight from the series table.
    """
    name = labels_name(f.name)
    if name in catalog:
        return catalog[name]
    if isinstance(f, AttributeField) and f.of == "series" and observed.series is not None:
        if f.attribute not in observed.series.column_names:
            raise PlanError(
                f"field {f.name!r}: {observed.name!r} series have no attribute {f.attribute!r}; "
                f"they have: {', '.join(observed.series.column_names)}"
            )
        table = observed.series.select(["series_key", f.attribute]).rename_columns(
            ["series_key", "value"]
        )
        return Dataset(name=name, role="auxiliary", table=table, kind="labels")
    raise PlanError(
        f"field {f.name!r} needs each observed series' location, which runs on the platform "
        f"look up ({observed.name!r} series and its locations layer)"
    )


def _percentile(catalog: Catalog, f: PercentileField, observed: Dataset, unit: str) -> Dataset:
    """Return a percentiles dataset's values at one percentile, in the evaluation unit."""
    if f.from_ not in catalog:
        raise PlanError(f"field {f.name!r}: {f.from_!r} is not in the catalog")
    ds = catalog[f.from_]
    if ds.role != "auxiliary" or ds.kind != "percentiles":
        raise PlanError(f"field {f.name!r}: {f.from_!r} is not a percentiles dataset")
    assert ds.unit is not None and observed.unit is not None
    _same_variable(ds, observed, f"field {f.name!r}")
    try:
        conversion_between(catalog.units, ds.unit, observed.unit)  # same dimension
        conv = conversion_between(catalog.units, ds.unit, unit)
    except UnitError as exc:
        raise PlanError(f"field {f.name!r}: {exc}") from exc
    at = pc.abs(pc.subtract(ds.table["percentile"], pa.scalar(f.percentile)))
    table = ds.table.filter(pc.less(at, pa.scalar(1e-9)))
    if table.num_rows == 0 and ds.table.num_rows > 0:
        built = sorted(set(ds.table["percentile"].to_pylist()))
        raise PlanError(
            f"field {f.name!r}: {f.from_!r} has no {f.percentile:g}th percentile; "
            f"it has {', '.join(f'{p:g}' for p in built)}"
        )
    value = pc.add(pc.multiply(table["value"], conv.scale), conv.offset)
    table = pa.table({"series_key": table["series_key"], "value": value})
    return replace(ds, table=table)


def _naive_scale(catalog: Catalog, f: NaiveScaleField, observed: Dataset, unit: str) -> Dataset:
    """Return a naive_scale dataset's rows for one lag, scaled to the evaluation unit."""
    if f.from_ not in catalog:
        raise PlanError(f"field {f.name!r}: {f.from_!r} is not in the catalog")
    ds = catalog[f.from_]
    if ds.role != "auxiliary" or ds.kind != "naive_scale":
        raise PlanError(f"field {f.name!r}: {f.from_!r} is not a naive_scale dataset")
    assert ds.unit is not None and observed.unit is not None
    _same_variable(ds, observed, f"field {f.name!r}")
    try:
        conversion_between(catalog.units, ds.unit, observed.unit)  # same dimension
        factor = conversion_between(catalog.units, ds.unit, unit).scale
    except UnitError as exc:
        raise PlanError(f"field {f.name!r}: {exc}") from exc
    seconds = int(parse_duration(f.lag).total_seconds())
    table = ds.table.filter(pc.equal(ds.table["lag_seconds"], pa.scalar(seconds, pa.int64())))
    i = table.schema.get_field_index("scale")
    return replace(ds, table=table.set_column(i, "scale", pc.multiply(table["scale"], factor)))
