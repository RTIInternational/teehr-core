import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from teehr_core.spec import ExpressionError, check_expression, json_schema, load_spec

ROOT = Path(__file__).resolve().parents[1]

BASE: dict[str, Any] = {
    "version": 1,
    "name": "t",
    "inputs": {"primary": {"dataset": "obs"}, "secondary": {"dataset": "fc"}},
    "pair": {"relation": "xw"},
    "aggregate": [{"group_by": ["primary_series_key"], "metrics": ["nash_sutcliffe_efficiency"]}],
}


def spec(**overrides: Any) -> dict[str, Any]:
    return {**BASE, **overrides}


def test_loads_fixture_specs() -> None:
    for path in sorted((Path(__file__).parent / "specs").glob("*.yaml")):
        assert load_spec(path).version == 1


def test_single_predicted_becomes_list() -> None:
    assert len(load_spec(spec()).inputs.secondary) == 1


def test_output_names() -> None:
    s = load_spec(
        spec(
            aggregate=[
                {
                    "group_by": ["a", "b"],
                    "metrics": [
                        "kling_gupta_efficiency",
                        {"name": "nash_sutcliffe_efficiency", "transform": "log"},
                        {"name": "root_mean_square_error", "as": "e"},
                    ],
                },
                {"group_by": ["a"], "metrics": [{"name": "mean", "of": "kling_gupta_efficiency"}]},
            ]
        )
    )
    assert s.aggregate[0].output_columns() == [
        "a",
        "b",
        "kling_gupta_efficiency",
        "nash_sutcliffe_efficiency_log",
        "e",
    ]
    assert s.aggregate[1].output_columns() == ["a", "mean_kling_gupta_efficiency"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"version": 2}, "version"),
        ({"surprise": True}, "Extra inputs"),
        ({"pair": {"relation": "a", "on_attribute": "b"}}, "exactly one"),
        ({"pair": {}}, "exactly one"),
        ({"aggregate": []}, "at least 1"),
        ({"aggregate": [{"metrics": ["nope"]}]}, "unknown metric"),
        (
            {"aggregate": [{"metrics": [{"name": "nash_sutcliffe_efficiency", "of": "x"}]}]},
            "later stages",
        ),
        (
            {
                "aggregate": [
                    {"group_by": ["a"], "metrics": ["nash_sutcliffe_efficiency"]},
                    {"metrics": ["mean"]},
                ]
            },
            "needs 'of'",
        ),
        (
            {
                "aggregate": [
                    {"group_by": ["a"], "metrics": ["nash_sutcliffe_efficiency"]},
                    {
                        "group_by": ["b"],
                        "metrics": [{"name": "mean", "of": "nash_sutcliffe_efficiency"}],
                    },
                ]
            },
            "subset",
        ),
        (
            {
                "aggregate": [
                    {"group_by": ["a"], "metrics": ["nash_sutcliffe_efficiency"]},
                    {"metrics": [{"name": "mean", "of": "kling_gupta_efficiency"}]},
                ]
            },
            "not a column",
        ),
        (
            {
                "aggregate": [
                    {"metrics": ["nash_sutcliffe_efficiency", "nash_sutcliffe_efficiency"]}
                ]
            },
            "duplicate",
        ),
        ({"fields": [{"name": "Bad Name", "expr": "1"}]}, "pattern"),
        ({"fields": [{"name": "b", "fn": "lead_time_bins"}]}, "exactly one"),
        (
            {"fields": [{"name": "b", "fn": "lead_time_bins", "edges": ["PT6H", "PT1H"]}]},
            "increasing",
        ),
        ({"fields": [{"name": "b", "fn": "lead_time_bins", "width": "PT0H"}]}, "positive"),
        ({"filter": "read_csv('x')"}, "not allowed"),
    ],
)
def test_invalid_specs(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        load_spec(spec(**overrides))


CRAFTED = [
    "public.usgs_nwm31_crosswalk\" AT (VERSION => 2) UNION ALL SELECT read_text('/x') --",
    'x"; SELECT 1 --',
    "public.a.b",
    "elsewhere.x",
    "Upper",
    "1starts_with_digit",
    "",
    "a" * 61,
]


def _with_ref(where: str, ref: str) -> dict[str, Any]:
    """BASE with one dataset reference replaced (the others stay valid)."""
    s: dict[str, Any] = json.loads(json.dumps(BASE))
    if where == "primary":
        s["inputs"]["primary"] = {"dataset": ref}
    elif where == "secondary":
        s["inputs"]["secondary"] = [{"dataset": ref}]
    elif where == "relation":
        s["pair"] = {"relation": ref}
    elif where == "like":
        s["inputs"]["secondary"] = [{"generate": "persistence", "like": ref}]
    elif where == "from":
        s["inputs"]["secondary"] = [{"generate": "climatology", "from": ref}]
    elif where == "naive_scale":
        s["fields"] = [{"name": "scale", "fn": "naive_scale", "from": ref, "lag": "PT1H"}]
    elif where == "percentile":
        s["fields"] = [{"name": "p", "fn": "percentile", "from": ref, "percentile": 90}]
    return s


WHERE = ["primary", "secondary", "relation", "like", "from", "naive_scale", "percentile"]


@pytest.mark.parametrize("where", WHERE)
@pytest.mark.parametrize("ref", CRAFTED)
def test_dataset_references_are_only_names(where: str, ref: str) -> None:
    """Dataset references reach SQL as table names, so the model admits only names."""
    with pytest.raises(ValidationError, match="isn't a dataset reference"):
        load_spec(_with_ref(where, ref))


@pytest.mark.parametrize("where", WHERE)
@pytest.mark.parametrize("ref", ["usgs_streamflow", "public.nwm31_short_range", "a", "x_1"])
def test_dataset_references_accept_names(where: str, ref: str) -> None:
    load_spec(_with_ref(where, ref))


@pytest.mark.parametrize(
    "expr",
    [
        "primary_value > 100",
        "month(value_time) IN (12, 1, 2)",
        "value_time - reference_time < INTERVAL '24' HOUR",
        "log10(primary_value) + ln(secondary_value) + log(2, primary_value) > 0",
        "CAST(member AS INT) = 1 AND extract(month FROM value_time) = 6",
        "CASE WHEN primary_value > 1 THEN 'hi' ELSE 'lo' END",
        "coalesce(primary_quality, '') LIKE 'A%'",
        "abs(secondary_value - primary_value) / greatest(primary_value, 1)",
        "value_time >= TIMESTAMP '2026-10-01 00:00:00+00'",
        "primary_value % 2 = 0 AND lower(primary_series_key) LIKE 'usgs-%'",
    ],
)
def test_allowed_expressions(expr: str) -> None:
    assert check_expression(expr)


@pytest.mark.parametrize(
    "expr",
    [
        "(SELECT 1)",
        "read_csv('/etc/passwd')",
        "read_parquet('s3://bucket/x')",
        "getenv('HOME')",
        "glob('*')",
        "primary_value IN (SELECT 1)",
        "*",
        "1; DROP TABLE t",
        "current_setting('threads')",
        "primary_value > ",
    ],
)
def test_rejected_expressions(expr: str) -> None:
    with pytest.raises(ExpressionError):
        check_expression(expr)


@pytest.mark.parametrize(
    ("expr", "reason"),
    [
        ("log(primary_value) > 1", "ln, log10"),
        ("round(primary_value) = 1", "ties"),
        ("dayofweek(value_time) = 1", "weekdays"),
        ("week(value_time) = 1", "weeks"),
        ("date_diff('hour', reference_time, value_time) < 24", "subtract"),
        ("isnan(primary_value)", "NaN"),
        ("primary_series_key ILIKE 'usgs%'", "lower"),
        ("isfinite(primary_value)", "not allowed"),
        ("epoch(value_time) > 0", "not allowed"),
        ("value_time - reference_time < INTERVAL '6 hours'", "HOUR"),
        ("extract(dow FROM value_time) = 1", "MONTH"),
        ("CAST(primary_value AS BLOB) IS NULL", "DOUBLE"),
    ],
)
def test_engine_dependent_expressions_are_refused(expr: str, reason: str) -> None:
    """Functions that give different answers in DuckDB, Spark and Trino are refused."""
    with pytest.raises(ExpressionError, match=reason):
        check_expression(expr)


def test_expressions_are_stored_as_written() -> None:
    from teehr_core.spec.expressions import to_duckdb

    text = "log10(primary_value) > 2 AND value_time >= TIMESTAMP '2026-01-01'"
    assert load_spec(spec(filter=text)).filter == text
    assert to_duckdb(text) == (
        "LOG(10, primary_value) > 2 AND value_time >= CAST('2026-01-01' AS TIMESTAMP)"
    )


def test_json_schema_is_serializable() -> None:
    schema = json_schema()
    assert schema["properties"]["version"]["const"] == 1
    assert "aggregate" in schema["required"]
    json.dumps(schema)


def test_load_from_yaml_text() -> None:
    text = "version: 1\nname: t\ninputs:\n  primary: {dataset: o}\n  secondary: {dataset: p}\n"
    text += "pair: {on_attribute: zone}\naggregate:\n  - metrics: [count]\n"
    assert load_spec(text).pair.on_attribute == "zone"


def test_published_schema_is_current() -> None:
    """The committed JSON Schema must match the models. Regenerate with:

    uv run python -c "import json; from teehr_core.spec import json_schema; \
open(ROOT / 'docs/spec/evaluation-spec-v1.schema.json','w').write(json.dumps(json_schema(), \
indent=2, sort_keys=True) + '\\n')"
    """
    published = json.loads(Path(ROOT, "docs/spec/evaluation-spec-v1.schema.json").read_text())
    assert published == json_schema()


def test_documented_functions_match_allow_list() -> None:
    from teehr_core.spec.expressions import ALLOWED_FUNCTIONS

    guide = Path(ROOT, "docs/spec/README.md").read_text()
    section = guide.split("these functions:")[1].split("Subqueries")[0]
    documented = sorted(re.findall(r"`([a-z_0-9]+)`", section))
    assert documented == ALLOWED_FUNCTIONS
    two_args = {
        "pow": "pow(a, 2)",
        "date_trunc": "date_trunc('day', a)",
        "extract": "extract(hour FROM a)",
        "greatest": "greatest(a, b)",
        "least": "least(a, b)",
        "nullif": "nullif(a, 0)",
        "power": "power(a, 2)",
        "concat": "concat(a, b)",
        "substring": "substring(a, 1, 2)",
        "coalesce": "coalesce(a, 0)",
    }
    for name in ALLOWED_FUNCTIONS:
        check_expression(two_args.get(name, f"{name}(a)"))


def test_documented_metrics_match_registry() -> None:
    from teehr_core.metrics import METRICS

    text = Path(ROOT, "docs/spec/README.md").read_text()
    table = text.split("| Name | Family | Aliases | Definition |")[1].split("\n\n")[0]
    documented = re.findall(r"^\| `([a-z0-9_]+)` \| [a-z]+ \|", table, re.M)
    assert documented == list(METRICS)


@pytest.mark.parametrize(
    ("fields", "output", "message"),
    [
        ([], {"geometry": "secondary_dataset"}, "must be primary_series_key or a within field"),
        (
            [{"name": "b", "fn": "within", "layer": "x"}],
            {"geometry": "b"},
            "final stage's group_by",
        ),
        (
            [{"name": "b", "fn": "within", "layer": "x", "attribute": "geometry"}],
            {},
            "output.geometry",
        ),
        ([{"name": "b", "fn": "attribute", "attribute": "geometry"}], {}, "output.geometry"),
        (
            [
                {"name": "b", "fn": "within", "layer": "x"},
                {"name": "c", "fn": "within", "layer": "x", "attribute": "name"},
            ],
            {},
            "one within field per layer",
        ),
        ([], {"attributes": ["name"]}, "needs output.geometry"),
    ],
)
def test_output_geometry_checks(
    fields: list[dict[str, Any]], output: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        load_spec(spec(fields=fields, output=output))


def test_output_geometry_accepts_within_and_series_keys() -> None:
    within = [{"name": "b", "fn": "within", "layer": "x"}]
    stage = [{"group_by": ["b", "primary_series_key"], "metrics": ["nash_sutcliffe_efficiency"]}]
    for column in ("b", "primary_series_key"):
        loaded = load_spec(spec(fields=within, aggregate=stage, output={"geometry": column}))
        assert loaded.output.geometry == column


def test_calendar_field_checks() -> None:
    ok = load_spec(spec(fields=[{"name": "wy", "fn": "calendar", "part": "water_year"}]))
    f = ok.fields[0]
    assert f.of == "value_time" and f.start_month() == 10  # type: ignore[union-attr]
    with pytest.raises(ValidationError, match="doesn't take year_start_month"):
        load_spec(
            spec(fields=[{"name": "m", "fn": "calendar", "part": "month", "year_start_month": 4}])
        )
    with pytest.raises(ValidationError, match="part"):
        load_spec(spec(fields=[{"name": "m", "fn": "calendar", "part": "fortnight"}]))


def test_conditions_field_checks() -> None:
    ok = load_spec(
        spec(fields=[{"name": "f", "fn": "conditions", "levels": {"high": "primary_value > 1"}}])
    )
    assert ok.fields[0].labels() == ["all", "high"]  # type: ignore[union-attr]
    with pytest.raises(ValidationError, match="'all' is the level every pair is in"):
        load_spec(spec(fields=[{"name": "f", "fn": "conditions", "levels": {"all": "true"}}]))
    with pytest.raises(ValidationError, match="levels"):
        load_spec(spec(fields=[{"name": "f", "fn": "conditions", "levels": {}}]))
    with pytest.raises(ValidationError):
        load_spec(spec(fields=[{"name": "f", "fn": "conditions", "levels": {"x": "(SELECT 1)"}}]))


def _rollup_spec(stage: dict[str, Any], fields: list[dict[str, Any]] | None = None) -> Any:
    base = spec(fields=fields or [])
    base["aggregate"] = [{"metrics": ["count"], **stage}]
    return base


def test_rollup_folds_into_grouping_sets_and_group_by() -> None:
    fields = [
        {"name": "wy", "fn": "calendar", "part": "water_year"},
        {"name": "q", "fn": "calendar", "part": "quarter"},
    ]
    s = load_spec(_rollup_spec({"group_by": ["primary_series_key"], "rollup": ["wy", "q"]}, fields))
    stage = s.aggregate[0]
    assert stage.group_by == ["primary_series_key", "wy", "q", "grouping"]
    assert stage.grouping_sets == [["wy", "q"], ["wy"], []]
    assert stage.rollup is None and stage.set_names() == ["q", "wy", "all"]
    assert stage.rolled_up() == ["wy", "q"] and stage.per_series()
    # A dumped spec reads back the same (the fold is idempotent).
    assert load_spec(s.model_dump(by_alias=True, mode="json")) == s
    # A later stage may group by the grouping column.
    two = _rollup_spec({"group_by": ["primary_series_key"], "rollup": ["wy", "q"]}, fields)
    two["aggregate"].append({"group_by": ["grouping"], "metrics": [{"name": "sum", "of": "count"}]})
    assert load_spec(two).aggregate[1].group_by == ["grouping"]


def test_rollup_over_series_is_not_per_series() -> None:
    s = load_spec(
        _rollup_spec({"group_by": ["secondary_dataset"], "rollup": ["primary_series_key"]})
    )
    assert not s.aggregate[0].per_series()


@pytest.mark.parametrize(
    ("stage", "fields", "message"),
    [
        ({"rollup": ["a"], "grouping_sets": [["a"]]}, None, "not both"),
        ({"rollup": []}, None, "at least one column"),
        ({"group_by": ["a"], "rollup": ["a"]}, None, "in both group_by and the rollup"),
        ({"grouping_sets": [["a"], ["a"]]}, None, "repeat a set"),
        ({"group_by": ["grouping"]}, None, "comes from rollup"),
        ({"rollup": ["a"]}, [{"name": "grouping", "expr": "1"}], "no field may be named"),
        (
            {"rollup": ["q", "wy"]},
            [
                {"name": "wy", "fn": "calendar", "part": "water_year"},
                {"name": "q", "fn": "calendar", "part": "quarter"},
            ],
            "keeps q but not wy",
        ),
    ],
)
def test_rollup_checks(
    stage: dict[str, Any], fields: list[dict[str, Any]] | None, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        load_spec(_rollup_spec(stage, fields))


def test_rollup_is_first_stage_only() -> None:
    two = _rollup_spec({"group_by": ["primary_series_key"]})
    two["aggregate"].append(
        {
            "group_by": [],
            "rollup": ["primary_series_key"],
            "metrics": [{"name": "sum", "of": "count"}],
        }
    )
    with pytest.raises(ValidationError, match="first stage only"):
        load_spec(two)


@pytest.mark.parametrize(
    ("reduce", "stage", "message"),
    [
        (
            {"by": ["secondary_dataset"], "stats": ["max"]},
            {"group_by": []},
            "must include primary_series_key",
        ),
        (
            {"by": ["primary_series_key"], "stats": ["max", "max"]},
            {"group_by": []},
            "repeat a statistic",
        ),
        (
            {"by": ["primary_series_key"], "stats": ["max"]},
            {"group_by": ["reference_time"]},
            "which a reduce doesn't keep",
        ),
        (
            {"by": ["primary_series_key"], "stats": ["max"]},
            {"group_by": [], "metrics": [{"name": "max_value_time", "side": "primary"}]},
            "read each pair's time",
        ),
    ],
)
def test_reduce_checks(reduce: dict[str, Any], stage: dict[str, Any], message: str) -> None:
    s = spec()
    s["reduce"] = reduce
    s["aggregate"] = [{"metrics": ["count"], **stage}]
    with pytest.raises(ValidationError, match=message):
        load_spec(s)


def test_reduce_keeps_window_keys_and_stat() -> None:
    s = spec(fields=[{"name": "q90", "fn": "percentile", "from": "p", "percentile": 90}])
    s["reduce"] = {"by": ["primary_series_key", "reference_time"], "stats": ["mean", "max"]}
    s["aggregate"] = [
        {"group_by": ["primary_series_key", "secondary_dataset", "stat"], "metrics": ["count"]}
    ]
    assert load_spec(s).reduce_keys() == ["primary_series_key", "reference_time"]


def test_within_fields_on_different_layers() -> None:
    """One within field per layer, as many layers as needed (basin, state, region)."""
    fields = [
        {"name": "basin", "fn": "within", "layer": "basins"},
        {"name": "state", "fn": "within", "layer": "public.states"},
        {"name": "region", "fn": "within", "layer": "regions", "attribute": "name"},
    ]
    stage = [{"group_by": ["basin", "state", "region"], "metrics": ["nash_sutcliffe_efficiency"]}]
    assert len(load_spec(spec(fields=fields, aggregate=stage)).fields) == 3


def test_bootstrap_blocks_of_days_and_months() -> None:
    from pydantic import ValidationError

    from teehr_core.spec.models import Bootstrap

    def periods(**kw: Any) -> object:
        return Bootstrap.model_validate(kw).block_periods()

    assert periods(block="P7D") == ("day", 7)
    assert periods(block="PT48H") == ("day", 2)
    assert periods(block="P3M", method="stationary") == ("month", 3)
    assert periods(block="PT12H") is None  # rows, from each group's span
    assert periods(block=24) is None  # rows
    assert periods(block="P7D", method="iid") is None
    assert periods(method="year_block") is None
    with pytest.raises(ValidationError, match="must be positive"):
        Bootstrap.model_validate({"block": "P0M"})
    with pytest.raises(ValidationError):
        Bootstrap.model_validate({"block": "1month"})
