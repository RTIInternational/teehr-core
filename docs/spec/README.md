# Evaluation spec, version 1

> The spec is shared by the evaluation systems that use teehr-core, TEEHR and VeriRace, and the same spec gives the same results on either. Where a system doesn't support something yet, this guide says so; commands are shown for both.

An evaluation spec describes one evaluation as YAML or JSON. The machine-readable definition is [`evaluation-spec-v1.schema.json`](evaluation-spec-v1.schema.json); every metric is in the [metrics reference](../metrics.md).

## Example

```yaml
version: 1
name: nwm-vs-usgs-by-lead-time
inputs:
  primary: {dataset: public.usgs_streamflow}
  secondary:
    - {dataset: public.nwm31_short_range}
    - {dataset: public.nwm31_medium_range_mem1, filter: "hour(reference_time) = 0"}
pair:
  relation: public.usgs_nwm31_crosswalk   # or: on_attribute: zone
fields:
  - {name: lead_time_bin, fn: lead_time_bins, width: PT6H}
  - {name: high_flow, expr: "primary_value > 1000"}
filter: "primary_approval_status = 'Provisional'"
aggregate:
  - group_by: [primary_series_key, secondary_dataset, lead_time_bin]
    metrics: [count, rmse, nse, kge, {name: nse, transform: log}]
  - group_by: [secondary_dataset, lead_time_bin]
    metrics: [{name: median, of: kge}, {name: sum, of: count}]
output:
  include_pairs: false
```

Save and run it:

With TEEHR:

```bash
teehr evaluations plan spec.yaml -p my_project
teehr evaluations save nwm_vs_usgs_by_lead_time spec.yaml -p my_project
teehr runs start nwm_vs_usgs_by_lead_time -p my_project --wait
teehr runs results <run-id> -p my_project --out results.parquet
```

With VeriRace:

```bash
verirace spec validate spec.yaml
verirace evaluations save spec.yaml -p my-project
verirace runs start nwm-vs-usgs-by-lead-time -p my-project --wait
verirace runs results <run-id> -p my-project --out results.parquet
```

Dataset names without a prefix refer to the project's own datasets; `public.` refers to the shared public ones. The example compares public data, so every reference has the prefix. A spec can mix both, e.g. your own forecast against `public.usgs_streamflow`. A reference is only a name: lowercase letters, digits and `_`, starting with a letter, up to 60 characters, with an optional `public.` prefix. That covers `dataset`, `relation`, `like` and `from`.

## Steps

Steps always run in this order: `inputs → pair → fields → filter → aggregate → output`.

### `inputs`
The two sides of a pairing are **primary** and **secondary**, and paired columns carry their names (`primary_value`, `secondary_series_key`, …). The names are shared with the TEEHR team's spec. A dataset's role (`observed`, `forecast`, `simulated`) still says what its data is.

- `primary`: one dataset with role `observed`, the values scored against.
- `secondary`: one or more datasets with role `forecast` or `simulated`. Each dataset's name becomes the value of the `secondary_dataset` column.
- Either can take a `filter` expression, applied before pairing.

#### Generated secondary inputs (persistence, climatology, blend)

A secondary input can be generated from the observations instead of read from a dataset:

```yaml
secondary:
  - {dataset: public.nwm31_short_range}
  - {generate: persistence, like: public.nwm31_short_range, name: persistence, max_age: PT2H}
```

`persistence` copies the grid of the `like` forecast (every series, reference time and value time) and uses, for each forecast, the last observation complete by its reference time from the observed series linked to that forecast series. Instantaneous values count at their time. Values over a period (e.g. hourly mean load) count once their window has ended: an hour starting at 05:00 (`hour_starting`) is usable for a forecast issued at 06:00. Its pairs are keyed by the observed series (`secondary_series_key` is the gage). `max_age` drops forecasts whose latest observation is older than that at the reference time. **Set it** (e.g. `PT2H` for hourly forecasts): without it a forecast may use an observation from any time before, so a run can't limit what it reads to the forecasts' span and reads every observed series' whole record. That can be many times the data and memory the evaluation needs, and a large run can run out of memory. `validate_spec` and `estimate` warn about it. `name` sets its `secondary_dataset` value (default `persistence:<like>`), which is also what a skill score's `reference` names.

It's computed within each run from the observations the run reads, so it suits references that need only recent data.

`climatology` looks up a stored climatology (an auxiliary dataset of kind `climatology`: one mean value per series, day of year and hour over the period of record, from hourly means so each hour weighs the same) at each observation's day of year and hour:

```yaml
secondary:
  - {dataset: public.nwm31_short_range}
  - {generate: climatology, from: public.usgs_streamflow_climatology, name: climatology}
```

`blend` is the optimal combination of persistence and climatology (Yang et al. 2020), the recommended reference in solar forecasting and a fair one generally. Persistence is hard to beat at short leads and climatology at long ones; the blend beats both. On the `like` forecast's grid, it is the climatology at the value time plus the persisted anomaly (the observation minus climatology at the persisted time), damped by the anomalies' correlation at that lag. The correlation is fitted per observed series and lag from the observations the run reads, and clipped to [0, 1].

```yaml
secondary:
  - {dataset: public.nyiso_load_forecast}
  - {generate: blend, like: public.nyiso_load_forecast, from: public.nyiso_load_climatology, name: blend}
```

Climatologies are derived datasets, built over the whole record and refreshed monthly by the platform, with each day smoothed over a window of days around it (VeriRace: the `build-climatology` flow, or `python -m verirace.warehouse climatology <lake> <dataset> [--window-days 15]`). **TEEHR can't build them yet** ([teehr-cloud-core#251](https://github.com/RTIInternational/teehr-cloud-core/issues/251)), so it refuses `climatology` and `blend` inputs; `persistence` works. A climatology has no reference times, so it can't be grouped by `lead_time_bins`. Its pairs are keyed by the observed series, and its `name` works as a skill `reference`.

### `pair`
Give exactly one of:
- `relation`: an auxiliary `crosswalk` dataset; `left_key` is the primary series, `right_key` the secondary series. Public crosswalks are named `public.<name>`; a project can upload its own (`teehr uploads add links.csv --kind crosswalk --left ... --right ...`; `verirace uploads add links.csv --kind crosswalk`).
- `on_attribute`: a series attribute (from the series tables) both sides share (e.g. `zone`).

`common_sample: true` scores every predicted input on the same sample. After fields and the filter, it keeps only observations (observed series and value time) that every predicted input paired with. Without it, a forecast missing on hard-to-predict days looks better than one that forecast them. This is the IEA Wind recommended practice for comparing forecasts.

The datasets must be of the same `variable` (each dataset has one) and the same `statistic`. Secondary values are converted to the primary dataset's unit. Pairs where either value is missing or not finite are dropped.

Paired rows have these columns: `primary_series_key`, `secondary_series_key`, `reference_time` (null for simulations), `value_time`, `primary_value`, `secondary_value`, `variable`, `unit`, `secondary_dataset`, plus `member` / `quantile` when present. Extra columns from the inputs are kept with a `primary_` or `secondary_` prefix.

Values pair when the spans of time they cover are the same: each dataset (or series) names its **period** ([periods standard](https://github.com/RTIInternational/teehr-shared-eval-spec/blob/main/standards/periods.md)), such as `instant` for readings, `hour_starting` or `hour_ending` for hourly means, or a season like April 1 to July 31. So an hourly mean labelled by its start pairs with one labelled by its end, and a seasonal forecast pairs with the observed volume of the same season whatever time inside it each is labelled. When values cover more than an instant, pairs also have their window, `period_start` and `period_end`: the same on both sides, since values pair only then (`value_time` is the observation's).

### `fields`
- `expr`: a SQL expression over the paired columns (see [Expressions](#expressions)).
- `fn: calendar`: a calendar label of each pair's time, to group by. `part` is `year` (`2025`), `month` (`2025-03`), `water_year`, `quarter` or `season`. `of` is the time labelled: `value_time` (default) or `reference_time` (when a forecast was issued; null for pairs without one). Times are UTC.
  - `water_year`: the year that starts in `year_start_month` (default 10, October), named for the calendar year it ends in, so October 2024 is water year 2025.
  - `quarter`: calendar quarters (`2025-Q1` is January to March). With `year_start_month`, quarters of that year instead, named by its water year (`2025-Q1` is October to December 2024 for `year_start_month: 10`).
  - `season`: meteorological `DJF`, `MAM`, `JJA` or `SON`, across years; add a `year` or `water_year` field to the grouping for one per year.

  ```yaml
  fields:
    - {name: water_year, fn: calendar, part: water_year, of: reference_time}
    - {name: quarter, fn: calendar, part: quarter, year_start_month: 10}
  ```

- `fn: conditions`: overlapping subsets. Each pair is copied into every level whose condition it meets, so grouping by the field scores each level over its own pairs: a pair above q95 counts in `above_q85` too. `levels` maps a label to a SQL condition (checked like `filter`), and `all` (default true) adds a level holding every pair, labelled `all` (a reserved name).
  - **When it's applied:** the copies are made after every other field and before `filter`. A condition can use any field, and `filter` can keep a level, e.g. `flow = 'above_q95'`. Series fields (exceedance, events, change) see each pair once.
  - **Several conditions fields** multiply: each pair lands in every combination of levels.
  - **Cost:** a run scores every copy. VeriRace's estimate measures the copies per pair on a sample (`expansion`), scales pairs and bootstrap work by it, and warns. Stored and included pairs carry the copies too.

  ```yaml
  fields:
    - {name: q85, fn: percentile, from: public.usgs_streamflow_percentiles, percentile: 85}
    - {name: q95, fn: percentile, from: public.usgs_streamflow_percentiles, percentile: 95}
    - name: flow
      fn: conditions
      levels:
        above_q85: "primary_value > q85"
        above_q95: "primary_value > q95"
  aggregate:
    - group_by: [primary_series_key, secondary_dataset, flow]
      metrics: [count, kge]
  ```

- `fn: lead_time_bins`: lead time (`value_time - reference_time`) as labels like `[0h, 6h)`. Give `width` (e.g. `PT6H`) or `edges` (e.g. `[PT0S, PT24H, PT72H]`; leads past the last edge fall in `[72h, inf)`).
- `fn: exceedance`: each value's exceedance probability within its series, `rank / (n + 1)` with the largest value first (ties share the lower rank). `side: primary` (default) ranks each primary series; `side: secondary` ranks each forecast (dataset, series, reference time, member or quantile). Use it to filter or group by flow regime, e.g. `filter: "p_obs <= 0.1"` for high flows.
- `fn: events`: runs of values at or above `threshold` (a value with its unit, `{value: v, unit: u}`; `{quantile: q}` of each series' own values in the evaluated window, or `{column: c}`, a pair column such as a `percentile` field, for a threshold from each series' whole record). Every pair inside an event gets the event's id (`<series>@<start>`), others null; `merge_gap` joins runs separated by at most that long (the gap included), and `min_duration` drops shorter events. `side` as for `exceedance`. Group by the field to score each event, or filter `ev IS NOT NULL` to score only events:

```yaml
fields:
  - {name: ev, fn: events, threshold: {quantile: 0.9}, merge_gap: PT6H, min_duration: PT3H}
filter: "ev IS NOT NULL"
aggregate:
  - group_by: [ev, secondary_dataset]
    metrics: [count, rmse, max_value_delta, max_value_time_delta]
```

- `fn: change`: each value's change over a `window` within its series: `value(t) - value(t - window)`, null when the series has no value exactly a window earlier among the pairs. `side` as for `exceedance` (`secondary` follows each forecast trajectory). With a metric's `values` option, contingency metrics score **ramps**, i.e. changes of at least a threshold:

```yaml
fields:
  - {name: rise_obs, fn: change, side: primary, window: PT3H}
  - {name: rise_pred, fn: change, side: secondary, window: PT3H}
aggregate:
  - group_by: [primary_series_key]
    metrics:
      - {name: csi, threshold: {value: 300, unit: MW}, values: {primary: rise_obs, secondary: rise_pred}, as: ramp_csi}
      - {name: rmse, values: {primary: rise_obs, secondary: rise_pred}, as: ramp_rmse}
```

  For falling ramps, add expression fields with the negated changes (`-rise_obs`); for either direction, `abs(...)`.
- `fn: naive_scale`: each pair's in-sample naive scale for MASE, looked up from a `naive_scale` dataset (`from`) for a `lag`: the observed series' mean `|y(t) - y(t - lag)|` over a training period, converted to the observed unit. Use it as the `mase` metric's `scale`:

```yaml
fields:
  - {name: scale_1d, fn: naive_scale, from: public.usgs_streamflow_naive_scale, lag: P1D}
aggregate:
  - group_by: [primary_series_key, secondary_dataset]
    metrics: [mean_absolute_error, {name: mase, scale: {column: scale_1d}}]
```

Naive scales are derived datasets, built over the whole record and refreshed monthly by the platform (VeriRace: the `build-naive-scale` flow at 1 h and 1 d, or `python -m verirace.warehouse naive-scale <lake> <dataset> [--lag P1D ...]`). **TEEHR can't build them yet** (teehr-cloud-core#251), so it refuses `naive_scale` fields, and with them `mean_absolute_scaled_error`, whose `scale` comes from one. Each pair is scaled by its own series' scale, so a group spanning many series averages comparable values.

- `fn: percentile`: each pair's stored `percentile` (0 to 100) of its observed series over the series' whole period of record, from a `percentiles` dataset (`from`), converted to the observed unit. Use it as a threshold for contingency metrics or events. It's steadier than `{primary_quantile: q}` or an events field's `{quantile: q}`, which use only the window being evaluated: a short or unusual window gives an unrepresentative threshold.

```yaml
fields:
  - {name: q99, fn: percentile, from: public.usgs_streamflow_percentiles, percentile: 99}
  - {name: high_flow, fn: events, threshold: {column: q99}, min_duration: PT3H}
aggregate:
  - group_by: [primary_series_key, secondary_dataset]
    metrics:
      - {name: pod, threshold: {column: q99}}
      - {name: csi, threshold: {column: q99}}
```

Percentiles are non-exceedance: the 90th is the value not exceeded 90% of the time, a high flow (hydrology's Q90 is the opposite, the flow exceeded 90% of the time). They're of hourly means (not daily means; each hour weighs the same whether a gage reported hourly or every 15 minutes), at 0.1, 0.5, every whole percentile from 1 to 99, 99.5 and 99.9, refreshed monthly by the platform (VeriRace: the `build-percentiles` flow, or `python -m verirace.warehouse percentiles <lake> <dataset> [--min-years 10]`). **TEEHR can't build them yet** (teehr-cloud-core#251), so it refuses `percentile` fields, and thresholds taken from them. Gages with less than 10 years of data get none: their field is null, and contingency metrics and events leave those pairs out.

- `fn: attribute`: an attribute of each pair's observed series, to group or filter by: `attribute` names a column of the observed dataset's series table, or with `of: location`, of its locations layer (the dataset's `location_layer`, matched on `location_key`). Null where a series or location has none.
- `fn: within`: the polygon of a polygon `layer` (yours, or `public.<name>`) that each pair's observed series is in. Its location comes from the observed dataset's locations layer (a line or polygon by its centroid), boundaries count as inside, and the value is the polygon's `attribute` (default its key). Where polygons overlap, the one with the smallest key; null, its own group, where no polygon contains the location. Upload the polygons as a layer (GeoJSON) first, or use a public one: `public.nws_rfc_boundaries` places gages in their NWS River Forecast Center (keys like `ABRFC`; `attribute: name` for `Arkansas-Red Basin`). One `within` field per layer: label by the key, and bring the polygon's other attributes (its name, area) into the results with `output.attributes`.

```yaml
fields:
  - {name: state, fn: attribute, attribute: state_code, of: location}
  - {name: basin, fn: within, layer: my_basins}                        # the polygon's key
  # or label by one of its attributes instead: {name: huc8, fn: within, layer: my_basins, attribute: HUC8}
aggregate:
  - group_by: [basin, secondary_dataset]
    metrics: [count, rmse, kge]
```

Both are computed once per observed series before pairing, so they cost little at any size. A change to a layer they read makes the next incremental run a full one. They label pairs with values, not shapes (`attribute: geometry` is refused): to get each result's polygon or location, group by the key and set `output.geometry`.

Series-aware fields use the pairs being evaluated: with a filter that limits time, quantiles and ranks come from that window, not the full record.

### `filter`
A SQL expression; only pairs where it's true are aggregated.

### `reduce`
Optional: reduce each window of pairs to statistics before the metrics, so they score summaries of each window (its mean, or its peak) rather than every hour.

```yaml
fields:
  - {name: lead_time_bin, fn: lead_time_bins, width: PT6H}
reduce:
  by: [primary_series_key, reference_time, lead_time_bin]   # one window per forecast and bin
  stats: [mean, max]                                    # mean, min, max or sum
aggregate:
  - group_by: [primary_series_key, secondary_dataset, lead_time_bin, stat]
    metrics: [count, nse, kge, relative_bias]
```

- **Windows:** pairs are grouped by `by`. The predicted dataset and series, an ensemble's `member` or `quantile`, and any conditions fields are added, so different forecasts and levels never merge. `by` must include `primary_series_key`, which keeps every window inside one observed series.
- **Reduced pairs:** each window becomes one pair per statistic, with observed and predicted values reduced alike and the statistic named in `stat`. It keeps `by`, the per-series fields (percentile, attribute, within, naive scale) and its first `value_time`. The first stage may group only by those, and metrics that read each pair's time (`max_value_time`, annual peaks) are refused.
- **When:** after `filter` and conditions, before a rollup's copies.
- **Cost:** windows can cross the days state is kept by, so a reduced spec doesn't take the state path. Reruns recompute changed series. VeriRace's estimate scales its pairs by the windows measured on its sample.

### `aggregate`
A list of stages.
- **First stage:** `group_by` columns (none means one overall group) and `metrics` from the registry. A metric can take a `transform` (`log`, natural; `sqrt`, `square`, `cube`, `exp`, `reciprocal` or its alias `inv`, `abs`), not on contingency, Brier, ensemble or rank metrics; pairs where the transform is undefined or not finite are left out of that metric only. To keep zeros under `log` or `reciprocal`, give the transform an `offset`, a value with its unit added to both sides first (`{name: log, offset: {value: 0.01, unit: "m3/s"}}` computes `log(Q + 0.01 m3/s)`); it converts to the evaluation unit by scale alone. The KGE family also takes weights `sr`, `sa`, `sb` (default 1). Output columns are named after the metric's canonical name (`nse` gives `nash_sutcliffe_efficiency`), the metric and transform (`nash_sutcliffe_efficiency_log`), or `as`.
- **`bootstrap`** (first stage only): confidence intervals for the stage's metrics; see [Bootstrap](#bootstrap).
- **`guards`** (first stage only): null metrics where a group's data is too thin or flat to trust them, and say why in a `null_reason` column. Each guard has conditions on the group's observations (`min_count` pairs, `min_obs_mean`, `min_obs_std`) and optional `metrics` (default: all of the stage's). Guarded metrics' bootstrap intervals are nulled too; later stages skip the nulls.

```yaml
aggregate:
  - group_by: [primary_series_key, secondary_dataset]
    metrics: [count, rmse, nse, kge, relative_bias]
    guards:
      - {min_count: 30}                                 # every metric
      - {metrics: [nse, kge], min_obs_std: 0.01}        # flat observations
      - {metrics: [relative_bias], min_obs_mean: 0.1}   # near-zero means
```

A tripped guard writes e.g. `nse, kge: obs std 0.004 < 0.01` (several reasons are joined with `; `).
- **`rollup` / `grouping_sets`** (first stage only): rows at several granularities in one result. `rollup: [water_year, quarter]` (coarsest first, as SQL ROLLUP) adds to the stage's `group_by` the sets (water_year, quarter), (water_year) and (): rows by quarter, by water year, and over everything. `grouping_sets: [[a, b], [c], []]` is the general form.
  - **`grouping`:** each row names its set in this column: the set's finest column (`quarter`, `water_year`) or `all` for the empty set (with columns joined by `+` when two sets end in the same one). Filter on it to get one granularity, e.g. `grouping = 'water_year'`.
  - **`(all)`:** a column a row is aggregated across reads `(all)`, never null, so it can't be confused with a missing value. Rolled-up columns are text.
  - **How a run builds them:**
    - **From state (no extra work):** a run that merges its first stage from mergeable state pairs each pair once, into the finest set. It then builds every coarser set from those state rows, which merge exactly. That's any saved run on the `state` path (the estimate's `path`): metrics that merge from running sums, a bootstrap only of whole time units (see [Bootstrap](#bootstrap)), no guards, a rollup holding all its columns in one set, and no DM test.
    - **By copying pairs:** otherwise (another bootstrap, guards, other metrics, a VeriRace on-demand evaluation), each pair is copied once per set after `filter`, so every metric, interval and guard works as usual, and VeriRace's estimate counts the copies.
    - **Across series:** a rollup across `primary_series_key` spans series, so such a run takes the state or shuffle path, not per-series.
  - **Checks:** a column goes in the rollup or `group_by`, not both. No field may be named `grouping`. A set that keeps a `quarter` or `month` calendar field but drops the year beside it in the rollup is refused, because those labels already name their year.

```yaml
fields:
  - {name: water_year, fn: calendar, part: water_year}
  - {name: quarter, fn: calendar, part: quarter, year_start_month: 10}
aggregate:
  - group_by: [primary_series_key, secondary_dataset]
    rollup: [water_year, quarter]
    metrics: [count, kge]
```

- **Later stages:** group the previous stage's output by a subset of its `group_by` and apply summaries (`mean`, `median`, `min`, `max`, `std`, `sum`, `count`) to one of its columns with `of`. Output columns are named `<summary>_<column>` unless `as` is given. Nulls are ignored.

### `output`
- `include_pairs`: also return the paired rows.
- `unit`: the unit to evaluate in, e.g. `m3/s` or `kW` (default: the observed dataset's). Observed and predicted values are converted to it when paired, so fields, filters on pairs and every metric are in it; thresholds and normalizers, which carry their own unit, are converted to it. Numbers in an input `filter` are in the variable's canonical unit (`teehr variables list`, `verirace variables list`), whatever unit its dataset is stored in, so the same filter keeps the same rows of data stored in cfs or m3/s. VeriRace can also convert a run's results afterwards (`?unit=`); that gives the same numbers except for ratio metrics between units with different zero points (°F and °C), which only evaluating in the unit gets right. Each metric's result unit is in the [metrics reference](../metrics.md#result-units).
- `geometry`: give each result row a geometry, by a column of the final stage's `group_by`: a `within` field (its polygon; polygons sharing the label are unioned) or `primary_series_key` (the observed series' location, from the dataset's locations layer). It's looked up once per result row after aggregating, so it costs nothing per pair. Results then carry a `geometry` column: a GeoJSON object in JSON rows, GeoArrow WKB in Parquet and Arrow, and with `Accept: application/geo+json` the results are a FeatureCollection (the other columns as properties), ready for a GIS. The run page maps a chosen metric.

  ```yaml
  fields:
    - {name: basin, fn: within, layer: my_basins}
  aggregate:
    - group_by: [basin, secondary_dataset]
      metrics: [count, rmse, kge]
  output:
    geometry: basin
    attributes: [name]     # the basin's name too
  ```
- `attributes`: with `geometry`, attributes of the same place to add as result columns, under their own names, from the layer the geometry comes from (the polygons, or the observed locations layer). Where polygons sharing a label disagree on an attribute, it's null. A name that's already a result column is refused.
- `pairs`: whether a saved evaluation's runs keep their pairs in the project, `all` or `none`. Unset uses the system's default: `none` (VeriRace sets it with `VERIRACE_STORE_PAIRS`). First-stage rows and results are always kept. Without pairs, runs write far less and incremental runs still work when the first stage groups by `primary_series_key`; a first stage that spans series then runs in full each time.

## Metric options

Some metrics take options, written next to the name:

```yaml
metrics:
  - {name: pod, threshold: {value: 5000, unit: "[cft_i]/s"}}   # events: value >= 5000 cfs
  - {name: csi, threshold: {primary_quantile: 0.9}, as: csi_q90}   # of the evaluated window
  - {name: far, threshold: {column: q90}}                 # a per-pair column, e.g. a percentile field
  - {name: nmae, normalize_by: {value: 150, unit: MW}}    # e.g. installed capacity
  - {name: nrmse, normalize_by: range_primary}
  - {name: fdc_slope, side: secondary, exceedance: [0.2, 0.7]}
  - {name: annual_peak_relative_bias, year_start_month: 10}   # water years
  - {name: center_of_timing, missing_day_threshold: 0.2}
  - {name: nse, transform: {name: log, offset: {value: 0.01, unit: "m3/s"}}, as: log_offset_nse}
```

| Option | Metrics | Values |
|--------|---------|--------|
| `threshold` | contingency (`hits` … `frequency_bias`); required | a value with its unit, `{value: v, unit: u}`; `{primary_quantile: q}` (that quantile of each group's observations); `{column: name}` (a numeric pair column, one threshold per pair) |
| `normalize_by` | `nmae`, `nrmse` | `mean_primary` (default), `range_primary`, a value with its unit (`{value: v, unit: u}`), or `{column: name}` (the group mean of that column) |
| `side` | signatures | `primary` (default) or `secondary`; the column is named `<metric>_<side>` |
| `year_start_month` | `annual_peak_relative_bias`, `center_of_timing`, `standard_deviation_of_timing` | 1 to 12, the month a year starts in: default 1 (calendar years) for annual peaks, 10 (US water years) for the timing signatures |
| `missing_day_threshold` | `center_of_timing`, `standard_deviation_of_timing` | 0 to 1, default 0.1: a year missing more than this fraction of its days is left out |
| `exceedance` | `fdc_slope` | `[e1, e2]`, default `[0.33, 0.66]` |
| `log` | `fdc_slope` | `true` (default): the log form, (ln Q(e1) − ln Q(e2)) / (e2 − e1) (Sawicz et al. 2011); `false`: the linear slope |
| `estimator` | `crps` | `standard` (default) or `fair` (unbiased for the ensemble size) |
| `interval` | `interval_coverage`, `interval_width`, `winkler_score` | the central interval's quantile levels, default `[0.05, 0.95]` |
| `rank` | `rank_frequency` | the observation's rank among the members: `0` below all, `m` above all |
| `values` | any (first stage) | `{primary: column, secondary: column}`: compute the metric on these pair columns instead of `primary_value` and `secondary_value`, e.g. change fields for ramps. The column is named `<metric>_<primary column>` unless `as` is given |

**Values carry their unit.** A threshold or normalizer that's a number is written with its unit, a UCUM symbol from the units table (`{value: 300, unit: MW}`, `{value: 5000, unit: "[cft_i]/s"}`); a bare number is refused. Building the plan converts it to the evaluation unit (`output.unit`, else the primary dataset's), so a spec means the same thing whatever unit its data are stored in. A threshold on change fields (a ramp) and a normalizer convert by scale alone: they're differences or magnitudes, so a 5 K ramp is a 5 Cel ramp. Quantile and column thresholds have no unit: they're already in the data's.

**Quantile sketches.** `relative_median` and `fdc_slope` take their quantiles from quantile sketches: each value counts as its 0.1% bucket's representative. Sketches merge, so runs can keep these metrics in state and rerun only changed months. The results are the same from every path (full, incremental, sharded, bootstrap), but differ slightly from the exact quantiles: `relative_median` by at most about 0.2%, and `fdc_slope` by at most about 0.006 (absolute, for the default exceedances). Which metrics merge is in the [metrics reference](../metrics.md).

A metric used twice with different options needs `as` for distinct column names. Contingency metrics don't take a `transform`. Timing metrics (`max_value_time_delta`, `annual_peak_relative_bias`, `max_value_time`, `center_of_timing`, `standard_deviation_of_timing`) use each pair's `value_time`; `max_value_time` returns a timestamp, `max_value_time_delta` is in seconds, and the timing signatures are in days. The timing signatures work on daily means (UTC days) year by year and average over the years, so they need every pair (they don't merge from state). Signatures are computed over complete pairs, the same sample as every other metric.

## Probabilistic forecasts

Ensemble members and quantile forecasts are paired one row per member or quantile, keeping the dataset's `member` or `quantile` column. A **forecast case** is one forecast of one series for one time: the rows sharing observed series, predicted dataset and series, reference time and value time.

- `crps` and `brier_score` score each case's ensemble, then average over cases. A deterministic forecast is a one-member ensemble (`crps` = MAE).
- Ensemble diagnostics: `ensemble_spread`, `ensemble_mean_rmse`, `spread_skill_ratio` (about 1 when the spread matches the error), `rank_outlier_fraction` (observation outside the ensemble's range; `2 / (m + 1)` if reliable), `rank_histogram_reliability` (0 is a flat histogram; null when a group mixes ensemble sizes) and `rank_frequency` (one per `rank`, with `as`, gives the rank histogram). When the observation ties members, its rank counts half of them, rounded down.
- `pinball_loss` averages over quantile rows; `interval_*` and `winkler_score` use each case's lower and upper quantiles (`interval`).
- Group by columns that are the same for a whole case (e.g. `primary_series_key`, `lead_time_bin`), not by `member`, unless you want each member scored on its own.
- Bootstrap resamples whole cases, so members of a forecast stay together, and `min_count` counts cases.

## Skill scores

A first stage can compare every predicted dataset with a reference dataset:

```yaml
aggregate:
  - group_by: [primary_series_key, secondary_dataset, lead_time_bin]
    metrics: [rmse, crps]
    skill: {reference: public.nwm31_medium_range_mem1, metrics: [rmse, crps], matched: true}
```

This adds `root_mean_square_error_skill = 1 - rmse / rmse_of_reference` (and `mean_crps_ensemble_skill`; skill `metrics` may name them by alias, `[rmse, crps]`), comparing rows that match on every `group_by` column except `secondary_dataset`; positive means better than the reference, and the reference's own rows get 0. `form: ratio` gives `root_mean_square_error_ratio = rmse / rmse_of_reference` instead: with `mean_absolute_error` and a seasonal-naive reference that is rMAE, the headline metric in electricity price forecasting (MASE scales by the naive forecast's in-sample error instead).

- **`matched: true`** compares each dataset with the reference only on observations both forecast (same observed series and `value_time` within a group), recomputing both scores on those. Use it when datasets cover different times or leads, e.g. a short-range forecast against a medium-range one issued less often; otherwise each is scored on its own forecasts. The metric columns themselves stay unmatched.
- Skill needs an error score (lower is better, 0 is perfect): MAE, MSE, RMSE, MAPE, sMAPE, nMAE, nRMSE, mean absolute relative error, CRPS, Brier, pinball loss, Winkler.
- The reference is any predicted input, including generated persistence, climatology and blend.
- **`test: dm`** adds a Diebold–Mariano test of each dataset against the reference: `dm_statistic` (negative: its errors are smaller) and `dm_p_value` (two-sided). Each dataset's loss (`loss: squared`, the default, or `absolute`) is averaged per observation and differenced with the reference's on observations both forecast. The variance allows for autocorrelated errors (Newey–West, `lag` time steps, default floor(4 (n/100)^(2/9)), within each observed series). Lag k pairs an observation's differential with the one exactly k time steps earlier in its series, the step being the forecasts' own (e.g. 1 h), so a missing hour leaves its pairs out rather than shifting the rest. A difference in RMSE with p < 0.05 is then unlikely to be noise. The test merges from sums per lag kept by month, so it runs on the bounded `state` path when the groups span series (not when grouped by `primary_series_key`, where it is bounded per series anyway).
- Skill columns don't get bootstrap intervals yet.

## Bootstrap

A first stage can add confidence intervals to its metrics:

```yaml
aggregate:
  - group_by: [primary_series_key, secondary_dataset, lead_time_bin]
    metrics: [rmse, nse, kge]
    bootstrap:
      method: circular_block      # iid | circular_block | stationary
      block: P7D                   # whole days (P7D), calendar months (P1M), another duration, or rows
      reps: 1000
      seed: 42
      quantiles: [0.05, 0.95]
```

This adds `root_mean_square_error_q05`, `root_mean_square_error_q95`, `nash_sutcliffe_efficiency_q05`, … (quantile 0.025 becomes `q2_5`); `count` gets no interval since every resample has the same number of pairs. Later stages can summarize these columns like any other.

| Option | Default | Meaning |
|--------|---------|---------|
| `method` | `circular_block` | `iid` resamples pairs independently. `circular_block` resamples fixed-length blocks of consecutive pairs, wrapping around at the end, which keeps autocorrelation (needed for time series). `stationary` (Politis & Romano) uses random block lengths averaging `block`. `year_block` (alias `gumboot`, the method's published name) resamples whole (water) years with replacement: each year keeps its seasonality, which suits long simulations. |
| `block` | required for `circular_block` and `stationary` | Whole days (`P7D`, `PT48H`) or calendar months (`P1M`, `P3M`): blocks of that many consecutive days or months of `value_time` (UTC), counted from a group's first to its last, empty ones included. Another duration (`PT36H`) is converted to rows per group from the group's density of pairs over its time span; a number is rows. |
| `reps` | 1000 | Resamples per group (10 to 10,000). |
| `seed` | 0 | Base seed. |
| `quantiles` | `[0.05, 0.95]` | Quantiles of the bootstrap distribution to report. |
| `sort_by` | `[value_time]` | Order of pairs within a group before blocks are drawn; the pair key (reference time, series, …) breaks ties. |
| `min_count` | 30 | Groups with fewer pairs get null intervals (with blocks of days or months, and `year_block`: pairs that have both values). |
| `year_start_month` | 10 | `year_block`: the month years start in (10 = US water year, 1 = calendar year). |
| `min_years` | 5 | `year_block`: groups spanning fewer years get null intervals. |

- **Shared resamples:** every metric in the stage uses the same resamples, so intervals are consistent across metrics.
- **Reproducible:** each group's resamples depend only on `seed` and the group's key values, not on how a run splits work across units or processes.
- **Undefined replicates:** a resample where a metric is undefined (e.g. constant observations) is left out of that metric's quantiles; a group with no defined resamples gets null.
- **From state:** blocks of whole days or months, and `year_block`, resample whole time units, so a saved run can draw them from its state per day or month instead of the pairs (VeriRace's estimate calls this the `state` path), with the same intervals. Runs then keep state by day for the whole record when blocks are days, and by month for older data otherwise. `iid`, blocks of rows or of other durations, a `sort_by` other than `value_time`, intervals of `max_value_time`, and metrics that need every pair keep the pair path; the estimate says which.

## Metrics

Each metric has one canonical name, the one its results are labelled with, and may have aliases (`rmse` for `root_mean_square_error`), accepted wherever a spec names a metric or a result column (`metrics`, `of`, skill and guard `metrics`, and result filters and sorts). An alias always means the same metric with its default options; an `as` may not be one. Names and aliases follow the shared metric standard.

Every metric, with its aliases, family, definition, parameters, result unit and whether it merges from state, is in the [metrics reference](../metrics.md), generated from the metric registry (`teehr metrics` and `verirace metrics` list it too). Every metric is computed from shared intermediate results (sums, centered sums of squares, sorted values), so asking for many metrics in one stage costs little more than asking for a few.

## Expressions

`expr`, `filter` and `conditions` levels are written in a small engine-neutral subset of standard SQL, the expression language the evaluation systems share, and are stored as written. Each is checked when the spec is loaded and transpiled to DuckDB when it runs. The subset is chosen so an expression means the same in DuckDB, Spark and Trino: column references, literals, arithmetic (`+ - * / %`), comparison and boolean operators, `CASE WHEN`, `IN`, `BETWEEN`, `LIKE`, `IS [NOT] NULL`, `||`, `CAST` to `DOUBLE`, `BIGINT`, `INT`, `VARCHAR`, `TEXT`, `BOOLEAN`, `DATE`, `TIMESTAMP` or `TIMESTAMPTZ` (`TIMESTAMP '2024-01-01'` is one), `INTERVAL '6' HOUR` (units `DAY`, `HOUR`, `MINUTE`, `SECOND`), `EXTRACT` of `YEAR`, `QUARTER`, `MONTH`, `DAY`, `HOUR` or `MINUTE`, and these functions:

`abs`, `ceil`, `coalesce`, `concat`, `date_trunc`, `day`, `exp`, `extract`, `floor`, `greatest`, `hour`, `least`, `length`, `ln`, `log10`, `lower`, `minute`, `month`, `nullif`, `pow`, `power`, `quarter`, `sqrt`, `substring`, `upper`, `year`.

Subqueries, table functions and any other function are rejected when the spec is loaded. Numbers in an input's `filter` are read in its variable's canonical unit; numbers in a pair-level `filter`, `expr` or level in the evaluation unit.

`log10(x)` and `log(base, x)` are allowed; one-argument `log` isn't, since it is base 10 in some engines and natural in others (use `ln` or `log10`). Functions that differ between engines are refused with the reason: `round` (ties), `dayofweek` and `week` (numbering), `date_diff` (argument order), `isnan`, `ILIKE`.

## Durations

Durations are ISO 8601 of fixed length: `PT15M`, `PT6H`, `P1D`, `P1DT6H`, `P2W`; zero is `PT0S`. Years and months vary in length, so they're refused, except where a whole number of calendar months is meant: a bootstrap `block` such as `P1M` or `P3M`. Lead-time bin labels stay short (`[0h, 6h)`): they're results, not inputs.

## Datasets

Specs refer to datasets registered in the system (`teehr datasets list`, `verirace datasets list`). Each declares the dataset's role (`observed`, `simulated`, `forecast`; crosswalks and derived datasets are auxiliary), variable, unit, statistic and default period; `teehr datasets show <name>` (or `verirace datasets show`) prints it. The engine's tests use a local catalog file with the same fields (`tests/fixtures/*/datasets.yaml`); that is test infrastructure, not a way to run specs.
