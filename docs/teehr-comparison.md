# teehr vs teehr-core: metric comparison

`scripts/compare_teehr.py` runs every metric teehr and teehr-core share on the same inputs
(flows, negatives, constant obs, constant pred, one pair, two pairs, obs summing to zero,
zeros, gaps) and lists where they differ. Run it with teehr's Python:

    /path/to/teehr/.venv/bin/python scripts/compare_teehr.py

## Results (2026-11-16, teehr 0.8.0 main, teehr-core with VeriRace 668ab7d)

39 shared metrics; 29 agree on every case. The rest, and what was decided
(teehr-shared-eval-spec `DECISIONS.md`):

| Metric | Difference | Outcome |
| --- | --- | --- |
| `relative_median` | within 0.1% | expected: snapped to sketch buckets on every path |
| `spearman_correlation` | teehr above 1 for small groups, 0.1% high otherwise | teehr bug (sample covariance over population std); fixed in teehr#873 |
| `nash_sutcliffe_efficiency`, `_normalized` | teehr null when obs sum to 0 | teehr bug (unneeded guard); fixed in teehr#873 |
| `success_ratio` | different everywhere | teehr computed accuracy: the core's definition stands; teehr's is the new `proportion_correct` |
| `flow_duration_curve_slope` | different everywhere | teehr's was a linear slope at other quantiles: the core's log form stands; the linear slope is `log: false` |
| `average`, `count`, `sum`, `variance` | differ only with gaps | the core uses complete pairs (the sample every metric uses); teehr used each side's own values |

teehr only: `center_of_timing`, `standard_deviation_of_timing` (being added).
Not yet compared: `mean_crps_ensemble`, `mean_brier_score` (ensemble inputs).
