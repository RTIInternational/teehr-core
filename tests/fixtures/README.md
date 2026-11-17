# Test fixtures

Small, committed datasets in the shape of the [time series contract](../../docs/design/initial_design.md#31-time-series-contract). Each directory has a `datasets.yaml` describing its datasets (role, variable, unit, statistic, period, source).

| Fixture | Contents | Built by |
|---------|----------|----------|
| `hydro/` | USGS streamflow (5-min and 15-min, provisional) for 3 gages, 2026-10-06 to 2026-10-08; NWM v3 short-range (2 runs × 18 h) and medium-range mem1 (1 run × 48 h) for the crosswalked reaches; USGS–NWM crosswalk from NLDI | `scripts/fixtures/build_hydro_fixture.py` |
| `energy/` | Synthetic hourly load for 2 zones (7 days) and a day-ahead forecast (leads 14–37 h); pairs on the `zone` attribute | `scripts/fixtures/build_energy_fixture.py` |

Notes:
- `hydro/` is a snapshot. USGS data is provisional and NWM files expire from the public bucket, so a rebuild later will differ. Rebuild only on purpose.
- `hydro/` includes real edge cases worth keeping: observed and forecast units differ (ft³/s vs. m³/s), gages report at different spacings (their `resolution`), and one gage has constant observations over the first forecast hours.
- `energy/` rebuilds byte-for-byte from its fixed seed.

Rebuild:

```bash
uv run --group fixtures python scripts/fixtures/build_hydro_fixture.py   # ~6 min, network
uv run --group fixtures python scripts/fixtures/build_energy_fixture.py
uv run --group fixtures python scripts/fixtures/compute_golden.py        # then review the diff
```
