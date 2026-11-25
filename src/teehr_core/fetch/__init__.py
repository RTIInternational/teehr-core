"""Sources of observations and model outputs, as v1 timeseries rows (needs the ``fetch`` extra).

A source only produces rows (Arrow tables of ``series_key``, ``value_time``, ``value`` and any
other key columns), in batches; it knows nothing of tables or catalogs. Writing them is
:func:`teehr_core.warehouse.ingest.write_timeseries`, from a scheduled flow or a person's notebook.

- :mod:`.usgs`: USGS streamflow observations (the USGS Water Data API).
- :mod:`.nwm`: NWM operational streamflow forecasts and analyses at reaches (NOAA's GCS bucket).
"""
