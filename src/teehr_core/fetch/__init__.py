"""Sources of observations and model outputs, as v1 timeseries rows (needs the ``fetch`` extra).

Each source fetches in whatever way suits it, but all of them return the same thing: Arrow
tables in v1 timeseries columns (``series_key``, ``value_time``, ``value``, plus
``reference_time``, ``member`` and so on where they apply), UTC times, values in the module's
``UNIT``, in batches. A source knows nothing of tables or catalogs. Writing the rows is always
:func:`teehr_core.warehouse.ingest.write_timeseries` (with ``unit=<module>.UNIT``), from a
scheduled flow or a person's notebook.

- :mod:`.usgs`: USGS streamflow observations (the USGS Water Data API).
- :mod:`.nwm`: NWM operational streamflow forecasts and analyses at reaches (NOAA's GCS bucket).
"""
