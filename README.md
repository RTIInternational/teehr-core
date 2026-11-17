# teehr-core

The dependency-light core of [TEEHR](https://github.com/RTIInternational/teehr): the canonical
data model, metric kernels, mergeable metric state, the bootstrap, and the evaluation spec and
the engine that runs it. numpy and pyarrow; no Spark, no JVM.

It is the one definition of each metric that `teehr` (including its Spark path for the largest
jobs), TEEHR Cloud and other evaluation systems share, so the same evaluation gives the same
numbers wherever it runs. The shared rules it implements (metric names and definitions, metric
state, the bootstrap procedure, units, the spec) are agreed in the TEEHR shared evaluation spec.

**Status:** 0.1: metrics, state, bootstrap, data model, units, the spec (v1) and the engine. The
design is in teehr-cloud-core `docs/design/teehr-core.md`. Conformance with the shared spec's
cases runs in CI: `python -m teehr_core.conformance <teehr-shared-eval-spec>/conformance/cases`.

## Install

```
pip install teehr-core            # metrics, state, bootstrap, spec
pip install "teehr-core[engine]"  # plus the spec engine (DuckDB)
```

## Releasing

Bump `version` in `pyproject.toml`, merge, then publish a GitHub release tagged `v<version>`;
the `Release` workflow tests, builds and publishes to PyPI.

## Acknowledgements

Much of teehr-core (the metric registry, components and state, the bootstrap, the spec models and
engine) is derived from **VeriRace**, an evaluation system developed alongside TEEHR, and is
published here with its author's permission. Files ported from VeriRace say so in their
docstring.

## License

MIT (see `LICENSE`).
