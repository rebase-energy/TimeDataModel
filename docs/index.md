# timedatamodel

A lightweight, metadata-rich data model for time series data, built on [Polars](https://pola.rs).

## Features

- **TimeSeriesPolars** — univariate time series with rich metadata (name, unit, frequency, timezone, location, …) backed by a Polars DataFrame
- **TimeSeriesTablePolars** — multivariate time series sharing the same `valid_time` index, with per-column metadata
- **Four data shapes** — `SIMPLE`, `VERSIONED`, `CORRECTED`, `AUDIT` — model everything from standard point-in-time data to full bi-temporal audit trails
- Built-in conversions to/from **pandas** (`from_pandas` / `to_pandas`)
- Enum-based **Frequency** (ISO 8601 durations) and **DataType** annotations
- Optional **pint** unit support and **shapely** geo-location metadata

```{toctree}
:maxdepth: 1
:caption: Getting Started

installation
overview
usage
```

```{toctree}
:maxdepth: 1
:caption: Examples

examples/index
```

```{toctree}
:maxdepth: 1
:caption: API Reference

api
```
