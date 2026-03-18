# timedatamodel

A lightweight Pythonic data model for time series data, interoperable with NumPy, Pandas and Polars.

## Features

- **TimeSeries** — univariate time series with rich metadata (name, unit, frequency, timezone, location, …)
- **TimeSeriesTable** — multivariate time series sharing the same `valid_time` index, with per-column metadata
- **Four data shapes** — `SIMPLE`, `VERSIONED`, `CORRECTED`, `AUDIT` — model everything from standard point-in-time data to full bi-temporal audit trails
- Full interoperability with **pandas**, **NumPy**, **Polars**, and **PyArrow** via `from_*` / `to_*` methods
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
