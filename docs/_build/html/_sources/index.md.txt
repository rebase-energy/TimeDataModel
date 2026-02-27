# timedatamodel

A lightweight data model for time series data with pandas, numpy, and polars support.

## Features

- **TimeSeries** -- univariate time series with rich metadata (unit, frequency, location, ...)
- **TimeSeriesTable** -- multivariate time series sharing the same index
- **TimeSeriesCube** -- N-dimensional time series with named dimensions and label-based selection
- **TimeSeriesCollection** -- heterogeneous container for series that don't share an index
- Built-in conversions to/from **pandas**, **numpy**, and **polars**
- Enum-based frequency (ISO 8601 durations) and data-type annotations
- Optional **pint** unit support and **shapely** geo-location metadata

## Installation

```bash
pip install timedatamodel
```

With optional extras:

```bash
pip install timedatamodel[pandas]       # pandas support
pip install timedatamodel[polars]       # polars support
pip install timedatamodel[pint]         # pint unit support
pip install timedatamodel[geo]          # shapely geo support
pip install timedatamodel[all]          # everything
```

## Quick start

```python
from datetime import datetime
from timedatamodel import TimeSeries, Frequency

ts = TimeSeries(
    Frequency.PT1H,
    timestamps=[datetime(2024, 1, 1, h) for h in range(24)],
    values=[float(h) for h in range(24)],
    name="temperature",
    unit="degC",
)

# Convert to pandas / numpy
df = ts.df
arr = ts.arr

# Arithmetic
ts_doubled = ts * 2
```

```{toctree}
:maxdepth: 2
:caption: Contents

api
```
