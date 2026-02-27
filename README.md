# TimeDataModel

A lightweight Python data model for time series data.

## Features

- TimeSeries class that can easily be converted to different data structures (lists/numpy/pandas/polars)
- Native bridges to pandas and numpy
- Geographical and hierarchical support for time series data
- Provides a nice repr-method for terminal and notebook (and potentially other plotting functions)


- `TimeSeries` class for univariate time series with scalar metadata
- `MultivariateTimeSeries` class for multi-column time series with list metadata
- Flexible `Resolution` (frequency + timezone)
- Geographic location support (`GeoLocation`, `GeoArea`)
- Native bridges to pandas DataFrames and numpy arrays
- Optional polars support
- JSON and CSV serialization

## Installation

```bash
pip install timedatamodel

# With polars support
pip install timedatamodel[polars]
```

## Quick Start

```python
from datetime import datetime, timezone
from timedatamodel import TimeSeries, Resolution
from timedatamodel.enums import Frequency

res = Resolution(Frequency.PT1H, "UTC")
ts = TimeSeries(
    resolution=res,
    timestamps=[datetime(2024, 1, 1, i, tzinfo=timezone.utc) for i in range(3)],
    values=[100.0, 110.0, 95.0],
    name="power",
    unit="kW",
)

df = ts.to_pandas_dataframe()
```

## Requirements

Python >= 3.11
