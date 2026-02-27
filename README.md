# TimeDataModel

A lightweight Python data model for time series data.

## Features

- `TimeSeries` class with typed timestamps and float/None values
- Flexible `Resolution` (frequency + timezone) and `Metadata`
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
from timedatamodel import TimeSeries, Resolution, Metadata
from timedatamodel.enums import Frequency

res = Resolution(Frequency.PT1H, "UTC")
ts = TimeSeries(
    resolution=res,
    metadata=Metadata(name="power", unit="kW"),
    timestamps=[datetime(2024, 1, 1, i, tzinfo=timezone.utc) for i in range(3)],
    values=[100.0, 110.0, 95.0],
)

df = ts.to_pandas_dataframe()
```

## Requirements

Python >= 3.11
