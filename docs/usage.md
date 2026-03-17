# Basic Usage

## Creating a TimeSeriesPolars

### From a pandas DataFrame

```python
import pandas as pd
from timedatamodel import TimeSeriesPolars, DataShape, Frequency

df = pd.DataFrame({
    "valid_time": pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"),
    "value": [float(i) for i in range(24)],
})

ts = TimeSeriesPolars.from_pandas(
    df,
    shape=DataShape.SIMPLE,
    frequency=Frequency.PT1H,
    name="wind_power",
    unit="MW",
)
```

### From a Polars DataFrame directly

```python
import polars as pl
from timedatamodel import TimeSeriesPolars, DataShape, Frequency

df = pl.DataFrame({
    "valid_time": pl.Series(
        pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    ),
    "value": [float(i) for i in range(24)],
})

ts = TimeSeriesPolars(
    df,
    shape=DataShape.SIMPLE,
    frequency=Frequency.PT1H,
    name="wind_power",
    unit="MW",
)
```

## Arithmetic

```python
ts_doubled = ts * 2
ts_shifted = ts + 50.0
ts_ratio   = ts_a / ts_b
```

## Slicing

```python
first_six = ts.head(6)
last_six  = ts.tail(6)
```

## Unit conversion

Requires the `[pint]` extra:

```python
ts_kw = ts.convert_unit("kW")
```

## Converting back to pandas

```python
df = ts.to_pandas()   # DatetimeTZDtype index, UTC-aware
```

## Multivariate tables

```python
from timedatamodel import TimeSeriesTablePolars

# Build from a list of TimeSeriesPolars (must share identical valid_time values)
table = TimeSeriesTablePolars.from_timeseries(
    [ts_wind, ts_solar],
    frequency=Frequency.PT1H,
)

# Select a single column back as a TimeSeriesPolars
ts_wind = table.select_column("wind_power")

# Round-trip to pandas (valid_time becomes the index)
df = table.to_pandas()
```

## Versioned (bi-temporal) series

```python
import pandas as pd
from timedatamodel import TimeSeriesPolars, DataShape, Frequency

df = pd.DataFrame({
    "knowledge_time": pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"),
    "valid_time":     pd.date_range("2024-01-02", periods=24, freq="h", tz="UTC"),
    "value":          [float(i) for i in range(24)],
})

ts = TimeSeriesPolars.from_pandas(
    df,
    shape=DataShape.VERSIONED,
    frequency=Frequency.PT1H,
    name="forecast",
)
```
