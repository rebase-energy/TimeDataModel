# Basic Usage

## Creating a TimeSeries

### From a pandas DataFrame

```python
import pandas as pd
from timedatamodel import TimeSeries, Frequency

df = pd.DataFrame({
    "valid_time": pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"),
    "value": [float(i) for i in range(24)],
})

ts = TimeSeries.from_pandas(
    df,
    frequency=Frequency.PT1H,
    name="wind_power",
    unit="MW",
)
```

### From a Polars DataFrame directly

```python
import polars as pl
from timedatamodel import TimeSeries, Frequency

df = pl.DataFrame({
    "valid_time": pl.Series(
        pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    ),
    "value": [float(i) for i in range(24)],
})

ts = TimeSeries.from_polars(
    df,
    frequency=Frequency.PT1H,
    name="wind_power",
    unit="MW",
)
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

## Format conversions

All export methods return the **full table** (all columns including timestamps).

```python
# Always available (polars is the base dependency)
df_pl  = ts.to_polars()       # pl.DataFrame
cols   = ts.to_list()         # dict[str, list] — column-oriented, datetime objects for timestamps

# Optional dependencies — raises ImportError with install hint if not present
arr = ts.to_numpy()    # dict[str, np.ndarray] — column-oriented; requires numpy
tbl = ts.to_pyarrow()  # pa.Table; requires pyarrow

# Pandas interop (requires timedatamodel[pandas])
df = ts.to_pandas()    # DatetimeTZDtype index per shape
```

`to_pandas()` restores the conventional index per shape — see the [API reference](api.md).

Each export format has a matching constructor:

```python
ts = TimeSeries.from_polars(df_pl, ...)
ts = TimeSeries.from_list(cols, ...)
ts = TimeSeries.from_numpy(arr, ...)    # requires numpy
ts = TimeSeries.from_pyarrow(tbl, ...)  # requires pyarrow
ts = TimeSeries.from_pandas(df, ...)    # requires pandas
```

## Coverage bar

```python
cb = ts.coverage_bar()   # CoverageBar — renders as SVG in Jupyter, Unicode blocks in terminal
```

For `TimeSeriesTable`, one bar row is shown per value column.

## Multivariate tables

```python
from timedatamodel import TimeSeriesTable, Frequency
import pandas as pd

table = TimeSeriesTable.from_pandas(
    pd.DataFrame({
        "valid_time":  pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"),
        "wind_power":  [float(i) for i in range(24)],
        "solar_power": [float(i) * 0.5 for i in range(24)],
    }),
    frequency=Frequency.PT1H,
)

# Select a single column back as a TimeSeries
ts_wind = table.select_column("wind_power")

# Round-trip to pandas (valid_time becomes the index)
df = table.to_pandas()
```

## Versioned (bi-temporal) series

```python
import pandas as pd
from timedatamodel import TimeSeries, Frequency

df = pd.DataFrame({
    "knowledge_time": pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"),
    "valid_time":     pd.date_range("2024-01-02", periods=24, freq="h", tz="UTC"),
    "value":          [float(i) for i in range(24)],
})

ts = TimeSeries.from_pandas(
    df,
    frequency=Frequency.PT1H,
    name="forecast",
)
```
