# Basic Usage

## Creating a TimeSeries

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
```

## Converting to pandas / numpy

```python
df = ts.df
arr = ts.arr
```

## Arithmetic

```python
ts_doubled = ts * 2
```

## Integrations

Each class provides `to_X`, `from_X`, and `apply_X` bridges to popular array and dataframe libraries.

| | numpy | pandas | polars | xarray |
| :--- | :---: | :---: | :---: | :---: |
| **TimeSeries** | | | | |
| &nbsp;&nbsp;`to_X` | ✅ | ✅ | ✅ | ✅ |
| &nbsp;&nbsp;`from_X` | — | ✅ | ✅ | ✅ |
| &nbsp;&nbsp;`apply_X` | ✅ | ✅ | ✅ | ✅ |
| **TimeSeriesTable** | | | | |
| &nbsp;&nbsp;`to_X` | ✅ | ✅ | ✅ | ✅ |
| &nbsp;&nbsp;`from_X` | — | ✅ | — | ✅ |
| &nbsp;&nbsp;`apply_X` | ✅ | ✅ | ✅ | ✅ |
| **TimeSeriesCube** | | | | |
| &nbsp;&nbsp;`to_X` | ✅ | ✅ | — | ✅ |
| &nbsp;&nbsp;`from_X` | ✅ | — | — | ✅ |
| &nbsp;&nbsp;`apply_X` | — | ✅¹ | ✅¹ | ✅ |

¹ Gated: raises `ValueError` if the cube has more than 2 non-time dimensions.
