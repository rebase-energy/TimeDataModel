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
