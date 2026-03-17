# Overview

**TimeDataModel** is a lightweight Python library for working with time series data. It provides
two structured, metadata-rich containers backed by [Polars](https://pola.rs), together with
enums and geographic types for annotating your data.

The library is designed for energy, weather, and forecasting workflows where data arrives from
many sources at different times — but it is general enough for any domain that works with
timestamped numerical data.

---

## Core data structures

### TimeSeriesPolars

A single univariate time series backed by a Polars DataFrame. The DataFrame always has a
`"value"` column and one or more timestamp columns whose layout is determined by the chosen
`DataShape` (see below).

```python
import pandas as pd
from timedatamodel import TimeSeriesPolars, DataShape, Frequency

df = pd.DataFrame({
    "valid_time": pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC"),
    "value": [float(i) for i in range(48)],
})

ts = TimeSeriesPolars.from_pandas(
    df,
    shape=DataShape.SIMPLE,
    frequency=Frequency.PT1H,
    name="wind_power",
    unit="MW",
)
```

Supported metadata fields: `name`, `frequency`, `timezone`, `unit`, `data_type`, `location`,
`description`, `timeseries_type`, `labels`.

**Key operations:**

| Operation | Example |
|-----------|---------|
| Arithmetic | `ts * 2`, `ts + other`, `ts / other` |
| Slicing | `ts.head(n)`, `ts.tail(n)` |
| Unit conversion | `ts.convert_unit("kW")` *(requires `[pint]`)* |
| Pandas round-trip | `ts.to_pandas()`, `TimeSeriesPolars.from_pandas(df, ...)` |
| Validation | `ts.validate_for_insert()` |

---

### TimeSeriesTablePolars

Multiple co-indexed time series stored as named columns in a single Polars DataFrame. All
columns share the same `valid_time` index. Each column carries independent metadata — unit,
data type, and geographic location.

```python
from timedatamodel import TimeSeriesTablePolars

# Build from a list of TimeSeriesPolars (all must have identical valid_time values)
table = TimeSeriesTablePolars.from_timeseries(
    [ts_wind, ts_solar, ts_load],
    frequency=Frequency.PT1H,
)

# Select a single column back as a TimeSeriesPolars
ts_wind = table.select_column("wind_power")

# Spatial filtering (requires [geo] extra)
nearby = table.filter_by_location(center, radius_km=50)
nearest_col = table.nearest(center)
```

**Key operations:** `select_column`, `filter_by_location`, `nearest`, `from_timeseries`,
`to_timeseries_list`, `from_pandas`, `to_pandas`, `head`, `tail`.

---

## Data shapes

The `DataShape` enum controls which timestamp columns are present in the underlying DataFrame.
This lets a single class cover standard time series as well as bi-temporal and audit-trail
patterns:

| Shape | Columns | Use case |
|-------|---------|----------|
| `SIMPLE` | `valid_time`, `value` | Standard time series |
| `VERSIONED` | `knowledge_time`, `valid_time`, `value` | Bi-temporal: record *when* each value was produced |
| `CORRECTED` | `valid_time`, `change_time`, `value` | Track *when* a value was revised |
| `AUDIT` | `knowledge_time`, `change_time`, `valid_time`, `value` | Full audit trail |

All timestamp columns are stored as `pl.Datetime("us", time_zone="UTC")`.

---

## Frequency

The `Frequency` enum defines ISO 8601 duration values. Three are **calendar-based** (their
exact duration depends on the calendar); the rest are **fixed-interval**:

| Value | Description | Calendar-based |
|-------|-------------|:--------------:|
| `P1Y` | 1 year | yes |
| `P3M` | 3 months (quarter) | yes |
| `P1M` | 1 month | yes |
| `P1W` | 1 week | |
| `P1D` | 1 day | |
| `PT1H` | 1 hour | |
| `PT30M` | 30 minutes | |
| `PT15M` | 15 minutes | |
| `PT10M` | 10 minutes | |
| `PT5M` | 5 minutes | |
| `PT1M` | 1 minute | |
| `PT1S` | 1 second | |
| `NONE` | No fixed frequency | |

---

## DataType taxonomy

Every time series can carry a `DataType` annotation describing the nature of the data. The
taxonomy has two roots and 10 leaf types:

```
ACTUAL
├── OBSERVATION ── directly measured or recorded values
└── DERIVED ────── computed from observations (e.g. aggregated measurements)

CALCULATED
├── ESTIMATION
│   ├── FORECAST ─────── future predictions with a known issue time
│   ├── PREDICTION ───── statistical/ML predictions (general)
│   ├── SCENARIO ─────── what-if analyses under assumed conditions
│   ├── SIMULATION ───── outputs from physical or numerical models
│   └── RECONSTRUCTION ─ hindcasts or gap-filled historical data
└── REFERENCE
    ├── BASELINE ──────── reference scenario for comparison
    ├── BENCHMARK ─────── industry-standard or best-practice reference
    └── IDEAL ─────────── theoretical optimum or design values
```

Each `DataType` member exposes `.parent`, `.children`, `.is_leaf`, and `.root` properties.

---

## Geographic support

Time series can carry geographic metadata via the `location` field, which accepts either a
`GeoLocation` or `GeoArea`.

**GeoLocation** — a single point defined by `latitude` and `longitude`:

```python
from timedatamodel import GeoLocation

loc = GeoLocation(latitude=59.33, longitude=18.07)  # Stockholm
```

Provides `distance_to()` (Haversine), `bearing_to()`, `midpoint()`, and `offset()`.

**GeoArea** — a polygon region (requires the `[geo]` extra):

```python
from timedatamodel import GeoArea

area = GeoArea.from_coordinates([(59.0, 17.5), (59.0, 18.5), (59.5, 18.5), (59.5, 17.5)])
```

Provides `contains_point()`, `overlaps()`, `centroid`, and `bounding_box()`.

`TimeSeriesTablePolars` supports spatial queries: `filter_by_location()` and `nearest()`.

---

## Other types

### DataPoint

A `NamedTuple` with two fields: `timestamp` (a `datetime`) and `value` (a `float` or `None`).

### TimeSeriesType

Describes the index structure of a time series:

- `FLAT` — standard non-overlapping timestamps
- `OVERLAPPING` — overlapping windows (e.g. rolling forecasts with a `(issue_time, valid_time)` multi-index)

---

## Next steps

- {doc}`usage` — quick-start examples for common operations
- {doc}`tutorials/index` — hands-on notebooks
- {doc}`api` — full API reference
