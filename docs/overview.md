# Overview

**TimeDataModel** is a lightweight Python library for working with time series data. It provides
a metadata-rich container and a matching pure-metadata descriptor — backed by
[Polars](https://pola.rs) internally, and fully interoperable with pandas, NumPy, Polars, and
PyArrow — together with enums and geographic types for annotating your data.

The library is designed for energy, weather, and forecasting workflows where data arrives from
many sources at different times — but it is general enough for any domain that works with
timestamped numerical data.

---

## Core data structures

### TimeSeries

A single univariate time series. The underlying Polars DataFrame always has a `"value"` column
and one or more timestamp columns whose layout is determined by the inferred `DataShape`
(see below).

```python
import pandas as pd
from timedatamodel import TimeSeries, Frequency

df = pd.DataFrame({
    "valid_time": pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC"),
    "value": [float(i) for i in range(48)],
})

ts = TimeSeries.from_pandas(
    df,
    frequency=Frequency.PT1H,
    name="wind_power",
    unit="MW",
)
```

Supported metadata fields: `name`, `unit`, `data_type`, `timeseries_type`, `frequency`,
`timezone`, `description`.

**Key operations:**

| Operation | Example |
|-----------|---------|
| Slicing | `ts.head(n)`, `ts.tail(n)` |
| Unit conversion | `ts.convert_unit("kW")` *(requires `[pint]`)* |
| Format export | `ts.to_pandas()`, `ts.to_polars()`, `ts.to_list()`, `ts.to_numpy()`, `ts.to_pyarrow()` |
| Format import | `TimeSeries.from_pandas(df, ...)`, `from_polars(...)`, `from_list(...)`, `from_numpy(...)`, `from_pyarrow(...)` |
| Validation | `ts.validate_for_insert()` |

---

### TimeSeriesDescriptor

A frozen, data-free companion to `TimeSeries`. It carries every metadata field a
`TimeSeries` can have — `name`, `unit`, `data_type`, `timeseries_type`, `frequency`,
`timezone`, `description` — but no DataFrame. Use it to register or catalog the
*structure* of a series before any data exists, then materialize a full `TimeSeries`
once a DataFrame is in hand.

```python
from timedatamodel import TimeSeriesDescriptor, TimeSeries, DataType

desc = TimeSeriesDescriptor(
    name="wind_power",
    unit="MW",
    data_type=DataType.FORECAST,
)

# Later, once data arrives:
ts = TimeSeries.from_descriptor(desc, df)

# And back again, without the DataFrame:
desc_again = ts.to_descriptor()
```

`DataShape` is **not** encoded in the descriptor — it is inferred from the DataFrame at
`from_descriptor` time, so a single descriptor can be paired with any supported shape.

The descriptor is metadata-only. Identity-shaped concepts (labels, tags) and spatial
context (locations) belong to consumer layers (e.g. `energydatamodel`, `energydb`),
not to time-series metadata itself.

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

## Geographic types

Geographic primitives are exported from the package for consumer layers (e.g.
`energydatamodel`) that need to attach spatial context to entities. They are no longer
attached to `TimeSeries` itself.

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
- {doc}`examples/index` — hands-on notebooks
- {doc}`api` — full API reference
