# Overview

**TimeDataModel** is a lightweight Python data model for time series data. It provides a set of
structured containers — from single univariate series to N-dimensional arrays and hierarchical
trees — with built-in metadata (units, frequency, location, data type) and native conversions
to pandas, numpy, polars, and xarray.

The library is designed for energy and weather data workflows where you need to carry metadata
alongside your numeric data, but it is general enough for any domain that works with time series.

## Core data structures

TimeDataModel offers five data structures, each suited to a different shape of time series data.

### TimeSeriesList

A single univariate time series: one sequence of timestamps paired with one sequence of values,
plus metadata fields like `name`, `unit`, `frequency`, `data_type`, and `location`.

```python
ts = TimeSeriesList(
    timestamps=[datetime(2024, 1, 1), datetime(2024, 1, 2)],
    values=[10.5, 12.3],
    frequency=Frequency.P1D,
    name="temperature",
    unit="degC",
)
```

Supports element-wise arithmetic (`+`, `-`, `*`, `/`), iteration over `DataPoint` named tuples,
and conversion to/from pandas, polars, numpy, and xarray.

### TimeSeriesTable

Multiple columns sharing the same timestamp index — a multivariate time series backed by a 2D
numpy array. Each column carries its own metadata (name, unit, data type, location).

```python
table = TimeSeriesTable(
    timestamps=[datetime(2024, 1, 1), datetime(2024, 1, 2)],
    values=np.array([[1.0, 2.0], [3.0, 4.0]]),
    frequency=Frequency.P1D,
    names=["solar", "wind"],
    units=["MW", "MW"],
)
```

Use `select_column()` to extract a single `TimeSeriesList`, or spatial filtering methods like
`filter_columns_by_location()` to select columns by geographic proximity.

### TimeSeriesArray

An N-dimensional time series with named dimensions and label-based selection. Values are stored
as a numpy masked array. Useful for forecast ensembles, weather grids, or any data with more
than two axes.

```python
array = TimeSeriesArray.from_numpy(
    dimensions=[
        Dimension("valid_time", [datetime(2024, 1, 1), datetime(2024, 1, 2)]),
        Dimension("model", ["A", "B", "C"]),
    ],
    values=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    frequency=Frequency.P1D,
)
```

Use `sel()` and `isel()` for label-based and index-based slicing. Selecting down to two
dimensions returns a `TimeSeriesTable`; selecting down to one returns a `TimeSeriesList`.

### TimeSeriesCollection

A heterogeneous container for series that don't share the same index. Stores an ordered
dictionary of `TimeSeriesList` and/or `TimeSeriesTable` objects, each potentially with different
frequencies, timezones, or time ranges.

```python
collection = TimeSeriesCollection([ts_hourly, ts_daily, table_weekly])
```

Implements the mapping protocol (`len`, `in`, iteration, key/index access) and supports spatial
filtering across all contained series.

### HierarchicalTimeSeries

A tree of time series organised into named hierarchy levels, designed for aggregation and
reconciliation. Each leaf node holds a `TimeSeriesList`; inner nodes aggregate their children
using a configurable `AggregationMethod` (sum, mean, min, max).

```python
hierarchy = HierarchicalTimeSeries.from_dict(
    tree={"Total": {"Region_A": ["Site_1", "Site_2"], "Region_B": ["Site_3"]}},
    series_map={"Site_1": ts1, "Site_2": ts2, "Site_3": ts3},
    levels=["country", "region", "site"],
)
```

Use `aggregate()` to roll up values bottom-up, `get_level()` to retrieve all nodes at a
hierarchy level, and `to_collection()` or `to_table()` to flatten back to other structures.

## DataType taxonomy

Every time series can carry a `DataType` annotation describing the nature of the data. The
taxonomy has two roots and 12 leaf/branch types:

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

Each `DataType` member exposes `.parent`, `.children`, `.is_leaf`, and `.root` properties for
navigating the hierarchy programmatically.

## Frequency

The `Frequency` enum defines 13 ISO 8601 duration values. Three are **calendar-based** (their
exact duration depends on the calendar), and the rest are **fixed-interval**:

| Value    | Description        | Calendar-based |
|----------|--------------------|:--------------:|
| `P1Y`    | 1 year             | yes            |
| `P3M`    | 3 months (quarter) | yes            |
| `P1M`    | 1 month            | yes            |
| `P1W`    | 1 week             |                |
| `P1D`    | 1 day              |                |
| `PT1H`   | 1 hour             |                |
| `PT30M`  | 30 minutes         |                |
| `PT15M`  | 15 minutes         |                |
| `PT10M`  | 10 minutes         |                |
| `PT5M`   | 5 minutes          |                |
| `PT1M`   | 1 minute           |                |
| `PT1S`   | 1 second           |                |
| `NONE`   | No fixed frequency |                |

Use `.is_calendar_based` to check, and `.to_timedelta()` to get a `timedelta` (returns `None`
for calendar-based frequencies and `NONE`).

## Other enums and concepts

### TimeSeriesType

Describes whether a time series has a flat (non-overlapping) or overlapping index:

- `FLAT` — standard time series with non-overlapping timestamps
- `OVERLAPPING` — overlapping windows, e.g. rolling forecasts with a multi-index of
  `(issue_time, valid_time)`

### AggregationMethod

Used by `HierarchicalTimeSeries` to define how children are aggregated into parents:

- `SUM` — sum of children (default)
- `MEAN` — arithmetic mean
- `MIN` — minimum value
- `MAX` — maximum value

### DataPoint

A lightweight `NamedTuple` with two fields:

- `timestamp` — a `datetime` object
- `value` — a `float` or `None`

Returned when indexing or iterating over a `TimeSeriesList`.

## Geographic support

Time series can carry geographic metadata via the `location` field, which accepts either a
`GeoLocation` or a `GeoArea`.

**GeoLocation** — a single point defined by `latitude` and `longitude`:

```python
loc = GeoLocation(latitude=59.33, longitude=18.07)  # Stockholm
```

Provides methods for `distance_to()` (Haversine), `bearing_to()`, `midpoint()`, and `offset()`.

**GeoArea** — a polygon region (requires the `[geo]` extra for shapely):

```python
area = GeoArea.from_coordinates([(59.0, 17.5), (59.0, 18.5), (59.5, 18.5), (59.5, 17.5)])
```

Provides `contains_point()`, `overlaps()`, `centroid`, and `bounding_box()`.

Both `TimeSeriesTable` and `TimeSeriesCollection` support spatial queries:
`filter_columns_by_location()`, `filter_by_area()`, and `nearest()`.

## Next steps

- {doc}`usage` — quick-start integrations with pandas, polars, and numpy
- {doc}`tutorials/getting_started` — hands-on tutorial building your first time series
- {doc}`tutorials/multivariate_timeseries` — working with tables and multiple columns
- {doc}`tutorials/arrays_and_collections` — N-dimensional arrays and heterogeneous collections
- {doc}`tutorials/hierarchical_timeseries` — building and aggregating hierarchical trees
- {doc}`tutorials/geographical_support` — attaching locations and spatial filtering
- {doc}`api` — full API reference
