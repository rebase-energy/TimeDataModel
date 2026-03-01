# timedatamodel

A lightweight data model for time series data with pandas, numpy, and polars support.

## Features

- **TimeSeries** -- univariate time series with rich metadata (unit, frequency, location, ...)
- **TimeSeriesTable** -- multivariate time series sharing the same index
- **TimeSeriesCube** -- N-dimensional time series with named dimensions and label-based selection
- **TimeSeriesCollection** -- heterogeneous container for series that don't share an index
- Built-in conversions to/from **pandas**, **numpy**, and **polars**
- Enum-based frequency (ISO 8601 durations) and data-type annotations
- **Hierarchical time series** -- define aggregation trees and reconcile forecasts
- Optional **pint** unit support and **shapely** geo-location metadata

```{toctree}
:maxdepth: 1
:caption: Getting Started

installation
usage
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/getting_started
tutorials/numpy_and_pandas_transforms
tutorials/unit_handling_and_validation
tutorials/timeseries_operations
tutorials/multivariate_timeseries
tutorials/cubes_and_collections
tutorials/data_quality_and_coverage
tutorials/io_and_interoperability
tutorials/geographical_support
tutorials/hierarchical_timeseries
```

```{toctree}
:maxdepth: 1
:caption: API Reference

api
```
