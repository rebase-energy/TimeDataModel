<div align="center">

# TimeDataModel

**A lightweight Python data model for time series data with native bridges to numpy, pandas, polars, and xarray.**

<a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square"></a>
<a href="https://pypi.org/project/timedatamodel/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/timedatamodel?color=blue&style=flat-square"></a>
<a href="https://pypi.org/project/timedatamodel/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/timedatamodel?style=flat-square"></a>
<a href="https://github.com/rebase-energy/TimeDataModel"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/rebase-energy/TimeDataModel?style=social"></a>

</div>

<br/>

**TimeDataModel** provides a structured, metadata-rich representation of time series data that lets you:

* 📐 **Structure** - Represent univariate, multivariate, N-dimensional, and hierarchical time series as typed Python objects;
* 🔄 **Convert** - Seamlessly convert between pandas DataFrames, numpy arrays, and polars DataFrames;
* 🗺️ **Locate** - Attach geographic coordinates and areas to time series with spatial filtering;
* 📏 **Measure** - Track and convert physical units with automatic dimensional validation;
* 📊 **Validate** - Inspect data quality with coverage bars, gap detection, and consistency checks;
* 💾 **Serialize** - Read and write time series to JSON and CSV with full metadata preservation.

**⬇️ [Installation](#installation)**
&ensp;|&ensp;
**📖 [Documentation](https://github.com/rebase-energy/TimeDataModel/tree/main/docs)**
&ensp;|&ensp;
**🚀 [Examples](https://github.com/rebase-energy/TimeDataModel/tree/main/examples)**

---

## 🧱 Core Data Classes

**TimeDataModel** represents time series data at multiple levels of complexity. The table below gives a summary of the available data classes.

| Class | Description |
| :---- | :---- |
| 📈&nbsp;`TimeSeries` | Univariate time series with scalar metadata (name, unit, frequency, timezone) |
| 📊&nbsp;`TimeSeriesTable` | Multivariate time series — multiple columns sharing the same timestamp index |
| 🧊&nbsp;`TimeSeriesCube` | N-dimensional time series with named dimensions and label-based selection |
| 📦&nbsp;`TimeSeriesCollection` | Heterogeneous container for series with different indices |
| 🌳&nbsp;`HierarchicalTimeSeries` | Tree-structured time series with aggregation across hierarchy levels |
| 🗺️&nbsp;`GeoLocation` / `GeoArea` | Geographic point and polygon types with distance, bearing, and containment |
| ⏱️&nbsp;`Frequency` | ISO 8601 duration-based frequencies (`PT1H`, `P1D`, `P1M`, etc.) |
| 🏷️&nbsp;`DataType` | Classification enum: `MEASUREMENT`, `FORECAST`, `SCENARIO`, `CLIMATE`, etc. |

---

## 🔌 Integrations

Each class provides `to_X`, `from_X`, and `apply_X` bridges to popular array and dataframe libraries.

### TimeSeries

| | numpy | pandas | polars | xarray |
| :--- | :---: | :---: | :---: | :---: |
| `to_X` | ✅ | ✅ | ✅ | ✅ |
| `from_X` | — | ✅ | ✅ | ✅ |
| `apply_X` | ✅ | ✅ | ✅ | ✅ |

### TimeSeriesTable

| | numpy | pandas | polars | xarray |
| :--- | :---: | :---: | :---: | :---: |
| `to_X` | ✅ | ✅ | ✅ | ✅ |
| `from_X` | — | ✅ | — | ✅ |
| `apply_X` | ✅ | ✅ | ✅ | ✅ |

### TimeSeriesCube

| | numpy | pandas | polars | xarray |
| :--- | :---: | :---: | :---: | :---: |
| `to_X` | ✅ | ✅ | — | ✅ |
| `from_X` | ✅ | — | — | ✅ |
| `apply_X` | — | ✅¹ | ✅¹ | ✅ |

¹ Gated: raises `ValueError` if the cube has more than 2 non-time dimensions.

---

## 🚀 Quick Start

```python
from datetime import datetime, timezone
from timedatamodel import TimeSeries, Frequency

# Create a univariate time series
ts = TimeSeries(
    Frequency.PT1H,
    timezone="UTC",
    timestamps=[datetime(2024, 1, 1, i, tzinfo=timezone.utc) for i in range(24)],
    values=[100.0 + i * 2.5 for i in range(24)],
    name="power_output",
    unit="MW",
)

# Convert to pandas DataFrame
df = ts.to_pandas_dataframe()

# Convert to numpy array
arr = ts.to_numpy()

# Arithmetic operations
ts_doubled = ts * 2
ts_shifted = ts + 50

# Inspect data quality
print(ts.coverage_bar())

# Serialize to JSON
json_str = ts.to_json()
```

---

## ✨ Key Features

- 📐 **Multi-dimensional** - From single series (`TimeSeries`) to N-dimensional cubes (`TimeSeriesCube`) with `.sel()` and `.isel()` for label- and index-based selection;
- 🔢 **Arithmetic** - Element-wise `+`, `-`, `*`, `/`, comparisons, and `abs()` with automatic NaN handling;
- 🗺️ **Geospatial** - Attach locations, compute distances (Haversine), filter by radius or area, and find nearest neighbors;
- 🌳 **Hierarchical** - Build tree structures with automatic aggregation (`SUM`, `MEAN`, `MIN`, `MAX`) and cross-level unit conversion;
- 📏 **Units** - Optional [pint](https://pint.readthedocs.io/) integration for physical unit tracking and conversion;
- 📊 **Data Quality** - Coverage bars (terminal + HTML), missing-value detection, and timestamp validation;
- 💾 **Serialization** - JSON and CSV I/O with full round-trip metadata preservation;
- 🐍 **Type-safe** - Full type hints with PEP 561 support.

---

## ⬇️ Installation

Install the **stable** release:
```bash
pip install timedatamodel
```

Install with **optional dependencies**:
```bash
pip install timedatamodel[pandas]    # pandas support
pip install timedatamodel[polars]    # polars support
pip install timedatamodel[geo]       # geospatial support (shapely)
pip install timedatamodel[pint]      # unit handling (pint)
pip install timedatamodel[all]       # everything
```

Install in editable mode for **development**:
```bash
git clone https://github.com/rebase-energy/TimeDataModel.git
cd TimeDataModel
pip install -e .[dev]
```

---

## 📓 Examples

Explore the library through the example notebooks:

| # | Notebook | Topic |
| :--- | :--- | :--- |
| 01 | [Getting Started](examples/nb_01_getting_started.ipynb) | Creating and inspecting time series |
| 02 | [NumPy & Pandas](examples/nb_02_numpy_and_pandas_transforms.ipynb) | Converting to and from numpy and pandas |
| 03 | [Unit Handling](examples/nb_03_unit_handling_and_validation.ipynb) | Physical unit conversion with pint |
| 04 | [Operations](examples/nb_04_timeseries_operations.ipynb) | Arithmetic, slicing, and indexing |
| 05 | [Multivariate](examples/nb_05_multivariate_timeseries.ipynb) | Working with multi-column time series |
| 06 | [Cubes & Collections](examples/nb_06_cubes_and_collections.ipynb) | N-dimensional data and heterogeneous containers |
| 07 | [Data Quality](examples/nb_07_data_quality_and_coverage.ipynb) | Coverage bars and validation |
| 08 | [I/O](examples/nb_08_io_and_interoperability.ipynb) | JSON and CSV serialization |
| 09 | [Geospatial](examples/nb_09_geographical_support.ipynb) | Locations, areas, and spatial filtering |
| 10 | [Hierarchical](examples/nb_10_hierarchical_timeseries.ipynb) | Tree structures and aggregation |

---

## 🤝 Contributing

Contributions are welcome! Here are some ways to contribute to **TimeDataModel**:

* Propose new data classes or extend existing ones;
* Improve documentation or add example notebooks;
* Report bugs or suggest features via [GitHub Issues](https://github.com/rebase-energy/TimeDataModel/issues).

---

## 📄 Licence

This project uses the [MIT Licence](LICENSE).
