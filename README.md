<div align="center">

# TimeDataModel

**A lightweight Python data model for time series data, built on [Polars](https://pola.rs).**

<a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square"></a>
<a href="https://pypi.org/project/timedatamodel/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/timedatamodel?color=blue&style=flat-square"></a>
<a href="https://pypi.org/project/timedatamodel/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/timedatamodel?style=flat-square"></a>
<a href="https://github.com/rebase-energy/TimeDataModel"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/rebase-energy/TimeDataModel?style=social"></a>

</div>

<br/>

**TimeDataModel** is a metadata-rich, Polars-backed container for time series data. It lets you carry your data *and* its context — name, unit, frequency, timezone, location — as a single, self-describing object, while staying fully interoperable with pandas.

**⬇️ [Installation](#installation)**
&ensp;|&ensp;
**📖 [Documentation](https://github.com/rebase-energy/TimeDataModel/tree/main/docs)**
&ensp;|&ensp;
**🚀 [Examples](https://github.com/rebase-energy/TimeDataModel/tree/main/examples)**

---

## 🧱 Core Data Classes

| Class | Description |
| :---- | :---------- |
| 📈&nbsp;`TimeSeriesPolars` | Univariate time series backed by a Polars DataFrame, supporting four temporal shapes |
| 📊&nbsp;`TimeSeriesTablePolars` | Multivariate time series — multiple named columns sharing the same `valid_time` index |
| 🔷&nbsp;`DataShape` | Enum that selects which timestamp columns are present: `SIMPLE`, `VERSIONED`, `CORRECTED`, or `AUDIT` |
| ⏱️&nbsp;`Frequency` | ISO 8601 duration-based frequencies (`PT1H`, `P1D`, `P1M`, …) |
| 🏷️&nbsp;`DataType` | Hierarchical taxonomy: `ACTUAL` → `OBSERVATION`, `DERIVED`; `CALCULATED` → `FORECAST`, `SIMULATION`, … |
| 🗺️&nbsp;`GeoLocation` / `GeoArea` | Geographic point and polygon types with distance, bearing, and containment |

---

## 📐 Data Shapes

`TimeSeriesPolars` supports four **temporal shapes** to model everything from simple point-in-time
data to fully bi-temporal audit trails:

| Shape | Columns | Use case |
| :---- | :------ | :------- |
| `SIMPLE` | `valid_time`, `value` | Standard time series |
| `VERSIONED` | `knowledge_time`, `valid_time`, `value` | Bi-temporal: track *when* each value was produced |
| `CORRECTED` | `valid_time`, `change_time`, `value` | Corrections: track *when* a value was revised |
| `AUDIT` | `knowledge_time`, `change_time`, `valid_time`, `value` | Full audit trail |

---

## 🚀 Quick Start

```python
import pandas as pd
from timedatamodel import TimeSeriesPolars, TimeSeriesTablePolars, DataShape, Frequency

# --- Univariate series from a pandas DataFrame ---
df = pd.DataFrame({
    "valid_time": pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"),
    "value": [100.0 + i * 2.5 for i in range(24)],
})

ts = TimeSeriesPolars.from_pandas(
    df,
    shape=DataShape.SIMPLE,
    frequency=Frequency.PT1H,
    name="wind_power",
    unit="MW",
)

print(ts)
# TimeSeriesPolars ─────────────────────────
#   Name        wind_power
#   Shape       SIMPLE
#   Rows        24
#   Frequency   PT1H
#   Timezone    UTC
#   Unit        MW
#  ──────────────────────────────────────────
#                  wind_power
#  2024-01-01 00:00   100.0
#  2024-01-01 01:00   102.5
#  ...

# --- Arithmetic ---
ts_doubled = ts * 2
ts_offset  = ts + 50.0

# --- Unit conversion (requires pint extra) ---
ts_kw = ts.convert_unit("kW")

# --- Round-trip to pandas ---
df_out = ts.to_pandas()

# --- Multivariate table ---
table = TimeSeriesTablePolars.from_timeseries(
    [ts_wind, ts_solar],
    frequency=Frequency.PT1H,
)
```

---

## ✨ Key Features

- 🔷 **Four data shapes** — from `SIMPLE` point-in-time to `AUDIT` full bi-temporal history;
- 🏷️ **Rich metadata** — name, unit, frequency, timezone, data type, location, labels, description on every series;
- 📊 **Multivariate tables** — `TimeSeriesTablePolars` groups co-indexed series with per-column metadata;
- 🔄 **Pandas interop** — `from_pandas` / `to_pandas` with automatic UTC enforcement;
- 🗺️ **Geospatial** — attach locations, filter by radius or area, find nearest columns;
- 📏 **Units** — optional [pint](https://pint.readthedocs.io/) integration for dimensional unit conversion;
- ⚡ **Polars native** — all internal operations use the Polars compute engine;
- 🐍 **Type-safe** — full type hints with PEP 561 support.

---

## ⬇️ Installation

Install the **stable** release:
```bash
pip install timedatamodel
```

Install with **optional dependencies**:
```bash
pip install timedatamodel[pandas]    # pandas interop (from_pandas / to_pandas)
pip install timedatamodel[pint]      # unit conversion
pip install timedatamodel[geo]       # geospatial support (shapely)
pip install timedatamodel[all]       # all optional extras
```

Install in editable mode for **development**:
```bash
git clone https://github.com/rebase-energy/TimeDataModel.git
cd TimeDataModel
pip install -e .[dev]
```

---

## 📓 Examples

| # | Notebook | Topic |
| :--- | :--- | :--- |
| 11 | [TimeSeriesPolars](examples/nb_11_timeseries_polars.ipynb) | Creating, inspecting, and operating on univariate polars-backed time series |
| 05 | [TimeSeriesTablePolars](examples/nb_05_timeseries_table_polars.ipynb) | Multivariate tables with per-column metadata and spatial filtering |

---

## 🤝 Contributing

Contributions are welcome! Here are some ways to contribute to **TimeDataModel**:

* Propose new features or extend existing classes;
* Improve documentation or add example notebooks;
* Report bugs or suggest features via [GitHub Issues](https://github.com/rebase-energy/TimeDataModel/issues).

---

## 📄 Licence

This project uses the [MIT Licence](LICENSE).
