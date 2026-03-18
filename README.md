<div align="center">

# TimeDataModel

**A lightweight Pythonic data model for time series data, interoperable with NumPy, Pandas and Polars.**

<a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square"></a>
<a href="https://pypi.org/project/timedatamodel/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/timedatamodel?color=blue&style=flat-square"></a>
<a href="https://pypi.org/project/timedatamodel/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/timedatamodel?style=flat-square"></a>
<a href="https://github.com/rebase-energy/TimeDataModel"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/rebase-energy/TimeDataModel?style=social"></a>

</div>

<br/>

**TimeDataModel** is a metadata-rich container for time series data. It lets you carry your data *and* its context — name, unit, frequency, timezone, location — as a single, self-describing object, fully interoperable with pandas, NumPy, Polars, and PyArrow.

**⬇️ [Installation](#installation)**
&ensp;|&ensp;
**📖 [Documentation](https://timedatamodel.readthedocs.io/en/latest/)**
&ensp;|&ensp;
**🚀 [Examples](https://timedatamodel.readthedocs.io/en/latest/examples/index.html)**

---

## 🧱 Core Data Classes

| Class | Description |
| :---- | :---------- |
| 📈&nbsp;`TimeSeries` | Univariate time series supporting four temporal shapes |
| 📊&nbsp;`TimeSeriesTable` | Multivariate time series — multiple named columns sharing the same `valid_time` index |
| 🔷&nbsp;`DataShape` | Enum that selects which timestamp columns are present: `SIMPLE`, `VERSIONED`, `CORRECTED`, or `AUDIT` |
| ⏱️&nbsp;`Frequency` | ISO 8601 duration-based frequencies (`PT1H`, `P1D`, `P1M`, …) |
| 🏷️&nbsp;`DataType` | Hierarchical taxonomy: `ACTUAL` → `OBSERVATION`, `DERIVED`; `CALCULATED` → `FORECAST`, `SIMULATION`, … |
| 🗺️&nbsp;`GeoLocation` / `GeoArea` | Geographic point and polygon types with distance, bearing, and containment |

---

## 📐 Data Shapes

`TimeSeries` supports four **temporal shapes** to model everything from simple point-in-time
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
from timedatamodel import TimeSeries, TimeSeriesTable, Frequency

# --- Univariate series from a pandas DataFrame ---
df = pd.DataFrame({
    "valid_time": pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"),
    "value": [100.0 + i * 2.5 for i in range(24)],
})

ts = TimeSeries.from_pandas(
    df,
    frequency=Frequency.PT1H,
    name="wind_power",
    unit="MW",
)

print(ts)
# TimeSeries ─────────────────────────
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

# --- Unit conversion (requires pint extra) ---
ts_kw = ts.convert_unit("kW")

# --- Format conversions ---
df_pd  = ts.to_pandas()       # pd.DataFrame with datetime index
df_pl  = ts.to_polars()       # pl.DataFrame
cols   = ts.to_list()         # dict[str, list] — column-oriented
arr    = ts.to_numpy()        # dict[str, np.ndarray] — column-oriented (requires numpy)
tbl    = ts.to_pyarrow()      # pa.Table (requires pyarrow)

# --- Multivariate table — see examples/nb_02_timeseriestable.ipynb ---
```

---

## ✨ Key Features

- 🔷 **Four data shapes** — from `SIMPLE` point-in-time to `AUDIT` full bi-temporal history;
- 🏷️ **Rich metadata** — name, unit, frequency, timezone, data type, location, labels, description on every series;
- 📊 **Multivariate tables** — `TimeSeriesTable` groups co-indexed series with per-column metadata;
- 🔄 **Format conversions** — `to_pandas`, `to_polars`, `to_list`, `to_numpy`, `to_pyarrow` with lazy optional-dependency checks;
- 📊 **Coverage bar** — `coverage_bar()` renders null coverage as a binned SVG in Jupyter or Unicode blocks in terminal;
- 🗺️ **Geospatial** — attach locations, filter by radius or area, find nearest columns;
- 📏 **Units** — optional [pint](https://pint.readthedocs.io/) integration for dimensional unit conversion;
- ⚡ **Polars-powered** — backed by the Polars compute engine for high-performance in-memory processing;
- 🐍 **Type-safe** — full type hints with PEP 561 support.

---

## ⬇️ Installation

Install the **stable** release:
```bash
pip install timedatamodel
```

Install with **optional dependencies**:
```bash
pip install timedatamodel[pandas]    # pandas interop (includes pyarrow for tz-aware columns)
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
| 01 | [TimeSeries](examples/nb_01_timeseries.ipynb) | Creating, inspecting, and operating on univariate time series |
| 02 | [TimeSeriesTable](examples/nb_02_timeseriestable.ipynb) | Multivariate tables with per-column metadata and spatial filtering |

---

## 🤝 Contributing

Contributions are welcome! Here are some ways to contribute to **TimeDataModel**:

* Propose new features or extend existing classes;
* Improve documentation or add example notebooks;
* Report bugs or suggest features via [GitHub Issues](https://github.com/rebase-energy/TimeDataModel/issues).

---

## 📄 Licence

This project uses the [MIT Licence](LICENSE).
