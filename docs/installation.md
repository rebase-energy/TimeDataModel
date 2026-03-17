# Installation

## Basic install

Polars is the only required dependency:

```bash
pip install timedatamodel
```

## Optional extras

| Extra | Command | Description |
|-------|---------|-------------|
| `pandas` | `pip install timedatamodel[pandas]` | `from_pandas` / `to_pandas` support |
| `pint` | `pip install timedatamodel[pint]` | Physical unit conversion |
| `geo` | `pip install timedatamodel[geo]` | Geospatial support (shapely) |
| `all` | `pip install timedatamodel[all]` | All optional extras |

## Development install

```bash
git clone https://github.com/rebase-energy/TimeDataModel.git
cd TimeDataModel
pip install -e .[dev]
```
