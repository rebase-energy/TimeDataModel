# Installation

## Basic install

```bash
pip install timedatamodel
```

## Optional extras

| Extra | Command | Description |
|-------|---------|-------------|
| `pandas` | `pip install timedatamodel[pandas]` | pandas support |
| `polars` | `pip install timedatamodel[polars]` | polars support |
| `pint` | `pip install timedatamodel[pint]` | pint unit support |
| `geo` | `pip install timedatamodel[geo]` | shapely geo support |
| `all` | `pip install timedatamodel[all]` | everything |

## Development install

```bash
git clone https://github.com/time-work/TimeDataModel.git
cd TimeDataModel
pip install -e .[dev]
```
