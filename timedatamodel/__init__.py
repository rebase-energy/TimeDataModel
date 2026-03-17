from importlib.metadata import version

from ._base import get_default_df, set_default_df
from ._repr import CoverageBar, HierarchyTree, get_repr_width, set_repr_width
from ._theme import get_theme, reset_theme, set_theme
from .array import Dimension, NDTimeSeries, TimeSeriesArray
from .collection import TimeSeriesCollection
from .datapoint import DataPoint
from .enums import DataType, Frequency, TimeSeriesType
from .hierarchy import AggregationMethod, HierarchicalTimeSeries, HierarchyNode
from .location import GeoArea, GeoLocation, Location
from .table import TimeSeriesTable
from .timeseries import TimeSeriesList
from .datashape import DataShape
from .timeseries_numpy import TimeSeriesNumpy

try:
    from .timeseries_polars import TimeSeriesPolars
    from .timeseriestable_polars import TimeSeriesTablePolars
except ImportError:
    pass

__version__ = version("timedatamodel")


def __getattr__(name: str):
    import warnings
    _deprecated_aliases = {
        "MultivariateTimeSeries": "TimeSeriesTable",
        "MultiTimeSeries": "TimeSeriesTable",
    }
    if name in _deprecated_aliases:
        warnings.warn(
            f"{name} is deprecated, use {_deprecated_aliases[name]} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return TimeSeriesTable
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
__all__ = [
    "AggregationMethod",
    "DataShape",
    "get_default_df",
    "get_repr_width",
    "get_theme",
    "CoverageBar",
    "DataPoint",
    "DataType",
    "Dimension",
    "Frequency",
    "GeoArea",
    "GeoLocation",
    "HierarchicalTimeSeries",
    "HierarchyNode",
    "HierarchyTree",
    "Location",
    "MultiTimeSeries",
    "MultivariateTimeSeries",
    "NDTimeSeries",
    "TimeSeriesCollection",
    "TimeSeriesArray",
    "TimeSeriesTable",
    "TimeSeriesType",
    "TimeSeriesNumpy",
    "TimeSeriesPolars",
    "TimeSeriesTablePolars",
    "TimeSeriesList",
    "reset_theme",
    "set_default_df",
    "set_repr_width",
    "set_theme",
]
