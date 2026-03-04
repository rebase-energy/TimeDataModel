from importlib.metadata import version

from ._base import get_default_df, set_default_df
from ._repr import CoverageBar, HierarchyTree, get_repr_width, set_repr_width
from ._theme import get_theme, reset_theme, set_theme
from .enums import DataType, Frequency, TimeSeriesType
from .location import GeoArea, GeoLocation, Location
from .datapoint import DataPoint
from .timeseries import TimeSeriesList
from .table import TimeSeriesTable, MultivariateTimeSeries, MultiTimeSeries
from .collection import TimeSeriesCollection
from .array import Dimension, NDTimeSeries, TimeSeriesArray
from .hierarchy import AggregationMethod, HierarchicalTimeSeries, HierarchyNode

__version__ = version("timedatamodel")
__all__ = [
    "AggregationMethod",
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
    "TimeSeriesList",
    "reset_theme",
    "set_default_df",
    "set_repr_width",
    "set_theme",
]
