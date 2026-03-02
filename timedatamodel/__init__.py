from importlib.metadata import version

from ._base import get_default_df, set_default_df
from .enums import DataType, Frequency, TimeSeriesType
from .location import GeoArea, GeoLocation, Location
from .coverage import CoverageBar
from .datapoint import DataPoint
from .timeseries import TimeSeries
from .table import TimeSeriesTable, MultivariateTimeSeries, MultiTimeSeries
from .collection import TimeSeriesCollection
from .cube import Dimension, NDTimeSeries, TimeSeriesCube
from .hierarchy import AggregationMethod, HierarchicalTimeSeries, HierarchyNode, HierarchyTree

__version__ = version("timedatamodel")
__all__ = [
    "AggregationMethod",
    "get_default_df",
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
    "TimeSeriesCube",
    "TimeSeriesTable",
    "TimeSeriesType",
    "TimeSeries",
    "set_default_df",
]
