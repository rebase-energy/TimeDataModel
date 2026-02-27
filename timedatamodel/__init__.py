from importlib.metadata import version

from .enums import DataType, Frequency, TimeSeriesType
from .location import GeoArea, GeoLocation, Location
from .coverage import CoverageBar
from .datapoint import DataPoint
from .timeseries import TimeSeries
from .table import TimeSeriesTable, MultivariateTimeSeries, MultiTimeSeries
from .collection import TimeSeriesCollection

__version__ = version("timedatamodel")
__all__ = [
    "CoverageBar",
    "DataPoint",
    "DataType",
    "Frequency",
    "GeoArea",
    "GeoLocation",
    "Location",
    "MultiTimeSeries",
    "MultivariateTimeSeries",
    "TimeSeriesCollection",
    "TimeSeriesTable",
    "TimeSeriesType",
    "TimeSeries",
]
