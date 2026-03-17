from importlib.metadata import version

from ._repr import get_repr_width, set_repr_width
from ._theme import get_theme, reset_theme, set_theme
from .datapoint import DataPoint
from .datashape import DataShape
from .enums import DataType, Frequency, TimeSeriesType
from .location import GeoArea, GeoLocation, Location
from .timeseries_polars import TimeSeriesPolars
from .timeseriestable_polars import TimeSeriesTablePolars

__version__ = version("timedatamodel")

__all__ = [
    "DataPoint",
    "DataShape",
    "DataType",
    "Frequency",
    "GeoArea",
    "GeoLocation",
    "Location",
    "TimeSeriesPolars",
    "TimeSeriesTablePolars",
    "TimeSeriesType",
    "get_repr_width",
    "get_theme",
    "reset_theme",
    "set_repr_width",
    "set_theme",
]
