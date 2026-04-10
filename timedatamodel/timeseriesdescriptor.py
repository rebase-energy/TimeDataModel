"""Structure-only metadata descriptor for a time series (no data)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .enums import DataType, Frequency, TimeSeriesType
from .location import GeoLocation


@dataclass(frozen=True, slots=True)
class TimeSeriesDescriptor:
    """Describes a time series without carrying any data.

    This is a pure metadata descriptor — it declares *what* a series is
    (name, unit, data type, etc.) without holding a DataFrame.  Useful for
    registering series structure in a database or catalog before any data
    exists.

    Follows the same field names as :class:`TimeSeries` so conversion
    between the two is straightforward.
    """

    name: Optional[str] = None
    unit: str = "dimensionless"
    data_type: Optional[DataType] = None
    timeseries_type: TimeSeriesType = TimeSeriesType.FLAT
    description: Optional[str] = None
    labels: dict[str, str] = field(default_factory=dict)
    frequency: Optional[Frequency] = None
    location: Optional[GeoLocation] = None
    timezone: str = "UTC"
