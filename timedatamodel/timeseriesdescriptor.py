"""Structure-only metadata descriptor for a time series (no data)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .enums import DataType, Frequency, TimeSeriesType


@dataclass(frozen=True, slots=True)
class TimeSeriesDescriptor:
    """Describes a time series without carrying any data.

    A pure metadata descriptor — it declares *what* a series is (name, unit,
    data type, …) without holding a DataFrame.  Useful for registering series
    structure in a database or catalog before any data exists.

    Follows the same field names as :class:`TimeSeries` so conversion between
    the two is straightforward.

    Identity-shaped concepts (labels, tags) and spatial context (locations)
    belong to consumer layers (e.g. ``energydatamodel``, ``energydb``), not to
    time-series metadata itself.
    """

    name: str
    unit: str = "dimensionless"
    data_type: Optional[DataType] = None
    timeseries_type: TimeSeriesType = TimeSeriesType.FLAT
    frequency: Optional[Frequency] = None
    timezone: str = "UTC"
    description: Optional[str] = None
