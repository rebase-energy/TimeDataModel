from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, overload

import numpy as np

from ._base import (
    _convert_unit_values,
    _DataFrameMixin,
    _get_pint_registry,
    _TimeSeriesBase,
    _validate_timestamp_sequence,
)
from ._converters import _TimeSeriesListConverterMixin
from ._io import _TimeSeriesListIOMixin
from ._ops import _TimeSeriesListOpsMixin
from ._repr import _TimeSeriesListReprMixin
from .datapoint import DataPoint
from .enums import DataType, Frequency, TimeSeriesType
from .location import Location


@dataclass(slots=True, repr=False, eq=False)
class TimeSeriesList(
    _TimeSeriesBase,
    _TimeSeriesListReprMixin,
    _DataFrameMixin,
    _TimeSeriesListOpsMixin,
    _TimeSeriesListIOMixin,
    _TimeSeriesListConverterMixin,
):
    frequency: Frequency
    timezone: str = "UTC"
    name: str | None = None
    unit: str | None = None
    description: str | None = None
    data_type: DataType | None = None
    location: Location | None = None
    timeseries_type: TimeSeriesType = TimeSeriesType.FLAT
    attributes: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    _timestamps: list[datetime] | list[tuple[datetime, ...]] = field(
        default_factory=list, repr=False
    )
    _values: list[float | None] = field(default_factory=list, repr=False)
    _index_names: list[str] | None = field(default=None, repr=False)

    def __init__(
        self,
        frequency: Frequency,
        *,
        timezone: str = "UTC",
        timestamps: list[datetime] | list[tuple[datetime, ...]] | None = None,
        values: list[float | None] | None = None,
        data: list[DataPoint] | None = None,
        name: str | None = None,
        unit: str | None = None,
        description: str | None = None,
        data_type: DataType | None = None,
        location: Location | None = None,
        timeseries_type: TimeSeriesType = TimeSeriesType.FLAT,
        attributes: dict[str, str] | None = None,
        labels: dict[str, str] | None = None,
        index_names: list[str] | None = None,
    ) -> None:
        self.frequency = frequency
        self.timezone = timezone
        self.name = name
        self.unit = unit
        self.description = description
        self.data_type = data_type
        self.location = location
        self.timeseries_type = timeseries_type
        self.attributes = attributes if attributes is not None else {}
        self.labels = labels if labels is not None else {}
        self._index_names = index_names

        if data is not None:
            if timestamps is not None or values is not None:
                raise ValueError(
                    "cannot specify both 'data' and 'timestamps'/'values'"
                )
            self._timestamps = [dp.timestamp for dp in data]
            self._values = [dp.value for dp in data]
        else:
            self._timestamps = timestamps or []
            self._values = values if values is not None else []

        _validate_timestamp_sequence(self._timestamps)
        if len(self._timestamps) != len(self._values):
            raise ValueError(
                f"length mismatch: {len(self._timestamps)} timestamps "
                f"vs {len(self._values)} values"
            )

    # ---- properties ------------------------------------------------------

    @property
    def values(self) -> list[float | None]:
        return self._values

    @property
    def column_names(self) -> tuple[str, ...]:
        return (self.name or "value",)

    @property
    def has_missing(self) -> bool:
        """True if any value is None."""
        return any(v is None for v in self._values)

    @property
    def pint_unit(self):
        """Resolve the unit string to a pint.Unit object."""
        if self.unit is None:
            raise ValueError("unit is not set")
        try:
            import pint
        except ImportError:
            raise ImportError(
                "pint is required for pint_unit. "
                "Install it with: pip install timedatamodel[pint]"
            ) from None
        ureg = _get_pint_registry()
        try:
            return ureg.Unit(self.unit)
        except pint.errors.UndefinedUnitError as e:
            raise ValueError(f"invalid unit string: {self.unit!r}") from e

    # ---- helpers for constructing new instances with same metadata --------

    def _meta_kwargs(self) -> dict:
        return dict(
            name=self.name,
            unit=self.unit,
            description=self.description,
            data_type=self.data_type,
            location=self.location,
            timeseries_type=self.timeseries_type,
            attributes=self.attributes,
            labels=self.labels,
            index_names=self._index_names,
        )

    def convert_unit(self, target_unit: str) -> TimeSeriesList:
        """Return a new TimeSeriesList with values converted to *target_unit*."""
        if self.unit is None:
            raise ValueError("cannot convert units: source unit is None")
        arr = _convert_unit_values(self._to_float_array(), self.unit, target_unit)
        kwargs = self._meta_kwargs()
        kwargs["unit"] = target_unit
        return TimeSeriesList(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=self._from_float_array(arr),
            **kwargs,
        )

    # ---- sequence protocol -----------------------------------------------

    @overload
    def __getitem__(self, index: int) -> DataPoint: ...
    @overload
    def __getitem__(self, index: slice) -> list[DataPoint]: ...

    def __getitem__(
        self, index: int | slice
    ) -> DataPoint | list[DataPoint]:
        if isinstance(index, slice):
            return [
                DataPoint(t, v)
                for t, v in zip(self._timestamps[index], self._values[index])
            ]
        return DataPoint(self._timestamps[index], self._values[index])

    def __iter__(self) -> Iterator[DataPoint]:
        return (
            DataPoint(t, v) for t, v in zip(self._timestamps, self._values)
        )

    # ---- head / tail / copy ----------------------------------------------

    def head(self, n: int = 5) -> TimeSeriesList:
        """Return a new TimeSeriesList with the first *n* points."""
        return TimeSeriesList(
            self.frequency,
            timezone=self.timezone,
            timestamps=self._timestamps[:n],
            values=self._values[:n],
            **self._meta_kwargs(),
        )

    def tail(self, n: int = 5) -> TimeSeriesList:
        """Return a new TimeSeriesList with the last *n* points."""
        if n == 0:
            return TimeSeriesList(
                self.frequency, timezone=self.timezone, timestamps=[], values=[], **self._meta_kwargs()
            )
        return TimeSeriesList(
            self.frequency,
            timezone=self.timezone,
            timestamps=self._timestamps[-n:],
            values=self._values[-n:],
            **self._meta_kwargs(),
        )

    def copy(self) -> TimeSeriesList:
        """Return a shallow copy (timestamps and values lists are new)."""
        return TimeSeriesList(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=list(self._values),
            **self._meta_kwargs(),
        )

    def validate(self) -> list[str]:
        """Return a list of validation warnings."""
        warnings = _TimeSeriesBase.validate(self)
        n_values = len(self._values)
        if len(self._timestamps) != n_values:
            warnings.insert(
                0,
                f"length mismatch: {len(self._timestamps)} timestamps "
                f"vs {n_values} values",
            )
        return warnings

    @staticmethod
    def merge(series: list[TimeSeriesList]) -> "TimeSeriesTable":
        """Combine multiple univariate TimeSeriesList into a TimeSeriesTable.

        All series must share the same timestamps (same length and values).
        """
        from .table import TimeSeriesTable

        if not series:
            raise ValueError("Cannot merge an empty list of TimeSeriesList.")

        ref_ts = series[0]._timestamps
        for i, s in enumerate(series[1:], 1):
            if s._timestamps != ref_ts:
                raise ValueError(
                    f"Timestamps of series[{i}] do not match series[0]. "
                    "All series must share the same timestamps."
                )

        arrays = [s._to_float_array() for s in series]
        values = np.column_stack(arrays)

        return TimeSeriesTable(
            series[0].frequency,
            timezone=series[0].timezone,
            timestamps=list(ref_ts),
            values=values,
            names=[s.name for s in series],
            units=[s.unit for s in series],
            descriptions=[s.description for s in series],
            data_types=[s.data_type for s in series],
            locations=[s.location for s in series],
            timeseries_types=[s.timeseries_type for s in series],
            attributes=[s.attributes or {} for s in series],
            labels=[s.labels or {} for s in series],
        )
