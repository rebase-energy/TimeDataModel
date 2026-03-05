from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, overload

import numpy as np

from ._base import (
    _convert_unit_values,
    _DataFrameMixin,
    _TimeSeriesBase,
    _validate_timestamp_sequence,
)
from ._converters import _TimeSeriesTableConverterMixin
from ._io import _TimeSeriesTableIOMixin
from ._ops import _TimeSeriesTableOpsMixin
from ._repr import _TimeSeriesTableReprMixin
from .enums import DataType, Frequency, TimeSeriesType
from .location import (
    GeoArea,
    GeoLocation,
    Location,
)


@dataclass(slots=True, repr=False, eq=False)
class TimeSeriesTable(
    _TimeSeriesBase,
    _TimeSeriesTableReprMixin,
    _DataFrameMixin,
    _TimeSeriesTableOpsMixin,
    _TimeSeriesTableIOMixin,
    _TimeSeriesTableConverterMixin,
):
    frequency: Frequency
    timezone: str = "UTC"
    names: list[str | None] = field(default_factory=lambda: [None])
    units: list[str | None] = field(default_factory=lambda: [None])
    descriptions: list[str | None] = field(default_factory=lambda: [None])
    data_types: list[DataType | None] = field(default_factory=lambda: [None])
    locations: list[Location | None] = field(default_factory=lambda: [None])
    timeseries_types: list[TimeSeriesType] = field(
        default_factory=lambda: [TimeSeriesType.FLAT]
    )
    attributes: list[dict[str, str]] = field(default_factory=lambda: [{}])
    labels: list[dict[str, str]] = field(default_factory=lambda: [{}])
    _timestamps: list[datetime] | list[tuple[datetime, ...]] = field(
        default_factory=list, repr=False
    )
    _values: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0)), repr=False
    )
    _index_names: list[str] | None = field(default=None, repr=False)

    def __init__(
        self,
        frequency: Frequency,
        *,
        timezone: str = "UTC",
        timestamps: list[datetime] | list[tuple[datetime, ...]] | None = None,
        values: np.ndarray | list,
        names: list[str | None] | None = None,
        units: list[str | None] | None = None,
        descriptions: list[str | None] | None = None,
        data_types: list[DataType | None] | None = None,
        locations: list[Location | None] | None = None,
        timeseries_types: list[TimeSeriesType] | None = None,
        attributes: list[dict[str, str]] | None = None,
        labels: list[dict[str, str]] | None = None,
        index_names: list[str] | None = None,
    ) -> None:
        self.frequency = frequency
        self.timezone = timezone
        self._timestamps = timestamps or []
        self._values = np.asarray(values, dtype=np.float64)
        self._index_names = index_names

        if self._values.ndim == 1:
            self._values = self._values.reshape(-1, 1)
        if self._values.ndim != 2:
            raise ValueError(
                f"values must be 1D or 2D, got {self._values.ndim}D"
            )

        _validate_timestamp_sequence(self._timestamps)
        if len(self._timestamps) != self._values.shape[0]:
            raise ValueError(
                f"length mismatch: {len(self._timestamps)} timestamps "
                f"vs {self._values.shape[0]} value rows"
            )

        ncols = self._values.shape[1]

        def _validate_list(attr_name, attr_list, default_factory):
            if attr_list is None:
                return [default_factory()]
            if len(attr_list) == 1 or len(attr_list) == ncols:
                return list(attr_list)
            raise ValueError(
                f"{attr_name} must have length 1 or {ncols}, "
                f"got {len(attr_list)}"
            )

        self.names = _validate_list("names", names, lambda: None)
        self.units = _validate_list("units", units, lambda: None)
        self.descriptions = _validate_list(
            "descriptions", descriptions, lambda: None
        )
        self.data_types = _validate_list(
            "data_types", data_types, lambda: None
        )
        self.locations = _validate_list(
            "locations", locations, lambda: None
        )
        self.timeseries_types = _validate_list(
            "timeseries_types", timeseries_types, lambda: TimeSeriesType.FLAT
        )
        self.attributes = _validate_list("attributes", attributes, dict)
        self.labels = _validate_list("labels", labels, dict)

    # ---- properties ------------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def n_columns(self) -> int:
        return self._values.shape[1]

    def _get_attr(self, attr_list: list, col: int):
        """Resolve broadcast: return attr_list[col] if len > 1, else attr_list[0]."""
        if len(attr_list) == 1:
            return attr_list[0]
        return attr_list[col]

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(
            self._get_attr(self.names, i) or f"value_{i}"
            for i in range(self.n_columns)
        )

    @property
    def has_missing(self) -> bool:
        """True if any value is NaN."""
        return bool(np.isnan(self._values).any()) if self._values.size else False

    # ---- helpers for constructing new instances ---------------------------

    def _list_meta_kwargs(self) -> dict:
        return dict(
            names=list(self.names),
            units=list(self.units),
            descriptions=list(self.descriptions),
            data_types=list(self.data_types),
            locations=list(self.locations),
            timeseries_types=list(self.timeseries_types),
            attributes=list(self.attributes),
            labels=list(self.labels),
            index_names=self._index_names,
        )

    def _clone_with(
        self, timestamps, values
    ) -> TimeSeriesTable:
        return TimeSeriesTable(
            self.frequency,
            timezone=self.timezone,
            timestamps=timestamps,
            values=values,
            **self._list_meta_kwargs(),
        )

    def convert_unit(
        self, target_unit: str, column: int | str | None = None
    ) -> TimeSeriesTable:
        """Return a new table with values converted to *target_unit*.

        Parameters
        ----------
        target_unit : str
            The unit to convert to.
        column : int, str, or None
            If None, convert all columns.  Otherwise convert only the
            specified column (by index or name).
        """
        if column is not None:
            if isinstance(column, str):
                names = self.column_names
                if column not in names:
                    raise KeyError(f"Column '{column}' not found. Available: {names}")
                column = names.index(column)
            cols_to_convert = [column]
        else:
            cols_to_convert = list(range(self.n_columns))

        new_vals = self._values.copy()
        new_units = [
            self._get_attr(self.units, i) for i in range(self.n_columns)
        ]
        for col in cols_to_convert:
            src_unit = self._get_attr(self.units, col)
            if src_unit is None:
                raise ValueError(
                    f"cannot convert units: source unit is None for column {col}"
                )
            new_vals[:, col] = _convert_unit_values(
                new_vals[:, col], src_unit, target_unit
            )
            new_units[col] = target_unit

        kwargs = self._list_meta_kwargs()
        kwargs["units"] = new_units
        return TimeSeriesTable(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=new_vals,
            **kwargs,
        )

    # ---- column selection by indices -------------------------------------

    def _select_columns(self, indices: list[int]) -> TimeSeriesTable:
        """Return a new table keeping only the given column indices."""
        if not indices:
            return TimeSeriesTable(
                self.frequency,
                timezone=self.timezone,
                timestamps=list(self._timestamps),
                values=np.empty((len(self._timestamps), 0)),
                names=[None],
                index_names=self._index_names,
            )
        new_values = self._values[:, indices]

        def _pick(attr_list: list, idxs: list[int]) -> list:
            if len(attr_list) == 1:
                return list(attr_list)
            return [attr_list[i] for i in idxs]

        return TimeSeriesTable(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=new_values,
            names=_pick(self.names, indices),
            units=_pick(self.units, indices),
            descriptions=_pick(self.descriptions, indices),
            data_types=_pick(self.data_types, indices),
            locations=_pick(self.locations, indices),
            timeseries_types=_pick(self.timeseries_types, indices),
            attributes=_pick(self.attributes, indices),
            labels=_pick(self.labels, indices),
            index_names=self._index_names,
        )

    # ---- spatial filtering -----------------------------------------------

    def filter_columns_by_location(
        self, center: GeoLocation, radius_km: float
    ) -> TimeSeriesTable:
        """Keep only columns within *radius_km* of *center*."""
        keep: list[int] = []
        for i in range(self.n_columns):
            loc = self._get_attr(self.locations, i)
            if isinstance(loc, GeoLocation) and loc.distance_to(center) <= radius_km:
                keep.append(i)
        return self._select_columns(keep)

    def filter_columns_by_area(self, area: GeoArea) -> TimeSeriesTable:
        """Keep only columns inside *area*."""
        keep: list[int] = []
        for i in range(self.n_columns):
            loc = self._get_attr(self.locations, i)
            if isinstance(loc, GeoLocation) and loc.is_within(area):
                keep.append(i)
            elif isinstance(loc, GeoArea) and area.contains_area(loc):
                keep.append(i)
        return self._select_columns(keep)

    def nearest_columns(
        self, target: GeoLocation, n: int = 1
    ) -> TimeSeriesTable:
        """Keep the *n* nearest columns to *target*."""
        dists: list[tuple[float, int]] = []
        for i in range(self.n_columns):
            loc = self._get_attr(self.locations, i)
            if isinstance(loc, GeoLocation):
                dists.append((loc.distance_to(target), i))
            elif isinstance(loc, GeoArea):
                dists.append((loc.centroid.distance_to(target), i))
        dists.sort(key=lambda x: x[0])
        keep = [idx for _, idx in dists[:n]]
        return self._select_columns(keep)

    # ---- column extraction ------------------------------------------------

    def select_column(self, col: int | str) -> "TimeSeriesList":
        """Extract a single column as a univariate TimeSeriesList."""
        from .timeseries import TimeSeriesList

        if isinstance(col, str):
            names = self.column_names
            if col not in names:
                raise KeyError(f"Column '{col}' not found. Available: {names}")
            col = names.index(col)

        arr = self._values[:, col]
        values = self._from_float_array(arr)

        return TimeSeriesList(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=values,
            name=self._get_attr(self.names, col),
            unit=self._get_attr(self.units, col),
            description=self._get_attr(self.descriptions, col),
            data_type=self._get_attr(self.data_types, col),
            location=self._get_attr(self.locations, col),
            timeseries_type=self._get_attr(self.timeseries_types, col),
            attributes=self._get_attr(self.attributes, col),
            labels=self._get_attr(self.labels, col),
            index_names=self._index_names,
        )

    def to_univariate_list(self) -> list["TimeSeriesList"]:
        """Convert to a list of univariate TimeSeriesList, one per column."""
        return [self.select_column(i) for i in range(self.n_columns)]

    # ---- sequence protocol -----------------------------------------------

    @overload
    def __getitem__(self, index: int) -> tuple: ...
    @overload
    def __getitem__(self, index: slice) -> list[tuple]: ...

    def __getitem__(self, index: int | slice) -> tuple | list[tuple]:
        if isinstance(index, slice):
            idxs = range(len(self._timestamps))[index]
            return [
                (self._timestamps[i], self._values[i].tolist()) for i in idxs
            ]
        return (self._timestamps[index], self._values[index].tolist())

    def __iter__(self) -> Iterator[tuple]:
        return (
            (t, self._values[i].tolist())
            for i, t in enumerate(self._timestamps)
        )

    # ---- head / tail / copy ----------------------------------------------

    def head(self, n: int = 5) -> TimeSeriesTable:
        """Return a new TimeSeriesTable with the first *n* points."""
        return self._clone_with(self._timestamps[:n], self._values[:n])

    def tail(self, n: int = 5) -> TimeSeriesTable:
        """Return a new TimeSeriesTable with the last *n* points."""
        if n == 0:
            return self._clone_with([], self._values[:0])
        return self._clone_with(self._timestamps[-n:], self._values[-n:])

    def copy(self) -> TimeSeriesTable:
        """Return a shallow copy (timestamps list and values array are new)."""
        return self._clone_with(
            list(self._timestamps), self._values.copy()
        )

    def __copy__(self) -> TimeSeriesTable:
        return self.copy()

    def __deepcopy__(self, memo: dict) -> TimeSeriesTable:
        import copy

        return TimeSeriesTable(
            self.frequency,
            timezone=self.timezone,
            timestamps=copy.deepcopy(self._timestamps, memo),
            values=self._values.copy(),
            names=copy.deepcopy(self.names, memo),
            units=copy.deepcopy(self.units, memo),
            descriptions=copy.deepcopy(self.descriptions, memo),
            data_types=copy.deepcopy(self.data_types, memo),
            locations=copy.deepcopy(self.locations, memo),
            timeseries_types=copy.deepcopy(self.timeseries_types, memo),
            attributes=copy.deepcopy(self.attributes, memo),
            labels=copy.deepcopy(self.labels, memo),
            index_names=copy.deepcopy(self._index_names, memo),
        )

    def validate(self) -> list[str]:
        """Return a list of validation warnings."""
        warnings = _TimeSeriesBase.validate(self)
        n_values = self._values.shape[0]
        if len(self._timestamps) != n_values:
            warnings.insert(
                0,
                f"length mismatch: {len(self._timestamps)} timestamps "
                f"vs {n_values} values",
            )
        return warnings


def __getattr__(name: str):
    import warnings
    if name == "MultivariateTimeSeries":
        warnings.warn(
            "MultivariateTimeSeries is deprecated, use TimeSeriesTable instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return TimeSeriesTable
    if name == "MultiTimeSeries":
        warnings.warn(
            "MultiTimeSeries is deprecated, use TimeSeriesTable instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return TimeSeriesTable
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
