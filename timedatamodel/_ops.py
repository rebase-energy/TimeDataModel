from __future__ import annotations

from typing import Callable

import numpy as np

from ._base import _convert_unit_values


class _TimeSeriesListOpsMixin:
    """Arithmetic and comparison operators for TimeSeriesList."""

    # ---- binary helpers --------------------------------------------------

    def _validate_alignment(self, other) -> None:
        """Raise ValueError if timezone, frequency, or timestamps differ."""
        if self.timezone != other.timezone:
            raise ValueError(
                f"timezone mismatch: {self.timezone!r} vs {other.timezone!r}"
            )
        if self.frequency != other.frequency:
            raise ValueError(
                f"frequency mismatch: {self.frequency} vs {other.frequency}"
            )
        if self._timestamps != other._timestamps:
            raise ValueError("timestamps do not match")

    def _convert_other_values(self, other) -> np.ndarray:
        """Return other's values as float64, converting units if needed."""
        arr = other._to_float_array()
        has_self = self.unit is not None
        has_other = other.unit is not None
        if has_self != has_other:
            raise ValueError(
                f"unit mismatch: one operand has unit={self.unit!r} "
                f"and the other has unit={other.unit!r}"
            )
        if has_self and has_other and self.unit != other.unit:
            arr = _convert_unit_values(arr, other.unit, self.unit)
        return arr

    def _apply_binary(self, other, func: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """Element-wise binary op between two aligned TimeSeriesList."""
        from .timeseries import TimeSeriesList

        self._validate_alignment(other)
        a = self._to_float_array()
        b = self._convert_other_values(other)
        result = func(a, b)
        kwargs = self._meta_kwargs()
        kwargs["name"] = None
        return TimeSeriesList(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=self._from_float_array(result),
            **kwargs,
        )

    def _apply_comparison(self, other, op):
        """Element-wise comparison returning TimeSeriesList of 1.0/0.0/NaN."""
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesList):
            self._validate_alignment(other)
            a = self._to_float_array()
            b = self._convert_other_values(other)
        elif isinstance(other, (int, float)):
            a = self._to_float_array()
            b = np.full_like(a, other)
        else:
            return NotImplemented
        nan_mask = np.isnan(a) | np.isnan(b)
        result = np.where(op(a, b), 1.0, 0.0)
        result[nan_mask] = np.nan
        return TimeSeriesList(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=self._from_float_array(result),
            name=None,
            unit=None,
        )

    # ---- equality / comparison -------------------------------------------

    def equals(self, other: object) -> bool:
        """Full structural equality (all metadata + NaN-aware values)."""
        from .timeseries import TimeSeriesList

        if not isinstance(other, TimeSeriesList):
            return NotImplemented
        if (
            self.frequency != other.frequency
            or self.timezone != other.timezone
            or self.name != other.name
            or self.unit != other.unit
            or self.description != other.description
            or self.data_type != other.data_type
            or self.timeseries_type != other.timeseries_type
            or self.attributes != other.attributes
            or self.labels != other.labels
            or self._timestamps != other._timestamps
        ):
            return False
        a = self._to_float_array()
        b = other._to_float_array()
        return bool(np.array_equal(a, b, equal_nan=True))

    def __eq__(self, other):
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesList):
            return self._apply_comparison(other, np.equal)
        if isinstance(other, (int, float)):
            return self._apply_comparison(other, np.equal)
        return NotImplemented

    def __ne__(self, other):
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesList):
            return self._apply_comparison(other, np.not_equal)
        if isinstance(other, (int, float)):
            return self._apply_comparison(other, np.not_equal)
        return NotImplemented

    def __gt__(self, other):
        return self._apply_comparison(other, np.greater)

    def __ge__(self, other):
        return self._apply_comparison(other, np.greater_equal)

    def __lt__(self, other):
        return self._apply_comparison(other, np.less)

    def __le__(self, other):
        return self._apply_comparison(other, np.less_equal)

    __hash__ = None

    # ---- scalar arithmetic -----------------------------------------------

    def _to_float_array(self) -> np.ndarray:
        """Convert _values to float64 ndarray — None → NaN."""
        return np.array(
            [v if v is not None else np.nan for v in self._values],
            dtype=np.float64,
        )

    def _apply_scalar(self, func):
        from .timeseries import TimeSeriesList

        arr = self._to_float_array()
        return TimeSeriesList(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=self._from_float_array(func(arr)),
            **self._meta_kwargs(),
        )

    def __add__(self, other):
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesList):
            return self._apply_binary(other, lambda a, b: a + b)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: v + other)
        return NotImplemented

    def __radd__(self, other):
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesList):
            return self._apply_binary(other, lambda a, b: b + a)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: other + v)
        return NotImplemented

    def __sub__(self, other):
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesList):
            return self._apply_binary(other, lambda a, b: a - b)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: v - other)
        return NotImplemented

    def __rsub__(self, other):
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesList):
            return self._apply_binary(other, lambda a, b: b - a)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: other - v)
        return NotImplemented

    def __mul__(self, other):
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesList):
            return self._apply_binary(other, lambda a, b: a * b)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: v * other)
        return NotImplemented

    def __rmul__(self, other):
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesList):
            return self._apply_binary(other, lambda a, b: b * a)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: other * v)
        return NotImplemented

    def __truediv__(self, other):
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesList):
            return self._apply_binary(other, lambda a, b: a / b)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: v / other)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: other / v)
        return NotImplemented

    def __neg__(self):
        return self._apply_scalar(lambda v: -v)

    def __abs__(self):
        return self._apply_scalar(abs)

    def __round__(self, n: int = 0):
        from .timeseries import TimeSeriesList

        arr = self._to_float_array()
        return TimeSeriesList(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=self._from_float_array(np.round(arr, n)),
            **self._meta_kwargs(),
        )


class _TimeSeriesTableOpsMixin:
    """Arithmetic and comparison operators for TimeSeriesTable."""

    # ---- equality --------------------------------------------------------

    def equals(self, other: object) -> bool:
        """Full structural equality (all metadata + NaN-aware values)."""
        from .table import TimeSeriesTable

        if not isinstance(other, TimeSeriesTable):
            return NotImplemented
        if (
            self.frequency != other.frequency
            or self.timezone != other.timezone
            or self.names != other.names
            or self.units != other.units
            or self.descriptions != other.descriptions
            or self.data_types != other.data_types
            or self.timeseries_types != other.timeseries_types
            or self.attributes != other.attributes
            or self.labels != other.labels
            or self._timestamps != other._timestamps
        ):
            return False
        return bool(np.array_equal(self._values, other._values, equal_nan=True))

    def __eq__(self, other: object) -> bool:
        from .table import TimeSeriesTable

        if not isinstance(other, TimeSeriesTable):
            return NotImplemented
        return self.equals(other)

    __hash__ = None

    # ---- arithmetic --------------------------------------------------------

    def _apply_scalar(self, func):
        arr = self._values.astype(np.float64, copy=True)
        return self._clone_with(list(self._timestamps), func(arr))

    # ---- table+table / table+series helpers ------------------------------

    def _validate_table_alignment(self, other) -> None:
        """Raise ValueError if timezone, frequency, or timestamps differ."""
        if self.timezone != other.timezone:
            raise ValueError(
                f"timezone mismatch: {self.timezone!r} vs {other.timezone!r}"
            )
        if self.frequency != other.frequency:
            raise ValueError(
                f"frequency mismatch: {self.frequency} vs {other.frequency}"
            )
        if self._timestamps != other._timestamps:
            raise ValueError("timestamps do not match")

    def _convert_other_table_values(self, other) -> np.ndarray:
        """Return *other*'s values converted to self's units per column."""
        if self.n_columns != other.n_columns:
            raise ValueError(
                f"column count mismatch: {self.n_columns} vs {other.n_columns}"
            )
        arr = other._values.astype(np.float64, copy=True)
        for col in range(self.n_columns):
            self_unit = self._get_attr(self.units, col)
            other_unit = other._get_attr(other.units, col)
            has_self = self_unit is not None
            has_other = other_unit is not None
            if has_self != has_other:
                raise ValueError(
                    f"unit mismatch in column {col}: one operand has "
                    f"unit={self_unit!r} and the other has unit={other_unit!r}"
                )
            if has_self and has_other and self_unit != other_unit:
                arr[:, col] = _convert_unit_values(arr[:, col], other_unit, self_unit)
        return arr

    def _convert_series_values(self, series) -> np.ndarray:
        """Broadcast a TimeSeriesList across all columns with per-column unit conversion."""
        arr = series._to_float_array()  # (n_rows,)
        result = np.empty_like(self._values)
        for col in range(self.n_columns):
            col_arr = arr.copy()
            self_unit = self._get_attr(self.units, col)
            has_self = self_unit is not None
            has_other = series.unit is not None
            if has_self != has_other:
                raise ValueError(
                    f"unit mismatch in column {col}: table has "
                    f"unit={self_unit!r} and series has unit={series.unit!r}"
                )
            if has_self and has_other and self_unit != series.unit:
                col_arr = _convert_unit_values(col_arr, series.unit, self_unit)
            result[:, col] = col_arr
        return result

    def _apply_table_binary(self, other, func):
        """Element-wise binary op between two aligned tables."""
        self._validate_table_alignment(other)
        a = self._values.astype(np.float64, copy=True)
        b = self._convert_other_table_values(other)
        return self._clone_with(list(self._timestamps), func(a, b))

    def _apply_series_binary(self, series, func):
        """Binary op: table (op) series, broadcasting the series."""
        if self.timezone != series.timezone:
            raise ValueError(
                f"timezone mismatch: {self.timezone!r} vs {series.timezone!r}"
            )
        if self.frequency != series.frequency:
            raise ValueError(
                f"frequency mismatch: {self.frequency} vs {series.frequency}"
            )
        if self._timestamps != series._timestamps:
            raise ValueError("timestamps do not match")
        a = self._values.astype(np.float64, copy=True)
        b = self._convert_series_values(series)
        return self._clone_with(list(self._timestamps), func(a, b))

    # ---- operators -------------------------------------------------------

    def __add__(self, other):
        from .table import TimeSeriesTable
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesTable):
            return self._apply_table_binary(other, lambda a, b: a + b)
        if isinstance(other, TimeSeriesList):
            return self._apply_series_binary(other, lambda a, b: a + b)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: v + other)
        return NotImplemented

    def __radd__(self, other):
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesList):
            return self._apply_series_binary(other, lambda a, b: b + a)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: other + v)
        return NotImplemented

    def __sub__(self, other):
        from .table import TimeSeriesTable
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesTable):
            return self._apply_table_binary(other, lambda a, b: a - b)
        if isinstance(other, TimeSeriesList):
            return self._apply_series_binary(other, lambda a, b: a - b)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: v - other)
        return NotImplemented

    def __rsub__(self, other):
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesList):
            return self._apply_series_binary(other, lambda a, b: b - a)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: other - v)
        return NotImplemented

    def __mul__(self, other):
        from .table import TimeSeriesTable
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesTable):
            return self._apply_table_binary(other, lambda a, b: a * b)
        if isinstance(other, TimeSeriesList):
            return self._apply_series_binary(other, lambda a, b: a * b)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: v * other)
        return NotImplemented

    def __rmul__(self, other):
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesList):
            return self._apply_series_binary(other, lambda a, b: b * a)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: other * v)
        return NotImplemented

    def __truediv__(self, other):
        from .table import TimeSeriesTable
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesTable):
            return self._apply_table_binary(other, lambda a, b: a / b)
        if isinstance(other, TimeSeriesList):
            return self._apply_series_binary(other, lambda a, b: a / b)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: v / other)
        return NotImplemented

    def __rtruediv__(self, other):
        from .timeseries import TimeSeriesList

        if isinstance(other, TimeSeriesList):
            return self._apply_series_binary(other, lambda a, b: b / a)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: other / v)
        return NotImplemented

    def __neg__(self):
        return self._apply_scalar(lambda v: -v)

    def __abs__(self):
        return self._apply_scalar(abs)

    def __round__(self, n: int = 0):
        arr = self._values.astype(np.float64, copy=True)
        return self._clone_with(list(self._timestamps), np.round(arr, n))
