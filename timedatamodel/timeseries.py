from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Callable, Iterator, overload

import numpy as np

from ._base import (
    _MAX_PREVIEW,
    _TimeSeriesBase,
    _build_repr_html,
    _convert_unit_values,
    _get_pint_registry,
    _import_pandas,
    _validate_timestamp_sequence,
    _xarray_labels_to_list,
)
from .coverage import CoverageBar
from .datapoint import DataPoint
from .enums import DataType, Frequency, TimeSeriesType
from .location import (
    GeoArea,
    GeoLocation,
    Location,
    _location_from_json,
    _location_to_json,
)


@dataclass(slots=True, repr=False, eq=False)
class TimeSeries(_TimeSeriesBase):
    frequency: Frequency
    timezone: str = "UTC"
    name: str | None = None
    unit: str | None = None
    description: str | None = None
    data_type: DataType | None = None
    location: Location | None = None
    timeseries_type: TimeSeriesType = TimeSeriesType.FLAT
    attributes: dict[str, str] = field(default_factory=dict)
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

    def _coverage_masks(self) -> list[tuple[str, list[bool]]]:
        return [(self.name or "value", [v is not None for v in self._values])]

    def coverage_bar(self) -> CoverageBar:
        """Return a displayable coverage bar."""
        return CoverageBar(self._coverage_masks(), self.begin, self.end)

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
            index_names=self._index_names,
        )

    def convert_unit(self, target_unit: str) -> TimeSeries:
        """Return a new TimeSeries with values converted to *target_unit*."""
        if self.unit is None:
            raise ValueError("cannot convert units: source unit is None")
        arr = _convert_unit_values(self._to_float_array(), self.unit, target_unit)
        kwargs = self._meta_kwargs()
        kwargs["unit"] = target_unit
        return TimeSeries(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=self._from_float_array(arr),
            **kwargs,
        )

    # ---- binary helpers --------------------------------------------------

    def _validate_alignment(self, other: TimeSeries) -> None:
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

    def _convert_other_values(self, other: TimeSeries) -> np.ndarray:
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

    def _apply_binary(
        self, other: TimeSeries, func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> TimeSeries:
        """Element-wise binary op between two aligned TimeSeries."""
        self._validate_alignment(other)
        a = self._to_float_array()
        b = self._convert_other_values(other)
        result = func(a, b)
        kwargs = self._meta_kwargs()
        kwargs["name"] = None
        return TimeSeries(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=self._from_float_array(result),
            **kwargs,
        )

    def _apply_comparison(self, other, op) -> TimeSeries:
        """Element-wise comparison returning TimeSeries of 1.0/0.0/NaN."""
        if isinstance(other, TimeSeries):
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
        return TimeSeries(
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
        if not isinstance(other, TimeSeries):
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
            or self._timestamps != other._timestamps
        ):
            return False
        a = self._to_float_array()
        b = other._to_float_array()
        return bool(np.array_equal(a, b, equal_nan=True))

    def __eq__(self, other):
        if isinstance(other, TimeSeries):
            return self._apply_comparison(other, np.equal)
        if isinstance(other, (int, float)):
            return self._apply_comparison(other, np.equal)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, TimeSeries):
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

    def head(self, n: int = 5) -> TimeSeries:
        """Return a new TimeSeries with the first *n* points."""
        return TimeSeries(
            self.frequency,
            timezone=self.timezone,
            timestamps=self._timestamps[:n],
            values=self._values[:n],
            **self._meta_kwargs(),
        )

    def tail(self, n: int = 5) -> TimeSeries:
        """Return a new TimeSeries with the last *n* points."""
        if n == 0:
            return TimeSeries(
                self.frequency, timezone=self.timezone, timestamps=[], values=[], **self._meta_kwargs()
            )
        return TimeSeries(
            self.frequency,
            timezone=self.timezone,
            timestamps=self._timestamps[-n:],
            values=self._values[-n:],
            **self._meta_kwargs(),
        )

    def copy(self) -> TimeSeries:
        """Return a shallow copy (timestamps and values lists are new)."""
        return TimeSeries(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=list(self._values),
            **self._meta_kwargs(),
        )

    # ---- scalar arithmetic -----------------------------------------------

    def _to_float_array(self) -> np.ndarray:
        """Convert _values to float64 ndarray — None → NaN."""
        return np.array(
            [v if v is not None else np.nan for v in self._values],
            dtype=np.float64,
        )

    def _apply_scalar(self, func) -> TimeSeries:
        arr = self._to_float_array()
        return TimeSeries(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=self._from_float_array(func(arr)),
            **self._meta_kwargs(),
        )

    def __add__(self, other) -> TimeSeries:
        if isinstance(other, TimeSeries):
            return self._apply_binary(other, lambda a, b: a + b)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: v + other)
        return NotImplemented

    def __radd__(self, other) -> TimeSeries:
        if isinstance(other, TimeSeries):
            return self._apply_binary(other, lambda a, b: b + a)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: other + v)
        return NotImplemented

    def __sub__(self, other) -> TimeSeries:
        if isinstance(other, TimeSeries):
            return self._apply_binary(other, lambda a, b: a - b)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: v - other)
        return NotImplemented

    def __rsub__(self, other) -> TimeSeries:
        if isinstance(other, TimeSeries):
            return self._apply_binary(other, lambda a, b: b - a)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: other - v)
        return NotImplemented

    def __mul__(self, other) -> TimeSeries:
        if isinstance(other, TimeSeries):
            return self._apply_binary(other, lambda a, b: a * b)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: v * other)
        return NotImplemented

    def __rmul__(self, other) -> TimeSeries:
        if isinstance(other, TimeSeries):
            return self._apply_binary(other, lambda a, b: b * a)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: other * v)
        return NotImplemented

    def __truediv__(self, other) -> TimeSeries:
        if isinstance(other, TimeSeries):
            return self._apply_binary(other, lambda a, b: a / b)
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: v / other)
        return NotImplemented

    def __rtruediv__(self, other) -> TimeSeries:
        if isinstance(other, (int, float)):
            return self._apply_scalar(lambda v: other / v)
        return NotImplemented

    def __neg__(self) -> TimeSeries:
        return self._apply_scalar(lambda v: -v)

    def __abs__(self) -> TimeSeries:
        return self._apply_scalar(abs)

    def __round__(self, n: int = 0) -> TimeSeries:
        arr = self._to_float_array()
        return TimeSeries(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=self._from_float_array(np.round(arr, n)),
            **self._meta_kwargs(),
        )

    # ---- repr hooks -------------------------------------------------------

    def _repr_meta_lines(self) -> list[str]:
        label_w = 18
        lines: list[str] = []
        lines.append(f"{'Columns:':<{label_w}}{self.name or 'unnamed'}")
        lines.append(f"{'Shape:':<{label_w}}({len(self._timestamps)},)")
        lines.append(f"{'Frequency:':<{label_w}}{self.frequency}")
        lines.append(f"{'Timezone:':<{label_w}}{self.timezone}")
        if self.unit:
            lines.append(f"{'Unit:':<{label_w}}{self.unit}")
        if self.data_type:
            lines.append(f"{'Data type:':<{label_w}}{self.data_type}")
        if self.location:
            lines.append(f"{'Location:':<{label_w}}{self._fmt_location(self.location)}")
        if self.timeseries_type and self.timeseries_type != "FLAT":
            lines.append(f"{'Timeseries type:':<{label_w}}{self.timeseries_type}")
        return lines

    def _repr_data_rows(self, indices: list[int]) -> list[list[str]]:
        return [
            [str(self._timestamps[i]), self._fmt_value(self._values[i])]
            for i in indices
        ]

    def _repr_html_(self) -> str:
        disp_name = escape(self.name or "unnamed")
        n = len(self._timestamps)

        meta_rows: list[tuple[str, str]] = [
            ("Columns", disp_name),
            ("Shape", f"({n:,},)"),
            ("Frequency", escape(str(self.frequency))),
            ("Timezone", escape(self.timezone)),
        ]
        if self.unit:
            meta_rows.append(("Unit", escape(self.unit)))
        if self.data_type:
            meta_rows.append(("Data type", escape(str(self.data_type))))
        if self.location:
            meta_rows.append(("Location", escape(self._fmt_location(self.location))))
        if self.description:
            meta_rows.append(("Description", escape(self.description)))

        def _html_row(i: int) -> str:
            ts = self._timestamps[i]
            if isinstance(ts, tuple):
                ts_cells = "".join(
                    f"<td>{escape(str(t))}</td>" for t in ts
                )
            else:
                ts_cells = f"<td>{escape(str(ts))}</td>"
            val_cells = (
                f"<td>{escape(self._fmt_value(self._values[i]))}</td>"
            )
            return f"<tr>{ts_cells}{val_cells}</tr>"

        return _build_repr_html(
            class_name="TimeSeries",
            meta_rows=meta_rows,
            index_names=self.index_names,
            column_names=self.column_names,
            n_rows=n,
            html_row_fn=_html_row,
        )

    # ---- numpy / pandas / polars -----------------------------------------

    @property
    def arr(self) -> np.ndarray:
        """Shorthand for ``to_numpy()``."""
        return self.to_numpy()

    @property
    def df(self) -> "pd.DataFrame":
        """Shorthand for ``to_pandas_dataframe()``."""
        return self.to_pandas_dataframe()

    def to_numpy(self) -> np.ndarray:
        """Return values as a 1D numpy float64 array (None -> nan)."""
        return self._to_float_array()

    def to_pandas_dataframe(self) -> "pd.DataFrame":
        """Return a pandas DataFrame with DatetimeIndex or MultiIndex."""
        pd = _import_pandas()
        arr = self.to_numpy()
        if self.is_multi_index:
            idx_names = list(self.index_names)
            index = pd.MultiIndex.from_tuples(
                self._timestamps, names=idx_names
            )
        else:
            index = pd.DatetimeIndex(
                self._timestamps, name=self.index_names[0]
            )
        col_name = self.name or "value"
        return pd.DataFrame({col_name: arr}, index=index)

    def to_pd_df(self) -> "pd.DataFrame":
        """Alias for to_pandas_dataframe()."""
        return self.to_pandas_dataframe()

    def to_pl_df(self):
        """Alias for to_polars_dataframe()."""
        return self.to_polars_dataframe()

    def to_polars_dataframe(self):
        """Return a polars DataFrame with timestamp and value columns."""
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError(
                "polars is required for to_polars_dataframe(). "
                "Install it with: pip install timedatamodel[polars]"
            ) from e

        data: dict = {}
        if self.is_multi_index:
            for i, iname in enumerate(self.index_names):
                data[iname] = [t[i] for t in self._timestamps]
        else:
            data[self.index_names[0]] = self._timestamps

        col_name = self.name or "value"
        data[col_name] = self._to_float_array()
        return pl.DataFrame(data)

    def to_xarray(self) -> "xr.DataArray":
        """Convert to a 1D xarray DataArray with timestamp coordinate."""
        import xarray as xr

        pd = _import_pandas()
        arr = self._to_float_array()

        attrs: dict[str, str] = {
            "frequency": str(self.frequency),
            "timezone": self.timezone,
        }
        if self.unit is not None:
            attrs["unit"] = self.unit
        if self.description is not None:
            attrs["description"] = self.description
        if self.data_type is not None:
            attrs["data_type"] = str(self.data_type)
        if self.timeseries_type != TimeSeriesType.FLAT:
            attrs["timeseries_type"] = str(self.timeseries_type)
        if self.attributes:
            attrs["attributes"] = json.dumps(self.attributes)

        if self.is_multi_index:
            # xarray requires dim name to differ from MultiIndex level names
            dim_name = "multi_index"
            coord = pd.MultiIndex.from_tuples(
                self._timestamps, names=list(self.index_names)
            )
            attrs["index_names"] = json.dumps(list(self.index_names))
        else:
            dim_name = self.index_names[0]
            coord = list(self._timestamps)
            if self._index_names is not None:
                attrs["index_names"] = json.dumps(self._index_names)

        return xr.DataArray(
            arr, coords={dim_name: coord}, dims=[dim_name],
            name=self.name, attrs=attrs,
        )

    @classmethod
    def from_xarray(
        cls,
        da: "xr.DataArray",
        frequency: Frequency | None = None,
        *,
        timezone: str | None = None,
        name: str | None = None,
        unit: str | None = None,
        description: str | None = None,
        data_type: DataType | None = None,
        timeseries_type: TimeSeriesType | None = None,
        attributes: dict[str, str] | None = None,
    ) -> "TimeSeries":
        """Construct a ``TimeSeries`` from a 1D ``xr.DataArray``."""
        if da.ndim != 1:
            raise ValueError(f"expected 1D DataArray, got {da.ndim}D")

        pd = _import_pandas()

        dim_name = da.dims[0]
        coord = da.coords[dim_name]

        # Detect multi-index
        if isinstance(coord.to_index(), pd.MultiIndex):
            mi = coord.to_index()
            timestamps = [
                tuple(
                    pd.Timestamp(v).to_pydatetime() if isinstance(v, (np.datetime64, pd.Timestamp)) else v
                    for v in tup
                )
                for tup in mi
            ]
            index_names = list(mi.names)
        else:
            raw = coord.values
            timestamps = []
            for v in raw:
                if isinstance(v, (np.datetime64, pd.Timestamp)):
                    timestamps.append(pd.Timestamp(v).to_pydatetime())
                else:
                    timestamps.append(v)
            index_names = da.attrs.get("index_names")
            if isinstance(index_names, str):
                index_names = json.loads(index_names)

        arr = da.values.astype(np.float64)
        values = cls._from_float_array(arr)

        freq = frequency if frequency is not None else Frequency(da.attrs.get("frequency", str(Frequency.NONE)))
        tz = timezone if timezone is not None else da.attrs.get("timezone", "UTC")
        nm = name if name is not None else da.name
        un = unit if unit is not None else da.attrs.get("unit")
        desc = description if description is not None else da.attrs.get("description")
        dt_ = data_type if data_type is not None else (
            DataType(da.attrs["data_type"]) if "data_type" in da.attrs else None
        )
        tst = timeseries_type if timeseries_type is not None else (
            TimeSeriesType(da.attrs["timeseries_type"]) if "timeseries_type" in da.attrs else TimeSeriesType.FLAT
        )
        attrs = attributes if attributes is not None else (
            json.loads(da.attrs["attributes"]) if "attributes" in da.attrs else {}
        )

        return cls(
            freq,
            timezone=tz,
            timestamps=timestamps,
            values=values,
            name=nm,
            unit=un,
            description=desc,
            data_type=dt_,
            timeseries_type=tst,
            attributes=attrs,
            index_names=index_names,
        )

    @classmethod
    def from_pandas(
        cls,
        df: "pd.DataFrame",
        frequency: Frequency | None = None,
        value_column: str | None = None,
        *,
        timezone: str = "UTC",
        name: str | None = None,
        unit: str | None = None,
        description: str | None = None,
        data_type: DataType | None = None,
        location: Location | None = None,
        timeseries_type: TimeSeriesType = TimeSeriesType.FLAT,
        attributes: dict[str, str] | None = None,
    ) -> TimeSeries:
        """Create a TimeSeries from a pandas DataFrame.

        Supports DatetimeIndex (single-index) and MultiIndex (multi-index).
        If *frequency* is ``None``, frequency and timezone are auto-inferred.
        """
        pd = _import_pandas()
        if isinstance(df.index, pd.MultiIndex):
            timestamps = [
                tuple(
                    lvl.to_pydatetime()
                    if hasattr(lvl, "to_pydatetime")
                    else lvl
                    for lvl in row
                )
                for row in df.index
            ]
            index_names = list(df.index.names)
        elif isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index.to_pydatetime().tolist()
            index_names = None
        else:
            raise ValueError(
                "DataFrame must have a DatetimeIndex or MultiIndex"
            )

        col = value_column or df.columns[0]
        arr = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
        values = cls._from_float_array(arr)

        if name is None:
            name = str(col)

        if frequency is None:
            if isinstance(df.index, pd.DatetimeIndex):
                frequency, timezone = cls._infer_freq_tz(
                    df, Frequency.NONE, timezone
                )
            else:
                frequency = Frequency.NONE

        return cls(
            frequency,
            timezone=timezone,
            timestamps=timestamps,
            values=values,
            name=name,
            unit=unit,
            description=description,
            data_type=data_type,
            location=location,
            timeseries_type=timeseries_type,
            attributes=attributes,
            index_names=index_names,
        )

    @classmethod
    def from_polars(
        cls,
        df,
        frequency: Frequency,
        timestamp_column: str = "timestamp",
        value_column: str | None = None,
        *,
        timezone: str = "UTC",
        name: str | None = None,
        unit: str | None = None,
        description: str | None = None,
        data_type: DataType | None = None,
        location: Location | None = None,
        timeseries_type: TimeSeriesType = TimeSeriesType.FLAT,
        attributes: dict[str, str] | None = None,
    ) -> TimeSeries:
        """Create a TimeSeries from a polars DataFrame."""
        val_col = value_column or [
            c for c in df.columns if c != timestamp_column
        ][0]
        timestamps = df[timestamp_column].to_list()
        arr = df[val_col].to_numpy(allow_copy=True)
        values = cls._from_float_array(arr)
        if name is None:
            name = val_col
        return cls(
            frequency,
            timezone=timezone,
            timestamps=timestamps,
            values=values,
            name=name,
            unit=unit,
            description=description,
            data_type=data_type,
            location=location,
            timeseries_type=timeseries_type,
            attributes=attributes,
        )

    def update_from_pandas(
        self,
        df: "pd.DataFrame",
        value_column: str | None = None,
        inplace: bool = False,
    ) -> TimeSeries | None:
        """Update a TimeSeries from a pandas DataFrame.

        By default returns a **new** TimeSeries.  With ``inplace=True``
        the current instance is mutated and ``None`` is returned.
        """
        pd = _import_pandas()
        if isinstance(df.index, pd.MultiIndex):
            timestamps = [
                tuple(
                    lvl.to_pydatetime()
                    if hasattr(lvl, "to_pydatetime")
                    else lvl
                    for lvl in row
                )
                for row in df.index
            ]
            index_names = list(df.index.names)
        elif isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index.to_pydatetime().tolist()
            index_names = None
        else:
            raise ValueError(
                "DataFrame must have a DatetimeIndex or MultiIndex"
            )

        col = value_column or df.columns[0]
        arr = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
        values = self._from_float_array(arr)

        if isinstance(df.index, pd.DatetimeIndex):
            new_freq, new_tz = self._infer_freq_tz(
                df, self.frequency, self.timezone
            )
        else:
            new_freq, new_tz = self.frequency, self.timezone
        original_col = self.name or "value"
        new_name = str(col) if str(col) != original_col else self.name

        if inplace:
            self._timestamps = timestamps
            self._values = values
            self._index_names = index_names
            self.frequency = new_freq
            self.timezone = new_tz
            self.name = new_name
            return None
        return TimeSeries(
            new_freq,
            timezone=new_tz,
            timestamps=timestamps,
            values=values,
            name=new_name,
            unit=self.unit,
            description=self.description,
            data_type=self.data_type,
            location=self.location,
            timeseries_type=self.timeseries_type,
            attributes=self.attributes,
            index_names=index_names,
        )

    def update_df(
        self,
        df: "pd.DataFrame",
        value_column: str | None = None,
    ) -> TimeSeries:
        """Shorthand for ``update_from_pandas(df)`` — always returns a new TimeSeries."""
        return self.update_from_pandas(df, value_column)

    def update_arr(self, arr: np.ndarray) -> TimeSeries:
        """Create a new TimeSeries with *arr* as values, keeping timestamps and metadata."""
        result = np.asarray(arr, dtype=np.float64)
        if result.ndim != 1:
            raise ValueError(
                f"update_arr: expected 1D array, got {result.ndim}D"
            )
        if result.shape[0] != len(self._timestamps):
            raise ValueError(
                f"update_arr: array length ({result.shape[0]}) must match "
                f"series length ({len(self._timestamps)})"
            )
        return TimeSeries(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=self._from_float_array(result),
            **self._meta_kwargs(),
        )

    # ---- serialization I/O -----------------------------------------------

    def to_json(self) -> str:
        """Serialize to a JSON string (timestamps as ISO-8601 strings)."""
        if self.is_multi_index:
            ts_json = [
                [dt.isoformat() for dt in tup] for tup in self._timestamps
            ]
        else:
            ts_json = [t.isoformat() for t in self._timestamps]

        val_json = (
            self._values
            if isinstance(self._values, list)
            else self._values.tolist()
        )

        payload: dict = {
            "timestamps": ts_json,
            "values": val_json,
            "frequency": str(self.frequency),
            "timezone": self.timezone,
        }
        if self._index_names is not None:
            payload["index_names"] = self._index_names
        if self.name is not None:
            payload["name"] = self.name
        if self.unit is not None:
            payload["unit"] = self.unit
        if self.description is not None:
            payload["description"] = self.description
        if self.data_type is not None:
            payload["data_type"] = str(self.data_type)
        if self.timeseries_type != TimeSeriesType.FLAT:
            payload["timeseries_type"] = str(self.timeseries_type)
        if self.attributes:
            payload["attributes"] = self.attributes
        if self.location is not None:
            payload["location"] = _location_to_json(self.location)
        return json.dumps(payload)

    @classmethod
    def from_json(
        cls,
        s: str,
        frequency: Frequency | None = None,
        *,
        timezone: str | None = None,
        name: str | None = None,
        unit: str | None = None,
        description: str | None = None,
        data_type: DataType | None = None,
        location: Location | None = None,
        timeseries_type: TimeSeriesType | None = None,
        attributes: dict[str, str] | None = None,
    ) -> TimeSeries:
        """Reconstruct a TimeSeries from a JSON string produced by to_json()."""
        data = json.loads(s)
        raw_ts = data["timestamps"]

        if raw_ts and isinstance(raw_ts[0], list):
            timestamps = [
                tuple(datetime.fromisoformat(dt) for dt in row)
                for row in raw_ts
            ]
        else:
            timestamps = [datetime.fromisoformat(t) for t in raw_ts]

        values: list[float | None] = data["values"]
        index_names = data.get("index_names")

        # Resolve frequency: explicit kwarg wins, then JSON, then error
        if frequency is not None:
            freq = frequency
        elif "frequency" in data:
            freq = Frequency(data["frequency"])
        else:
            raise ValueError("frequency must be provided either in JSON or as argument")

        # Resolve other metadata: explicit kwarg wins, then JSON value
        tz = timezone if timezone is not None else data.get("timezone", "UTC")
        nm = name if name is not None else data.get("name")
        un = unit if unit is not None else data.get("unit")
        desc = description if description is not None else data.get("description")
        dt_ = data_type if data_type is not None else (
            DataType(data["data_type"]) if "data_type" in data else None
        )
        tst = timeseries_type if timeseries_type is not None else (
            TimeSeriesType(data["timeseries_type"]) if "timeseries_type" in data else TimeSeriesType.FLAT
        )
        attrs = attributes if attributes is not None else data.get("attributes")
        loc = location if location is not None else _location_from_json(
            data.get("location")
        )

        return cls(
            freq,
            timezone=tz,
            timestamps=timestamps,
            values=values,
            name=nm,
            unit=un,
            description=desc,
            data_type=dt_,
            location=loc,
            timeseries_type=tst,
            attributes=attrs,
            index_names=index_names,
        )

    def to_csv(self, path: str | Path) -> None:
        """Write timestamps and values to a CSV file."""
        idx_names = list(self.index_names)
        col_names = list(self.column_names)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(idx_names + col_names)
            for i, t in enumerate(self._timestamps):
                if isinstance(t, tuple):
                    ts_cells = [dt.isoformat() for dt in t]
                else:
                    ts_cells = [t.isoformat()]
                v = self._values[i]
                val_cells = ["" if v is None else v]
                writer.writerow(ts_cells + val_cells)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        frequency: Frequency,
        *,
        timezone: str = "UTC",
        name: str | None = None,
        unit: str | None = None,
        description: str | None = None,
        data_type: DataType | None = None,
        location: Location | None = None,
        timeseries_type: TimeSeriesType = TimeSeriesType.FLAT,
        attributes: dict[str, str] | None = None,
    ) -> TimeSeries:
        """Read a TimeSeries from a CSV file produced by to_csv()."""
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

            idx_cols: list[int] = []
            val_cols: list[int] = []
            for i, hname in enumerate(header):
                if hname in ("timestamp",) or hname.endswith("_time"):
                    idx_cols.append(i)
                else:
                    val_cols.append(i)
            if not idx_cols:
                idx_cols = [0]
                val_cols = list(range(1, len(header)))

            multi_index = len(idx_cols) > 1

            timestamps: list = []
            rows: list = []
            for row in reader:
                if multi_index:
                    timestamps.append(
                        tuple(
                            datetime.fromisoformat(row[i]) for i in idx_cols
                        )
                    )
                else:
                    timestamps.append(
                        datetime.fromisoformat(row[idx_cols[0]])
                    )

                raw = row[val_cols[0]] if val_cols else ""
                rows.append(None if raw == "" else float(raw))

        index_names = (
            [header[i] for i in idx_cols] if multi_index else None
        )
        if name is None and val_cols:
            name = header[val_cols[0]]

        return cls(
            frequency,
            timezone=timezone,
            timestamps=timestamps,
            values=rows,
            name=name,
            unit=unit,
            description=description,
            data_type=data_type,
            location=location,
            timeseries_type=timeseries_type,
            attributes=attributes,
            index_names=index_names,
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

    # ---- pandas / numpy bridges ------------------------------------------

    def apply_pandas(
        self,
        func: Callable[["pd.DataFrame"], "pd.DataFrame"],
    ) -> TimeSeries:
        """Apply a pandas transformation, preserving metadata and auto-detecting frequency."""
        pd = _import_pandas()
        df = self.to_pandas_dataframe()
        result = func(df)
        new_freq, new_tz = self._infer_freq_tz(
            result, self.frequency, self.timezone
        )

        original_col = self.name or "value"
        new_col = (
            result.columns[0] if len(result.columns) > 0 else original_col
        )
        new_name = str(new_col) if str(new_col) != original_col else self.name

        if isinstance(result.index, pd.MultiIndex):
            timestamps = [
                tuple(
                    lvl.to_pydatetime()
                    if hasattr(lvl, "to_pydatetime")
                    else lvl
                    for lvl in row
                )
                for row in result.index
            ]
            index_names = list(result.index.names)
        elif isinstance(result.index, pd.DatetimeIndex):
            timestamps = result.index.to_pydatetime().tolist()
            index_names = None
        else:
            raise ValueError(
                "apply_pandas result must have a DatetimeIndex or MultiIndex"
            )

        arr = result.iloc[:, 0].to_numpy(dtype=np.float64, na_value=np.nan)
        values = self._from_float_array(arr)

        return TimeSeries(
            new_freq,
            timezone=new_tz,
            timestamps=timestamps,
            values=values,
            name=new_name,
            unit=self.unit,
            description=self.description,
            data_type=self.data_type,
            location=self.location,
            timeseries_type=self.timeseries_type,
            attributes=self.attributes,
            index_names=index_names,
        )

    def apply_numpy(
        self,
        func: Callable[[np.ndarray], np.ndarray],
    ) -> TimeSeries:
        """Apply a numpy transformation to values, keeping timestamps and frequency unchanged."""
        arr = self.to_numpy()
        result = np.asarray(func(arr), dtype=np.float64)
        if result.shape[0] != len(self._timestamps):
            raise ValueError(
                f"apply_numpy: result length ({result.shape[0]}) must match "
                f"series length ({len(self._timestamps)})"
            )
        return TimeSeries(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=self._from_float_array(result),
            **self._meta_kwargs(),
        )

    def apply_polars(self, func: Callable) -> TimeSeries:
        """Apply a polars transformation, preserving frequency/timezone/metadata."""
        df = self.to_polars_dataframe()
        result = func(df)

        ts_col = self.index_names[0]
        timestamps = result[ts_col].to_list()

        val_cols = [c for c in result.columns if c != ts_col]
        val_col = val_cols[0] if val_cols else (self.name or "value")
        arr = result[val_col].to_numpy(allow_copy=True).astype(np.float64)
        values = self._from_float_array(arr)

        return TimeSeries(
            self.frequency,
            timezone=self.timezone,
            timestamps=timestamps,
            values=values,
            **self._meta_kwargs(),
        )

    def apply_xarray(self, func: Callable) -> TimeSeries:
        """Apply an xarray transformation, reading metadata from result.attrs with self as fallback."""
        da = self.to_xarray()
        result = func(da)
        return TimeSeries.from_xarray(
            result,
            frequency=Frequency(result.attrs.get("frequency", str(self.frequency))),
            timezone=result.attrs.get("timezone", self.timezone),
            name=result.name if result.name is not None else self.name,
            unit=result.attrs.get("unit", self.unit),
            description=result.attrs.get("description", self.description),
            data_type=(
                DataType(result.attrs["data_type"])
                if "data_type" in result.attrs
                else self.data_type
            ),
            timeseries_type=(
                TimeSeriesType(result.attrs["timeseries_type"])
                if "timeseries_type" in result.attrs
                else self.timeseries_type
            ),
            attributes=(
                json.loads(result.attrs["attributes"])
                if "attributes" in result.attrs
                else self.attributes
            ),
        )

    @staticmethod
    def merge(series: list[TimeSeries]) -> "TimeSeriesTable":
        """Combine multiple univariate TimeSeries into a TimeSeriesTable.

        All series must share the same timestamps (same length and values).
        """
        from .table import TimeSeriesTable

        if not series:
            raise ValueError("Cannot merge an empty list of TimeSeries.")

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
        )
