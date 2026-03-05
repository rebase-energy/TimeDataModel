from __future__ import annotations

import json
from typing import Callable

import numpy as np

from ._base import (
    _extract_timestamps_from_pandas_index,
    _import_pandas,
    _import_polars,
    _xarray_labels_to_list,
)
from .enums import DataType, Frequency, TimeSeriesType
from .location import Location


def _carry_over(attr: list, new_ncols: int, default_factory) -> list:
    """Carry over column metadata when column count changes."""
    if len(attr) == 1 or len(attr) == new_ncols:
        return list(attr)
    return [default_factory()]


class _TimeSeriesListConverterMixin:
    """Conversion methods (numpy/pandas/polars/xarray) for TimeSeriesList."""

    @property
    def arr(self) -> np.ndarray:
        """Shorthand for ``to_numpy()``."""
        return self.to_numpy()

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
        pl = _import_polars()

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
        if self.labels:
            attrs["labels"] = json.dumps(self.labels)

        if self.is_multi_index:
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
        labels: dict[str, str] | None = None,
    ):
        """Construct a ``TimeSeriesList`` from a 1D ``xr.DataArray``."""
        if da.ndim != 1:
            raise ValueError(f"expected 1D DataArray, got {da.ndim}D")

        pd = _import_pandas()

        dim_name = da.dims[0]
        coord = da.coords[dim_name]

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
        lbls = labels if labels is not None else (
            json.loads(da.attrs["labels"]) if "labels" in da.attrs else {}
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
            labels=lbls,
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
        labels: dict[str, str] | None = None,
    ):
        """Create a TimeSeriesList from a pandas DataFrame."""
        pd = _import_pandas()
        timestamps, index_names = _extract_timestamps_from_pandas_index(df)

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
            labels=labels,
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
        labels: dict[str, str] | None = None,
    ):
        """Create a TimeSeriesList from a polars DataFrame."""
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
            labels=labels,
        )

    def update_from_pandas(
        self,
        df: "pd.DataFrame",
        value_column: str | None = None,
        inplace: bool = False,
    ):
        """Update a TimeSeriesList from a pandas DataFrame."""
        from .timeseries import TimeSeriesList

        pd = _import_pandas()
        timestamps, index_names = _extract_timestamps_from_pandas_index(df)

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
        return TimeSeriesList(
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
            labels=self.labels,
            index_names=index_names,
        )

    def update_df(
        self,
        df: "pd.DataFrame",
        value_column: str | None = None,
    ):
        """Shorthand for ``update_from_pandas(df)`` — always returns a new TimeSeriesList."""
        return self.update_from_pandas(df, value_column)

    def update_arr(self, arr: np.ndarray):
        """Create a new TimeSeriesList with *arr* as values, keeping timestamps and metadata."""
        from .timeseries import TimeSeriesList

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
        return TimeSeriesList(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=self._from_float_array(result),
            **self._meta_kwargs(),
        )

    # ---- pandas / numpy bridges ------------------------------------------

    def apply_pandas(
        self,
        func: Callable[["pd.DataFrame"], "pd.DataFrame"],
    ):
        """Apply a pandas transformation, preserving metadata and auto-detecting frequency."""
        from .timeseries import TimeSeriesList

        _import_pandas()
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

        timestamps, index_names = _extract_timestamps_from_pandas_index(result)

        arr = result.iloc[:, 0].to_numpy(dtype=np.float64, na_value=np.nan)
        values = self._from_float_array(arr)

        return TimeSeriesList(
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
            labels=self.labels,
            index_names=index_names,
        )

    def apply_numpy(
        self,
        func: Callable[[np.ndarray], np.ndarray],
    ):
        """Apply a numpy transformation to values, keeping timestamps and frequency unchanged."""
        from .timeseries import TimeSeriesList

        arr = self.to_numpy()
        result = np.asarray(func(arr), dtype=np.float64)
        if result.shape[0] != len(self._timestamps):
            raise ValueError(
                f"apply_numpy: result length ({result.shape[0]}) must match "
                f"series length ({len(self._timestamps)})"
            )
        return TimeSeriesList(
            self.frequency,
            timezone=self.timezone,
            timestamps=list(self._timestamps),
            values=self._from_float_array(result),
            **self._meta_kwargs(),
        )

    def apply_polars(self, func: Callable):
        """Apply a polars transformation, preserving frequency/timezone/metadata."""
        from .timeseries import TimeSeriesList

        df = self.to_polars_dataframe()
        result = func(df)

        ts_col = self.index_names[0]
        timestamps = result[ts_col].to_list()

        val_cols = [c for c in result.columns if c != ts_col]
        val_col = val_cols[0] if val_cols else (self.name or "value")
        arr = result[val_col].to_numpy(allow_copy=True).astype(np.float64)
        values = self._from_float_array(arr)

        return TimeSeriesList(
            self.frequency,
            timezone=self.timezone,
            timestamps=timestamps,
            values=values,
            **self._meta_kwargs(),
        )

    def apply_xarray(self, func: Callable):
        """Apply an xarray transformation, reading metadata from result.attrs with self as fallback."""
        from .timeseries import TimeSeriesList

        da = self.to_xarray()
        result = func(da)
        return TimeSeriesList.from_xarray(
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
            labels=(
                json.loads(result.attrs["labels"])
                if "labels" in result.attrs
                else self.labels
            ),
        )


class _TimeSeriesTableConverterMixin:
    """Conversion methods (numpy/pandas/polars/xarray) for TimeSeriesTable."""

    @property
    def arr(self) -> np.ndarray:
        """Shorthand for ``to_numpy()``."""
        return self.to_numpy()

    def to_numpy(self) -> np.ndarray:
        """Return values as a 2D numpy float64 array."""
        return self._values.astype(np.float64, copy=True)

    def to_pandas_dataframe(self) -> "pd.DataFrame":
        """Return a pandas DataFrame with multiple value columns."""
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
        cols = list(self.column_names)
        return pd.DataFrame(arr, index=index, columns=cols)

    def to_pd_df(self) -> "pd.DataFrame":
        """Alias for to_pandas_dataframe()."""
        return self.to_pandas_dataframe()

    def to_pl_df(self):
        """Alias for to_polars_dataframe()."""
        return self.to_polars_dataframe()

    def to_polars_dataframe(self):
        """Return a polars DataFrame with timestamp and value columns."""
        pl = _import_polars()

        data: dict = {}
        if self.is_multi_index:
            for i, iname in enumerate(self.index_names):
                data[iname] = [t[i] for t in self._timestamps]
        else:
            data[self.index_names[0]] = self._timestamps

        arr = self.to_numpy()
        for j, cname in enumerate(self.column_names):
            data[cname] = arr[:, j]

        return pl.DataFrame(data)

    def to_xarray(self) -> "xr.DataArray":
        """Convert to a 2D xarray DataArray (timestamp x column)."""
        import xarray as xr

        arr = self.to_numpy()
        dim_name = self.index_names[0]
        col_names = list(self.column_names)

        coords = {
            dim_name: list(self._timestamps),
            "column": col_names,
        }

        attrs: dict[str, str] = {
            "frequency": str(self.frequency),
            "timezone": self.timezone,
        }
        if any(n is not None for n in self.names):
            attrs["names"] = json.dumps(self.names)
        if any(u is not None for u in self.units):
            attrs["units"] = json.dumps(self.units)
        if any(d is not None for d in self.descriptions):
            attrs["descriptions"] = json.dumps(self.descriptions)
        if any(d is not None for d in self.data_types):
            attrs["data_types"] = json.dumps([str(d) if d else None for d in self.data_types])
        if any(t != TimeSeriesType.FLAT for t in self.timeseries_types):
            attrs["timeseries_types"] = json.dumps([str(t) for t in self.timeseries_types])
        if any(a for a in self.attributes):
            attrs["attributes"] = json.dumps(self.attributes)
        if any(lbl for lbl in self.labels):
            attrs["labels"] = json.dumps(self.labels)
        if self._index_names is not None:
            attrs["index_names"] = json.dumps(self._index_names)

        return xr.DataArray(
            arr, coords=coords, dims=[dim_name, "column"], attrs=attrs,
        )

    @classmethod
    def from_xarray(
        cls,
        da: "xr.DataArray",
        frequency: Frequency | None = None,
        *,
        timezone: str | None = None,
        names: list[str | None] | None = None,
        units: list[str | None] | None = None,
        descriptions: list[str | None] | None = None,
        data_types: list[DataType | None] | None = None,
        locations: list[Location | None] | None = None,
        timeseries_types: list[TimeSeriesType] | None = None,
        attributes: list[dict[str, str]] | None = None,
        labels: list[dict[str, str]] | None = None,
    ):
        """Construct a ``TimeSeriesTable`` from a 2D ``xr.DataArray``."""
        if da.ndim != 2:
            raise ValueError(f"expected 2D DataArray, got {da.ndim}D")

        pd = _import_pandas()

        dim0, dim1 = da.dims
        coord0 = da.coords[dim0].values
        coord1 = da.coords[dim1].values

        def _is_datetime_coord(vals):
            if len(vals) == 0:
                return False
            return isinstance(vals[0], (np.datetime64, pd.Timestamp))

        if _is_datetime_coord(coord0):
            ts_dim = dim0
            values = da.values.astype(np.float64)
        elif _is_datetime_coord(coord1):
            ts_dim = dim1
            values = da.values.astype(np.float64).T
        else:
            ts_dim = dim0
            values = da.values.astype(np.float64)

        timestamps = _xarray_labels_to_list(da.coords[ts_dim].values)

        freq = frequency if frequency is not None else Frequency(da.attrs.get("frequency", str(Frequency.NONE)))
        tz = timezone if timezone is not None else da.attrs.get("timezone", "UTC")
        nm = names if names is not None else (
            json.loads(da.attrs["names"]) if "names" in da.attrs else None
        )
        un = units if units is not None else (
            json.loads(da.attrs["units"]) if "units" in da.attrs else None
        )
        desc = descriptions if descriptions is not None else (
            json.loads(da.attrs["descriptions"]) if "descriptions" in da.attrs else None
        )
        dt_ = data_types if data_types is not None else (
            [DataType(d) if d else None for d in json.loads(da.attrs["data_types"])]
            if "data_types" in da.attrs else None
        )
        tst = timeseries_types if timeseries_types is not None else (
            [TimeSeriesType(t) for t in json.loads(da.attrs["timeseries_types"])]
            if "timeseries_types" in da.attrs else None
        )
        attrs = attributes if attributes is not None else (
            json.loads(da.attrs["attributes"]) if "attributes" in da.attrs else None
        )
        lbls = labels if labels is not None else (
            json.loads(da.attrs["labels"]) if "labels" in da.attrs else None
        )
        index_names = (
            json.loads(da.attrs["index_names"]) if "index_names" in da.attrs else None
        )

        return cls(
            freq,
            timezone=tz,
            timestamps=timestamps,
            values=values,
            names=nm,
            units=un,
            descriptions=desc,
            data_types=dt_,
            locations=locations,
            timeseries_types=tst,
            attributes=attrs,
            labels=lbls,
            index_names=index_names,
        )

    @classmethod
    def from_pandas(
        cls,
        df: "pd.DataFrame",
        frequency: Frequency | None = None,
        *,
        timezone: str = "UTC",
        names: list[str | None] | None = None,
        units: list[str | None] | None = None,
        descriptions: list[str | None] | None = None,
        data_types: list[DataType | None] | None = None,
        locations: list[Location | None] | None = None,
        timeseries_types: list[TimeSeriesType] | None = None,
        attributes: list[dict[str, str]] | None = None,
        labels: list[dict[str, str]] | None = None,
    ):
        """Create a TimeSeriesTable from a pandas DataFrame."""
        pd = _import_pandas()
        timestamps, index_names = _extract_timestamps_from_pandas_index(df)

        cols = list(df.columns)
        values = df[cols].to_numpy(dtype=np.float64)

        if names is None:
            names = [str(c) for c in cols]

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
            names=names,
            units=units,
            descriptions=descriptions,
            data_types=data_types,
            locations=locations,
            timeseries_types=timeseries_types,
            attributes=attributes,
            labels=labels,
            index_names=index_names,
        )

    @classmethod
    def from_polars(
        cls,
        df,
        frequency: Frequency,
        timestamp_column: str = "timestamp",
        *,
        timezone: str = "UTC",
        names: list[str | None] | None = None,
        units: list[str | None] | None = None,
        descriptions: list[str | None] | None = None,
        data_types: list[DataType | None] | None = None,
        locations: list[Location | None] | None = None,
        timeseries_types: list[TimeSeriesType] | None = None,
        attributes: list[dict[str, str]] | None = None,
        labels: list[dict[str, str]] | None = None,
    ):
        """Create a TimeSeriesTable from a polars DataFrame."""
        timestamps = df[timestamp_column].to_list()
        value_cols = [c for c in df.columns if c != timestamp_column]
        if names is None:
            names = value_cols
        values = np.column_stack(
            [df[c].to_numpy(allow_copy=True) for c in value_cols]
        ).astype(np.float64)
        return cls(
            frequency,
            timezone=timezone,
            timestamps=timestamps,
            values=values,
            names=names,
            units=units,
            descriptions=descriptions,
            data_types=data_types,
            locations=locations,
            timeseries_types=timeseries_types,
            attributes=attributes,
            labels=labels,
        )

    def update_from_pandas(self, df: "pd.DataFrame"):
        """Create a new TimeSeriesTable from a DataFrame, preserving metadata."""
        from .table import TimeSeriesTable

        _import_pandas()
        timestamps, index_names = _extract_timestamps_from_pandas_index(df)

        values = df.to_numpy(dtype=np.float64)
        new_freq, new_tz = self._infer_freq_tz(
            df, self.frequency, self.timezone
        )
        new_names = [str(c) for c in df.columns]
        new_ncols = len(df.columns)

        return TimeSeriesTable(
            new_freq,
            timezone=new_tz,
            timestamps=timestamps,
            values=values,
            names=new_names,
            units=_carry_over(self.units, new_ncols, lambda: None),
            descriptions=_carry_over(self.descriptions, new_ncols, lambda: None),
            data_types=_carry_over(self.data_types, new_ncols, lambda: None),
            locations=_carry_over(self.locations, new_ncols, lambda: None),
            timeseries_types=_carry_over(
                self.timeseries_types, new_ncols, lambda: TimeSeriesType.FLAT
            ),
            attributes=_carry_over(self.attributes, new_ncols, dict),
            labels=_carry_over(self.labels, new_ncols, dict),
            index_names=index_names,
        )

    def update_df(self, df: "pd.DataFrame"):
        """Shorthand for ``update_from_pandas(df)`` — always returns a new TimeSeriesTable."""
        return self.update_from_pandas(df)

    def update_arr(self, arr: np.ndarray):
        """Create a new TimeSeriesTable with *arr* as values, keeping timestamps and metadata."""
        result = np.asarray(arr, dtype=np.float64)
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        if result.ndim != 2:
            raise ValueError(
                f"update_arr: expected 1D or 2D array, got {result.ndim}D"
            )
        if result.shape[0] != len(self._timestamps):
            raise ValueError(
                f"update_arr: array rows ({result.shape[0]}) must match "
                f"series length ({len(self._timestamps)})"
            )
        return self._clone_with(list(self._timestamps), result)

    # ---- pandas / numpy bridges ------------------------------------------

    def apply_pandas(
        self,
        func: Callable[["pd.DataFrame"], "pd.DataFrame"],
    ):
        """Apply a pandas transformation, preserving metadata and auto-detecting frequency."""
        from .table import TimeSeriesTable

        _import_pandas()
        df = self.to_pandas_dataframe()
        result = func(df)
        new_freq, new_tz = self._infer_freq_tz(
            result, self.frequency, self.timezone
        )

        timestamps, index_names = _extract_timestamps_from_pandas_index(result)

        values = result.to_numpy(dtype=np.float64)
        new_names = [str(c) for c in result.columns]
        new_ncols = len(result.columns)

        return TimeSeriesTable(
            new_freq,
            timezone=new_tz,
            timestamps=timestamps,
            values=values,
            names=new_names,
            units=_carry_over(self.units, new_ncols, lambda: None),
            descriptions=_carry_over(self.descriptions, new_ncols, lambda: None),
            data_types=_carry_over(self.data_types, new_ncols, lambda: None),
            locations=_carry_over(self.locations, new_ncols, lambda: None),
            timeseries_types=_carry_over(
                self.timeseries_types, new_ncols, lambda: TimeSeriesType.FLAT
            ),
            attributes=_carry_over(self.attributes, new_ncols, dict),
            labels=_carry_over(self.labels, new_ncols, dict),
            index_names=index_names,
        )

    def apply_numpy(
        self,
        func: Callable[[np.ndarray], np.ndarray],
    ):
        """Apply a numpy transformation to values, keeping timestamps and resolution unchanged."""
        arr = self.to_numpy()
        result = np.asarray(func(arr), dtype=np.float64)
        if result.shape[0] != len(self._timestamps):
            raise ValueError(
                f"apply_numpy: result length ({result.shape[0]}) must match "
                f"series length ({len(self._timestamps)})"
            )
        return self._clone_with(list(self._timestamps), result)

    def apply_polars(self, func: Callable):
        """Apply a polars transformation, preserving frequency/timezone/metadata."""
        from .table import TimeSeriesTable

        df = self.to_polars_dataframe()
        result = func(df)

        ts_col = self.index_names[0]
        timestamps = result[ts_col].to_list()

        val_cols = [c for c in result.columns if c != ts_col]
        values = result.select(val_cols).to_numpy().astype(np.float64)
        new_names = val_cols
        new_ncols = len(val_cols)

        return TimeSeriesTable(
            self.frequency,
            timezone=self.timezone,
            timestamps=timestamps,
            values=values,
            names=new_names,
            units=_carry_over(self.units, new_ncols, lambda: None),
            descriptions=_carry_over(self.descriptions, new_ncols, lambda: None),
            data_types=_carry_over(self.data_types, new_ncols, lambda: None),
            locations=_carry_over(self.locations, new_ncols, lambda: None),
            timeseries_types=_carry_over(
                self.timeseries_types, new_ncols, lambda: TimeSeriesType.FLAT
            ),
            attributes=_carry_over(self.attributes, new_ncols, dict),
            labels=_carry_over(self.labels, new_ncols, dict),
        )

    def apply_xarray(self, func: Callable):
        """Apply an xarray transformation, reading metadata from result.attrs with self as fallback."""
        from .table import TimeSeriesTable

        da = self.to_xarray()
        result = func(da)
        return TimeSeriesTable.from_xarray(
            result,
            frequency=Frequency(result.attrs.get("frequency", str(self.frequency))),
            timezone=result.attrs.get("timezone", self.timezone),
            names=(
                json.loads(result.attrs["names"])
                if "names" in result.attrs
                else list(self.names)
            ),
            units=(
                json.loads(result.attrs["units"])
                if "units" in result.attrs
                else list(self.units)
            ),
            descriptions=(
                json.loads(result.attrs["descriptions"])
                if "descriptions" in result.attrs
                else list(self.descriptions)
            ),
            data_types=(
                [DataType(d) if d else None for d in json.loads(result.attrs["data_types"])]
                if "data_types" in result.attrs
                else list(self.data_types)
            ),
            timeseries_types=(
                [TimeSeriesType(t) for t in json.loads(result.attrs["timeseries_types"])]
                if "timeseries_types" in result.attrs
                else list(self.timeseries_types)
            ),
            attributes=(
                json.loads(result.attrs["attributes"])
                if "attributes" in result.attrs
                else list(self.attributes)
            ),
            labels=(
                json.loads(result.attrs["labels"])
                if "labels" in result.attrs
                else list(self.labels)
            ),
        )
