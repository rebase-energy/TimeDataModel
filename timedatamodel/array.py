from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from itertools import product

import numpy as np

from ._base import (
    _MAX_COL_PREVIEW,
    _MAX_PREVIEW,
    _REPR_CSS,
    _TimeSeriesBase,
    _build_repr_html,
    _fmt_short_date,
    _import_pandas,
    _import_polars,
    _render_box,
    _xarray_labels_to_list,
)
from .coverage import CoverageBar
from .enums import DataType, Frequency


@dataclass(frozen=True)
class Dimension:
    name: str
    labels: list[datetime] | list[float] | list[str]


@dataclass(slots=True, repr=False, eq=False)
class TimeSeriesArray:
    frequency: Frequency
    timezone: str = "UTC"
    name: str | None = None
    unit: str | None = None
    description: str | None = None
    data_type: DataType | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    dimensions: list[Dimension] = field(default_factory=list)
    _values: np.ma.MaskedArray = field(
        default_factory=lambda: np.ma.MaskedArray(np.empty(0)), repr=False
    )

    def __init__(
        self,
        frequency: Frequency,
        *,
        timezone: str = "UTC",
        name: str | None = None,
        unit: str | None = None,
        description: str | None = None,
        data_type: DataType | None = None,
        attributes: dict[str, str] | None = None,
        dimensions: list[Dimension] | None = None,
        values: np.ndarray | np.ma.MaskedArray,
    ) -> None:
        self.frequency = frequency
        self.timezone = timezone
        self.name = name
        self.unit = unit
        self.description = description
        self.data_type = data_type
        self.attributes = attributes if attributes is not None else {}
        self.dimensions = dimensions if dimensions is not None else []

        if isinstance(values, np.ma.MaskedArray):
            self._values = values
        else:
            arr = np.asarray(values, dtype=np.float64)
            self._values = np.ma.MaskedArray(arr, mask=np.isnan(arr))

        expected = tuple(len(d.labels) for d in self.dimensions)
        if self._values.shape != expected:
            raise ValueError(
                f"values shape {self._values.shape} does not match "
                f"dimensions {expected}"
            )

    __hash__ = None

    # ---- properties ----------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self._values.shape

    @property
    def ndim(self) -> int:
        return self._values.ndim

    @property
    def dim_names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.dimensions)

    @property
    def coords(self) -> dict[str, list]:
        return {d.name: list(d.labels) for d in self.dimensions}

    @property
    def primary_time_dim(self) -> Dimension:
        # Prefer "valid_time"
        for d in self.dimensions:
            if d.name == "valid_time":
                return d
        # Fall back to first datetime-labelled dimension
        for d in self.dimensions:
            if d.labels and isinstance(d.labels[0], datetime):
                return d
        # Fall back to first dimension
        return self.dimensions[0]

    @property
    def begin(self) -> datetime | float | str | None:
        ptd = self.primary_time_dim
        return ptd.labels[0] if ptd.labels else None

    @property
    def end(self) -> datetime | float | str | None:
        ptd = self.primary_time_dim
        return ptd.labels[-1] if ptd.labels else None

    @property
    def has_missing(self) -> bool:
        return bool(self._values.mask.any()) if self._values.size else False

    # ---- internal helpers ----------------------------------------------------

    def _get_dim(self, name: str) -> Dimension:
        for d in self.dimensions:
            if d.name == name:
                return d
        raise KeyError(f"Dimension {name!r} not found. Available: {self.dim_names}")

    def _dim_index(self, name: str) -> int:
        for i, d in enumerate(self.dimensions):
            if d.name == name:
                return i
        raise KeyError(f"Dimension {name!r} not found. Available: {self.dim_names}")

    def _meta_kwargs(self) -> dict:
        return dict(
            name=self.name,
            unit=self.unit,
            description=self.description,
            data_type=self.data_type,
            attributes=self.attributes,
        )

    # ---- sel / isel ----------------------------------------------------------

    def sel(self, **kwargs) -> TimeSeriesArray | "TimeSeriesTable" | "TimeSeries":
        remaining_dims = list(self.dimensions)
        values = self._values

        any_scalar = False
        for dim_name, selector in kwargs.items():
            axis = next(
                i for i, d in enumerate(remaining_dims) if d.name == dim_name
            )
            dim = remaining_dims[axis]

            if isinstance(selector, slice):
                labels = list(dim.labels)
                start_idx = 0 if selector.start is None else labels.index(selector.start)
                stop_idx = len(labels) if selector.stop is None else labels.index(selector.stop) + 1
                slc = slice(start_idx, stop_idx)
                values = values[(slice(None),) * axis + (slc,)]
                remaining_dims[axis] = Dimension(dim.name, labels[slc])
            else:
                any_scalar = True
                try:
                    idx = list(dim.labels).index(selector)
                except ValueError:
                    raise KeyError(
                        f"Label {selector!r} not found in dimension {dim_name!r}"
                    ) from None
                values = np.take(values, idx, axis=axis)
                remaining_dims.pop(axis)

        if not any_scalar:
            return TimeSeriesArray(
                self.frequency, timezone=self.timezone,
                dimensions=remaining_dims, values=values,
                **self._meta_kwargs(),
            )
        return self._maybe_collapse(values, remaining_dims)

    def isel(self, **kwargs) -> TimeSeriesArray | "TimeSeriesTable" | "TimeSeries":
        remaining_dims = list(self.dimensions)
        values = self._values

        any_scalar = False
        for dim_name, selector in kwargs.items():
            axis = next(
                i for i, d in enumerate(remaining_dims) if d.name == dim_name
            )
            dim = remaining_dims[axis]

            if isinstance(selector, slice):
                values = values[(slice(None),) * axis + (selector,)]
                remaining_dims[axis] = Dimension(dim.name, list(dim.labels)[selector])
            else:
                any_scalar = True
                values = np.take(values, selector, axis=axis)
                remaining_dims.pop(axis)

        if not any_scalar:
            return TimeSeriesArray(
                self.frequency, timezone=self.timezone,
                dimensions=remaining_dims, values=values,
                **self._meta_kwargs(),
            )
        return self._maybe_collapse(values, remaining_dims)

    def _maybe_collapse(self, values, remaining_dims):
        ndim = values.ndim if hasattr(values, 'ndim') else 0

        if ndim == 0:
            raise ValueError("Selection collapsed all dimensions; scalar result.")

        if ndim >= 3:
            return TimeSeriesArray(
                self.frequency,
                timezone=self.timezone,
                dimensions=remaining_dims,
                values=values,
                **self._meta_kwargs(),
            )

        # Deferred imports to avoid circular imports
        from .table import TimeSeriesTable
        from .timeseries import TimeSeries

        filled = np.ma.filled(values, fill_value=np.nan)

        if ndim == 2:
            # Find which dimension has datetime labels for timestamps
            time_axis = None
            for i, d in enumerate(remaining_dims):
                if d.labels and isinstance(d.labels[0], datetime):
                    time_axis = i
                    break
            if time_axis is None:
                raise ValueError(
                    "Cannot collapse to TimeSeriesTable: no dimension "
                    "has datetime labels."
                )
            # Transpose so time is axis 0
            if time_axis != 0:
                filled = filled.T
                remaining_dims = [remaining_dims[1], remaining_dims[0]]
            timestamps = list(remaining_dims[0].labels)
            col_dim = remaining_dims[1]
            col_names = [str(lbl) for lbl in col_dim.labels]
            return TimeSeriesTable(
                self.frequency,
                timezone=self.timezone,
                timestamps=timestamps,
                values=filled,
                names=col_names,
            )

        # ndim == 1
        dim0 = remaining_dims[0]
        if not (dim0.labels and isinstance(dim0.labels[0], datetime)):
            raise ValueError(
                f"Cannot collapse to TimeSeries: dimension "
                f"{dim0.name!r} labels are not datetimes."
            )
        timestamps = list(dim0.labels)
        values_list = _TimeSeriesBase._from_float_array(filled)
        return TimeSeries(
            self.frequency,
            timezone=self.timezone,
            timestamps=timestamps,
            values=values_list,
            **self._meta_kwargs(),
        )

    # ---- conversion methods --------------------------------------------------

    def to_timeseries(self, **sel_kwargs) -> "TimeSeries":
        from .timeseries import TimeSeries

        if sel_kwargs:
            result = self.sel(**sel_kwargs)
        else:
            result = self._maybe_collapse(self._values, list(self.dimensions))
        if not isinstance(result, TimeSeries):
            raise ValueError(
                f"Selection did not collapse to TimeSeries, got {type(result).__name__}"
            )
        return result

    def to_table(self, **sel_kwargs) -> "TimeSeriesTable":
        from .table import TimeSeriesTable

        if sel_kwargs:
            result = self.sel(**sel_kwargs)
        else:
            result = self._maybe_collapse(self._values, list(self.dimensions))
        if not isinstance(result, TimeSeriesTable):
            raise ValueError(
                f"Selection did not collapse to TimeSeriesTable, got {type(result).__name__}"
            )
        return result

    def to_numpy(self) -> np.ma.MaskedArray:
        return self._values.copy()

    def to_pandas_dataframe(self) -> "pd.DataFrame":
        pd = _import_pandas()
        dim_labels = [list(d.labels) for d in self.dimensions]
        dim_names = [d.name for d in self.dimensions]
        index = pd.MultiIndex.from_product(dim_labels, names=dim_names)
        flat = np.ma.filled(self._values, fill_value=np.nan).ravel()
        col_name = self.name or "value"
        return pd.DataFrame({col_name: flat}, index=index)

    def to_xarray(self) -> "xr.DataArray":
        """Convert to an xarray DataArray.

        Each ``Dimension`` becomes a named coordinate.  Masked values are
        exported as ``NaN``.  Metadata is stored in ``DataArray.attrs`` so
        that ``from_xarray`` can round-trip it.
        """
        import json
        import xarray as xr

        data = np.ma.filled(self._values, fill_value=np.nan)
        coords = {d.name: list(d.labels) for d in self.dimensions}
        dims = list(self.dim_names)

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
        if self.attributes:
            attrs["attributes"] = json.dumps(self.attributes)

        return xr.DataArray(
            data, coords=coords, dims=dims, name=self.name, attrs=attrs,
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
        attributes: dict[str, str] | None = None,
    ) -> "TimeSeriesArray":
        """Construct a :class:`TimeSeriesArray` from an ``xr.DataArray``.

        Metadata is read from ``da.attrs`` but explicit keyword arguments
        take precedence.
        """
        import json

        dimensions = [
            Dimension(dim, _xarray_labels_to_list(da.coords[dim].values))
            for dim in da.dims
        ]

        raw = da.values.astype(np.float64)
        mask = np.isnan(raw)
        values = np.ma.MaskedArray(raw, mask=mask)

        freq = frequency if frequency is not None else Frequency(da.attrs.get("frequency", str(Frequency.NONE)))
        tz = timezone if timezone is not None else da.attrs.get("timezone", "UTC")
        nm = name if name is not None else da.name
        un = unit if unit is not None else da.attrs.get("unit")
        desc = description if description is not None else da.attrs.get("description")
        dt_ = data_type if data_type is not None else (
            DataType(da.attrs["data_type"]) if "data_type" in da.attrs else None
        )
        attrs = attributes if attributes is not None else (
            json.loads(da.attrs["attributes"]) if "attributes" in da.attrs else {}
        )

        return cls(
            freq,
            timezone=tz,
            name=nm,
            unit=un,
            description=desc,
            data_type=dt_,
            attributes=attrs,
            dimensions=dimensions,
            values=values,
        )

    # ---- apply methods -------------------------------------------------------

    def apply_xarray(self, func) -> TimeSeriesArray:
        """Apply an xarray transformation, reading metadata from result.attrs with self as fallback."""
        import json

        da = self.to_xarray()
        result = func(da)
        return TimeSeriesArray.from_xarray(
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
            attributes=(
                json.loads(result.attrs["attributes"])
                if "attributes" in result.attrs
                else self.attributes
            ),
        )

    def _non_time_dims(self):
        """Return list of dimensions that are not the primary time dimension."""
        ptd = self.primary_time_dim
        return [d for d in self.dimensions if d.name != ptd.name]

    def _array_to_pandas_df(self, non_time):
        """Convert array to a pandas DataFrame with time index and non-time columns."""
        import xarray as xr

        pd = _import_pandas()
        da = self.to_xarray()
        ptd_name = self.primary_time_dim.name

        if len(non_time) == 0:
            series = da.to_series()
            return series.to_frame(name=da.name or "value")
        elif len(non_time) == 1:
            return da.transpose(ptd_name, non_time[0].name).to_pandas()
        else:
            stack_dims = tuple(d.name for d in non_time)
            stacked = da.stack(columns=stack_dims).transpose(ptd_name, "columns")
            return stacked.to_pandas()

    def _pandas_df_to_xr(self, result_df, non_time, da):
        """Convert a pandas DataFrame back to an xarray DataArray."""
        import xarray as xr

        ptd_name = self.primary_time_dim.name

        if len(non_time) == 0:
            result_da = xr.DataArray(
                result_df.iloc[:, 0].values,
                dims=[ptd_name],
                coords={ptd_name: result_df.index},
                name=da.name,
            )
        elif len(non_time) == 1:
            dim_name = non_time[0].name
            result_da = xr.DataArray(
                result_df.values,
                dims=[ptd_name, dim_name],
                coords={
                    ptd_name: result_df.index,
                    dim_name: list(result_df.columns),
                },
                name=da.name,
            )
        else:
            stack_dims = tuple(d.name for d in non_time)
            result_da = xr.DataArray.from_series(
                result_df.stack(list(stack_dims))
            ).unstack(list(stack_dims))
            result_da = result_da.transpose(ptd_name, *stack_dims)
            result_da.name = da.name

        # Restore original dimension order
        original_dims = list(da.dims)
        if set(result_da.dims) == set(original_dims):
            result_da = result_da.transpose(*original_dims)

        for key, val in da.attrs.items():
            if key not in result_da.attrs:
                result_da.attrs[key] = val

        return result_da

    def apply_pandas(self, func) -> TimeSeriesArray:
        """Apply a pandas transformation to the array as a DataFrame.

        Gated to arrays with at most 2 non-time dimensions.
        """
        pd = _import_pandas()

        non_time = self._non_time_dims()
        if len(non_time) > 2:
            raise ValueError(
                f"apply_pandas requires at most 2 non-time dimensions, "
                f"got {len(non_time)}: {[d.name for d in non_time]}"
            )

        da = self.to_xarray()
        df = self._array_to_pandas_df(non_time)
        result_df = func(df)
        result_da = self._pandas_df_to_xr(result_df, non_time, da)

        freq_df = pd.DataFrame(
            {"_v": 0.0},
            index=(
                result_df.index
                if isinstance(result_df.index, pd.DatetimeIndex)
                else pd.DatetimeIndex(result_df.index)
            ),
        )
        new_freq, new_tz = _TimeSeriesBase._infer_freq_tz(
            freq_df, self.frequency, self.timezone,
        )

        return TimeSeriesArray.from_xarray(
            result_da,
            frequency=new_freq,
            timezone=new_tz,
            name=self.name,
            unit=self.unit,
            description=self.description,
            data_type=self.data_type,
            attributes=self.attributes,
        )

    def apply_polars(self, func) -> TimeSeriesArray:
        """Apply a polars transformation to the array as a DataFrame.

        Gated to arrays with at most 2 non-time dimensions.
        """
        non_time = self._non_time_dims()
        if len(non_time) > 2:
            raise ValueError(
                f"apply_polars requires at most 2 non-time dimensions, "
                f"got {len(non_time)}: {[d.name for d in non_time]}"
            )

        pl = _import_polars()

        pd = _import_pandas()
        da = self.to_xarray()
        ptd_name = self.primary_time_dim.name

        pdf = self._array_to_pandas_df(non_time)

        # Flatten MultiIndex columns to strings for polars compatibility
        original_columns = pdf.columns
        if hasattr(original_columns, 'to_flat_index'):
            pdf.columns = [str(c) for c in original_columns]

        # Build polars DataFrame manually to avoid pyarrow dependency
        data: dict = {ptd_name: list(pdf.index)}
        for col in pdf.columns:
            data[str(col)] = pdf[col].values
        pl_df = pl.DataFrame(data)

        result_pl = func(pl_df)

        # Reconstruct pandas DataFrame from polars result
        ts_list = result_pl[ptd_name].to_list()
        result_pdf = pd.DataFrame(
            {c: result_pl[c].to_numpy(allow_copy=True)
             for c in result_pl.columns if c != ptd_name},
            index=pd.DatetimeIndex(ts_list, name=ptd_name),
        )

        # Restore original column types
        if hasattr(original_columns, 'to_flat_index') and len(non_time) == 2:
            import ast
            non_time_names = [d.name for d in non_time]
            tuples = []
            for c in result_pdf.columns:
                try:
                    tuples.append(ast.literal_eval(c))
                except (ValueError, SyntaxError):
                    tuples.append(c)
            if all(isinstance(t, tuple) for t in tuples):
                result_pdf.columns = pd.MultiIndex.from_tuples(
                    tuples, names=non_time_names,
                )

        result_da = self._pandas_df_to_xr(result_pdf, non_time, da)

        return TimeSeriesArray.from_xarray(
            result_da,
            frequency=self.frequency,
            timezone=self.timezone,
            name=self.name,
            unit=self.unit,
            description=self.description,
            data_type=self.data_type,
            attributes=self.attributes,
        )

    # ---- class method constructors -------------------------------------------

    @classmethod
    def from_numpy(
        cls,
        dimensions: list[Dimension],
        values: np.ndarray | np.ma.MaskedArray,
        frequency: Frequency,
        *,
        timezone: str = "UTC",
        name: str | None = None,
        unit: str | None = None,
        description: str | None = None,
        data_type: DataType | None = None,
        attributes: dict[str, str] | None = None,
    ) -> TimeSeriesArray:
        return cls(
            frequency,
            timezone=timezone,
            name=name,
            unit=unit,
            description=description,
            data_type=data_type,
            attributes=attributes,
            dimensions=dimensions,
            values=values,
        )

    @classmethod
    def from_timeseries_list(
        cls,
        series: list,
        dimension: Dimension,
        *,
        frequency: Frequency | None = None,
        timezone: str | None = None,
        name: str | None = None,
        unit: str | None = None,
        description: str | None = None,
        data_type: DataType | None = None,
        attributes: dict[str, str] | None = None,
    ) -> TimeSeriesArray:
        if not series:
            raise ValueError("Cannot build array from an empty list of TimeSeries.")
        if len(dimension.labels) != len(series):
            raise ValueError(
                f"dimension has {len(dimension.labels)} labels but "
                f"{len(series)} series were provided."
            )

        # Compute sorted union of all timestamps
        all_ts: set[datetime] = set()
        for s in series:
            all_ts.update(s.timestamps)
        union_ts = sorted(all_ts)

        ts_index = {t: i for i, t in enumerate(union_ts)}

        n_series = len(series)
        n_timestamps = len(union_ts)
        data = np.full((n_series, n_timestamps), np.nan, dtype=np.float64)

        for row, s in enumerate(series):
            for t, v in zip(s.timestamps, s.values):
                col = ts_index[t]
                data[row, col] = v if v is not None else np.nan

        mask = np.isnan(data)
        values = np.ma.MaskedArray(data, mask=mask)

        ref = series[0]
        time_dim = Dimension("valid_time", union_ts)

        return cls(
            frequency=frequency or ref.frequency,
            timezone=timezone or ref.timezone,
            name=name or ref.name,
            unit=unit or ref.unit,
            description=description or ref.description,
            data_type=data_type or ref.data_type,
            attributes=attributes or ref.attributes,
            dimensions=[dimension, time_dim],
            values=values,
        )

    # ---- repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        class_name = type(self).__name__
        label_w = 18

        # Dimensions line
        dim_parts = [f"{d.name}: {len(d.labels)}" for d in self.dimensions]
        dim_str = ", ".join(dim_parts)

        meta_lines: list[str] = []
        meta_lines.append(f"{'Dimensions:':<{label_w}}{dim_str}")
        meta_lines.append(f"{'Shape:':<{label_w}}{self.shape}")
        meta_lines.append(f"{'Frequency:':<{label_w}}{self.frequency}")
        meta_lines.append(f"{'Timezone:':<{label_w}}{self.timezone}")
        if self.name:
            meta_lines.append(f"{'Name:':<{label_w}}{self.name}")
        if self.unit:
            meta_lines.append(f"{'Unit:':<{label_w}}{self.unit}")
        if self.data_type:
            meta_lines.append(f"{'Data type:':<{label_w}}{self.data_type}")

        total = self._values.size
        if total > 0:
            n_masked = int(self._values.mask.sum()) if self._values.mask.any() else 0
            if n_masked > 0:
                pct = n_masked / total * 100
                meta_lines.append(
                    f"{'Masked:':<{label_w}}{n_masked}/{total} ({pct:.1f}%)"
                )

        return _render_box(class_name, meta_lines)

    def _repr_html_(self) -> str:
        n_dims = self.ndim
        meta_rows: list[tuple[str, str]] = []

        dim_parts = [f"{d.name}: {len(d.labels)}" for d in self.dimensions]
        meta_rows.append(("Dimensions", ", ".join(dim_parts)))
        meta_rows.append(("Shape", str(self.shape)))
        meta_rows.append(("Frequency", escape(str(self.frequency))))
        meta_rows.append(("Timezone", escape(self.timezone)))
        if self.name:
            meta_rows.append(("Name", escape(self.name)))
        if self.unit:
            meta_rows.append(("Unit", escape(self.unit)))

        if n_dims >= 2:
            # Classify dimensions: datetime → rows, others → columns
            row_dims: list[Dimension] = []
            col_dims: list[Dimension] = []
            for d in self.dimensions:
                if d.labels and isinstance(d.labels[0], datetime):
                    row_dims.append(d)
                else:
                    col_dims.append(d)
            # Edge: all datetime → move last to columns
            if not col_dims:
                col_dims.append(row_dims.pop())
            # Edge: no datetime → move first to rows
            elif not row_dims:
                row_dims.append(col_dims.pop(0))

            # Map dimension names to original axis indices
            dim_to_axis = {d.name: i for i, d in enumerate(self.dimensions)}

            # Cross-product index combinations
            row_combos = list(
                product(*(range(len(d.labels)) for d in row_dims))
            )
            col_combos = list(
                product(*(range(len(d.labels)) for d in col_dims))
            )
            n_rows = len(row_combos)
            n_cols = len(col_combos)
            n_col_levels = len(col_dims)

            # Visible row indices (truncation)
            show_all_rows = n_rows <= _MAX_PREVIEW * 2 + 1
            if show_all_rows:
                vis_rows = list(range(n_rows))
            else:
                vis_rows = list(range(_MAX_PREVIEW)) + list(
                    range(n_rows - _MAX_PREVIEW, n_rows)
                )

            # Visible column indices (truncation)
            show_all_cols = n_cols <= _MAX_COL_PREVIEW * 2 + 1
            if show_all_cols:
                vis_cols = list(range(n_cols))
            else:
                vis_cols = list(range(_MAX_COL_PREVIEW)) + list(
                    range(n_cols - _MAX_COL_PREVIEW, n_cols)
                )

            def _fmt_label(label):
                if isinstance(label, datetime):
                    return _fmt_short_date(label)
                return str(label)

            # ---- build <thead> ----
            def _group_header(indices, level):
                """Group consecutive column indices by label at *level*."""
                cells: list[str] = []
                if not indices:
                    return cells
                cur_lbl = col_combos[indices[0]][level]
                cur_cnt = 1
                for k in range(1, len(indices)):
                    lbl = col_combos[indices[k]][level]
                    if lbl == cur_lbl:
                        cur_cnt += 1
                    else:
                        txt = escape(
                            _fmt_label(col_dims[level].labels[cur_lbl])
                        )
                        cells.append(
                            f'<th colspan="{cur_cnt}">{txt}</th>'
                            if cur_cnt > 1
                            else f"<th>{txt}</th>"
                        )
                        cur_lbl = lbl
                        cur_cnt = 1
                txt = escape(_fmt_label(col_dims[level].labels[cur_lbl]))
                cells.append(
                    f'<th colspan="{cur_cnt}">{txt}</th>'
                    if cur_cnt > 1
                    else f"<th>{txt}</th>"
                )
                return cells

            thead_rows: list[str] = []
            for level in range(n_col_levels):
                tr: list[str] = []
                if level == 0:
                    for rd in row_dims:
                        if n_col_levels > 1:
                            tr.append(
                                f'<th rowspan="{n_col_levels}">'
                                f"{escape(rd.name)}</th>"
                            )
                        else:
                            tr.append(f"<th>{escape(rd.name)}</th>")
                if not show_all_cols:
                    head_cells = _group_header(
                        vis_cols[:_MAX_COL_PREVIEW], level
                    )
                    tail_cells = _group_header(
                        vis_cols[_MAX_COL_PREVIEW:], level
                    )
                    tr.extend(head_cells)
                    tr.append("<th>&hellip;</th>")
                    tr.extend(tail_cells)
                else:
                    tr.extend(_group_header(vis_cols, level))
                thead_rows.append(f'<tr>{"".join(tr)}</tr>')

            # ---- build <tbody> ----
            def _data_row(ri):
                rc = row_combos[ri]
                cells: list[str] = []
                for rl, rd in enumerate(row_dims):
                    lbl = _fmt_label(rd.labels[rc[rl]])
                    cells.append(
                        f'<td class="ts-idx">{escape(lbl)}</td>'
                    )
                for k, ci in enumerate(vis_cols):
                    cc = col_combos[ci]
                    idx = [0] * len(self.dimensions)
                    for rl, rd in enumerate(row_dims):
                        idx[dim_to_axis[rd.name]] = rc[rl]
                    for cl, cd in enumerate(col_dims):
                        idx[dim_to_axis[cd.name]] = cc[cl]
                    v = float(
                        np.ma.filled(
                            self._values[tuple(idx)], fill_value=np.nan
                        )
                    )
                    cells.append(
                        f"<td>"
                        f"{escape(_TimeSeriesBase._fmt_value(v))}</td>"
                    )
                    if not show_all_cols and k == _MAX_COL_PREVIEW - 1:
                        cells.append(
                            '<td class="ts-ellipsis">&hellip;</td>'
                        )
                return f'<tr>{"".join(cells)}</tr>'

            tbody: list[str] = []
            head_vis = (
                vis_rows[:_MAX_PREVIEW] if not show_all_rows else vis_rows
            )
            tail_vis = (
                vis_rows[_MAX_PREVIEW:] if not show_all_rows else []
            )
            for ri in head_vis:
                tbody.append(_data_row(ri))

            if not show_all_rows:
                n_td = (
                    len(row_dims)
                    + len(vis_cols)
                    + (1 if not show_all_cols else 0)
                )
                ell = "".join(
                    '<td class="ts-ellipsis">&hellip;</td>'
                    for _ in range(n_td)
                )
                tbody.append(f"<tr>{ell}</tr>")
                for ri in tail_vis:
                    tbody.append(_data_row(ri))

            # ---- assemble HTML ----
            html = [_REPR_CSS, '<div class="ts-repr">']
            html.append(
                f'<div class="ts-header">'
                f"{escape(type(self).__name__)}</div>"
            )
            html.append('<div class="ts-meta"><table>')
            for label, value in meta_rows:
                html.append(
                    f"<tr><td>{escape(label)}</td><td>{value}</td></tr>"
                )
            html.append("</table></div>")
            html.append('<div class="ts-data"><table>')
            html.append("<thead>")
            for tr_str in thead_rows:
                html.append(tr_str)
            html.append("</thead>")
            html.append("<tbody>")
            for tr_str in tbody:
                html.append(tr_str)
            html.append("</tbody>")
            html.append("</table></div>")
            html.append("</div>")
            return "\n".join(html)
        elif n_dims == 1:
            dim0 = self.dimensions[0]
            n_rows = len(dim0.labels)
            col_name = self.name or "value"

            def _html_row_1d(i: int) -> str:
                ts_cell = f"<td>{escape(str(dim0.labels[i]))}</td>"
                v = float(np.ma.filled(self._values[i], fill_value=np.nan))
                val_cell = f"<td>{escape(_TimeSeriesBase._fmt_value(v))}</td>"
                return f"<tr>{ts_cell}{val_cell}</tr>"

            return _build_repr_html(
                class_name=type(self).__name__,
                meta_rows=meta_rows,
                index_names=(dim0.name,),
                column_names=(col_name,),
                n_rows=n_rows,
                html_row_fn=_html_row_1d,
            )
        else:
            return _build_repr_html(
                class_name=type(self).__name__,
                meta_rows=meta_rows,
                index_names=(),
                column_names=(),
                n_rows=0,
                html_row_fn=lambda i: "",
            )

    def coverage_bar(self) -> CoverageBar:
        ptd = self.primary_time_dim
        ptd_axis = self._dim_index(ptd.name)

        if self.ndim == 1:
            filled = np.ma.filled(self._values, fill_value=np.nan)
            mask = [not np.isnan(v) for v in filled]
            masks = [(self.name or "value", mask)]
        else:
            # Use the first non-time dimension for rows
            other_axis = 1 if ptd_axis == 0 else 0
            other_dim = self.dimensions[other_axis]

            # Collapse remaining dims by taking index 0
            vals = self._values
            dims_to_remove = []
            for i in range(self.ndim - 1, -1, -1):
                if i != ptd_axis and i != other_axis:
                    vals = np.take(vals, 0, axis=i)
                    dims_to_remove.append(i)

            masks = []
            for j, label in enumerate(other_dim.labels):
                if other_axis < ptd_axis:
                    row = np.take(vals, j, axis=0 if other_axis == 0 else other_axis)
                else:
                    row = np.take(vals, j, axis=other_axis - len(dims_to_remove))
                filled = np.ma.filled(row, fill_value=np.nan)
                mask = [not np.isnan(float(v)) for v in filled]
                masks.append((str(label), mask))

        begin = ptd.labels[0] if ptd.labels and isinstance(ptd.labels[0], datetime) else None
        end = ptd.labels[-1] if ptd.labels and isinstance(ptd.labels[-1], datetime) else None
        return CoverageBar(masks, begin, end)

    # ---- equality ------------------------------------------------------------

    def equals(self, other: object) -> bool:
        if not isinstance(other, TimeSeriesArray):
            return NotImplemented
        if (
            self.frequency != other.frequency
            or self.timezone != other.timezone
            or self.name != other.name
            or self.unit != other.unit
            or self.description != other.description
            or self.data_type != other.data_type
            or self.attributes != other.attributes
            or len(self.dimensions) != len(other.dimensions)
        ):
            return False
        for d1, d2 in zip(self.dimensions, other.dimensions):
            if d1.name != d2.name or list(d1.labels) != list(d2.labels):
                return False
        return bool(
            np.array_equal(
                np.ma.filled(self._values, fill_value=np.nan),
                np.ma.filled(other._values, fill_value=np.nan),
                equal_nan=True,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeSeriesArray):
            return NotImplemented
        return self.equals(other)


NDTimeSeries = TimeSeriesArray
