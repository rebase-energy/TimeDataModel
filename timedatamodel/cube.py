from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from html import escape

import numpy as np

from ._base import (
    _MAX_PREVIEW,
    _TimeSeriesBase,
    _build_repr_html,
    _import_pandas,
)
from .coverage import CoverageBar
from .enums import DataType, Frequency


@dataclass(frozen=True)
class Dimension:
    name: str
    labels: list[datetime] | list[float] | list[str]


@dataclass(slots=True, repr=False, eq=False)
class TimeSeriesCube:
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

    def sel(self, **kwargs) -> TimeSeriesCube | "TimeSeriesTable" | "TimeSeries":
        remaining_dims = list(self.dimensions)
        values = self._values

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
                try:
                    idx = list(dim.labels).index(selector)
                except ValueError:
                    raise KeyError(
                        f"Label {selector!r} not found in dimension {dim_name!r}"
                    ) from None
                values = np.take(values, idx, axis=axis)
                remaining_dims.pop(axis)

        return self._maybe_collapse(values, remaining_dims)

    def isel(self, **kwargs) -> TimeSeriesCube | "TimeSeriesTable" | "TimeSeries":
        remaining_dims = list(self.dimensions)
        values = self._values

        for dim_name, selector in kwargs.items():
            axis = next(
                i for i, d in enumerate(remaining_dims) if d.name == dim_name
            )
            dim = remaining_dims[axis]

            if isinstance(selector, slice):
                values = values[(slice(None),) * axis + (selector,)]
                remaining_dims[axis] = Dimension(dim.name, list(dim.labels)[selector])
            else:
                values = np.take(values, selector, axis=axis)
                remaining_dims.pop(axis)

        return self._maybe_collapse(values, remaining_dims)

    def _maybe_collapse(self, values, remaining_dims):
        ndim = values.ndim if hasattr(values, 'ndim') else 0

        if ndim == 0:
            raise ValueError("Selection collapsed all dimensions; scalar result.")

        if ndim >= 3:
            return TimeSeriesCube(
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
    ) -> TimeSeriesCube:
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
    ) -> TimeSeriesCube:
        if not series:
            raise ValueError("Cannot build cube from an empty list of TimeSeries.")
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

        # Box drawing
        padding = 2
        max_w = max(len(line) for line in meta_lines)
        box_inner = max_w + padding * 2

        lines: list[str] = [class_name]
        lines.append("\u250c" + "\u2500" * box_inner + "\u2510")
        for line in meta_lines:
            lines.append(
                "\u2502" + " " * padding + line.ljust(max_w) + " " * padding + "\u2502"
            )
        lines.append("\u2514" + "\u2500" * box_inner + "\u2518")
        return "\n".join(lines)

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

        # Build a 2D preview slice: first two dims, index 0 on remaining
        if n_dims >= 2:
            # Slice down to 2D
            slice_vals = self._values
            slice_dims = list(self.dimensions)
            while len(slice_dims) > 2:
                slice_vals = np.take(slice_vals, 0, axis=len(slice_dims) - 1)
                slice_dims = slice_dims[:-1]

            dim0 = slice_dims[0]
            dim1 = slice_dims[1]
            n_rows = len(dim0.labels)
            col_names = tuple(str(lbl) for lbl in dim1.labels)

            def _html_row(i: int) -> str:
                ts_cell = f"<td>{escape(str(dim0.labels[i]))}</td>"
                val_cells = "".join(
                    f"<td>{escape(_TimeSeriesBase._fmt_value(float(v)))}</td>"
                    for v in np.ma.filled(slice_vals[i], fill_value=np.nan)
                )
                return f"<tr>{ts_cell}{val_cells}</tr>"

            return _build_repr_html(
                class_name=type(self).__name__,
                meta_rows=meta_rows,
                index_names=(dim0.name,),
                column_names=col_names,
                n_rows=n_rows,
                html_row_fn=_html_row,
            )
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
        if not isinstance(other, TimeSeriesCube):
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
        if not isinstance(other, TimeSeriesCube):
            return NotImplemented
        return self.equals(other)


NDTimeSeries = TimeSeriesCube
