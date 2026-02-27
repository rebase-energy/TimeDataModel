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
    _import_pandas,
)
from .coverage import CoverageBar
from .enums import DataType, Frequency, TimeSeriesType
from .location import GeoArea, GeoLocation, Location


@dataclass(slots=True, repr=False, eq=False)
class TimeSeriesTable(_TimeSeriesBase):
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

    def _coverage_masks(self) -> list[tuple[str, list[bool]]]:
        masks: list[tuple[str, list[bool]]] = []
        for col, name in enumerate(self.column_names):
            col_data = self._values[:, col]
            masks.append((name, [not np.isnan(v) for v in col_data]))
        return masks

    def coverage_bar(self) -> CoverageBar:
        """Return a displayable coverage bar."""
        return CoverageBar(self._coverage_masks(), self.begin, self.end)

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

    # ---- equality --------------------------------------------------------

    def equals(self, other: object) -> bool:
        """Full structural equality (all metadata + NaN-aware values)."""
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
            or self._timestamps != other._timestamps
        ):
            return False
        return bool(np.array_equal(self._values, other._values, equal_nan=True))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeSeriesTable):
            return NotImplemented
        return self.equals(other)

    __hash__ = None

    # ---- column extraction ------------------------------------------------

    def select_column(self, col: int | str) -> "TimeSeries":
        """Extract a single column as a univariate TimeSeries."""
        from .timeseries import TimeSeries

        if isinstance(col, str):
            names = self.column_names
            if col not in names:
                raise KeyError(f"Column '{col}' not found. Available: {names}")
            col = names.index(col)

        arr = self._values[:, col]
        values = self._from_float_array(arr)

        return TimeSeries(
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
            index_names=self._index_names,
        )

    def to_univariate_list(self) -> list["TimeSeries"]:
        """Convert to a list of univariate TimeSeries, one per column."""
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

    # ---- scalar arithmetic -----------------------------------------------

    def _apply_scalar(self, func) -> TimeSeriesTable:
        arr = self._values.astype(np.float64, copy=True)
        return self._clone_with(list(self._timestamps), func(arr))

    def __add__(self, scalar: float) -> TimeSeriesTable:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: v + scalar)

    def __radd__(self, scalar: float) -> TimeSeriesTable:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: scalar + v)

    def __sub__(self, scalar: float) -> TimeSeriesTable:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: v - scalar)

    def __rsub__(self, scalar: float) -> TimeSeriesTable:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: scalar - v)

    def __mul__(self, scalar: float) -> TimeSeriesTable:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: v * scalar)

    def __rmul__(self, scalar: float) -> TimeSeriesTable:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: scalar * v)

    def __truediv__(self, scalar: float) -> TimeSeriesTable:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: v / scalar)

    def __neg__(self) -> TimeSeriesTable:
        return self._apply_scalar(lambda v: -v)

    def __abs__(self) -> TimeSeriesTable:
        return self._apply_scalar(abs)

    def __round__(self, n: int = 0) -> TimeSeriesTable:
        arr = self._values.astype(np.float64, copy=True)
        return self._clone_with(list(self._timestamps), np.round(arr, n))

    # ---- repr hooks -------------------------------------------------------

    def _repr_meta_lines(self) -> list[str]:
        label_w = 18
        lines: list[str] = []
        cn = self.column_names
        lines.append(f"{'Columns:':<{label_w}}{', '.join(cn)}")
        n = len(self._timestamps)
        lines.append(f"{'Shape:':<{label_w}}({n}, {self.n_columns})")
        lines.append(f"{'Frequency:':<{label_w}}{self.frequency}")
        lines.append(f"{'Timezone:':<{label_w}}{self.timezone}")

        # Unit — show if any is set
        unit_vals = [self._get_attr(self.units, i) for i in range(self.n_columns)]
        if any(u is not None for u in unit_vals):
            lines.append(f"{'Unit:':<{label_w}}{', '.join(str(u) if u else '-' for u in unit_vals)}")

        # Data type — show if any is set
        dt_vals = [self._get_attr(self.data_types, i) for i in range(self.n_columns)]
        if any(d is not None for d in dt_vals):
            lines.append(f"{'Data type:':<{label_w}}{', '.join(str(d) if d else '-' for d in dt_vals)}")

        # Location — show if any is set
        loc_vals = [self._get_attr(self.locations, i) for i in range(self.n_columns)]
        if any(loc is not None for loc in loc_vals):
            lines.append(f"{'Location:':<{label_w}}{', '.join(self._fmt_location(loc) or '-' for loc in loc_vals)}")

        # Timeseries type — show if any is not FLAT
        tst_vals = [self._get_attr(self.timeseries_types, i) for i in range(self.n_columns)]
        if any(t != "FLAT" for t in tst_vals):
            lines.append(f"{'Timeseries type:':<{label_w}}{', '.join(str(t) for t in tst_vals)}")

        return lines

    def _repr_data_rows(self, indices: list[int]) -> list[list[str]]:
        return [
            [str(self._timestamps[i])]
            + [self._fmt_value(float(v)) for v in self._values[i]]
            for i in indices
        ]

    def _repr_html_(self) -> str:
        cn = self.column_names
        ncols = self.n_columns
        n = len(self._timestamps)

        label = ", ".join(escape(c) for c in cn)
        meta_rows: list[tuple[str, str]] = [
            ("Columns", label),
            ("Shape", f"({n:,}, {ncols})"),
            ("Frequency", escape(str(self.frequency))),
            ("Timezone", escape(self.timezone)),
        ]

        def _html_row(i: int) -> str:
            ts = self._timestamps[i]
            if isinstance(ts, tuple):
                ts_cells = "".join(
                    f"<td>{escape(str(t))}</td>" for t in ts
                )
            else:
                ts_cells = f"<td>{escape(str(ts))}</td>"
            val_cells = "".join(
                f"<td>{escape(self._fmt_value(float(v)))}</td>"
                for v in self._values[i]
            )
            return f"<tr>{ts_cells}{val_cells}</tr>"

        return _build_repr_html(
            class_name=type(self).__name__,
            meta_rows=meta_rows,
            index_names=self.index_names,
            column_names=cn,
            n_rows=n,
            html_row_fn=_html_row,
        )

    # ---- numpy / pandas / polars -----------------------------------------

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

        arr = self.to_numpy()
        for j, cname in enumerate(self.column_names):
            data[cname] = arr[:, j]

        return pl.DataFrame(data)

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
    ) -> TimeSeriesTable:
        """Create a TimeSeriesTable from a pandas DataFrame."""
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
            index_names=index_names,
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

        payload: dict = {
            "timestamps": ts_json,
            "values": self._values.tolist(),
            "column_names": list(self.column_names),
            "frequency": str(self.frequency),
            "timezone": self.timezone,
        }
        if any(n is not None for n in self.names):
            payload["names"] = self.names
        if any(u is not None for u in self.units):
            payload["units"] = self.units
        if any(d is not None for d in self.descriptions):
            payload["descriptions"] = self.descriptions
        if any(d is not None for d in self.data_types):
            payload["data_types"] = [str(d) if d else None for d in self.data_types]
        if any(t != TimeSeriesType.FLAT for t in self.timeseries_types):
            payload["timeseries_types"] = [str(t) for t in self.timeseries_types]
        if any(a for a in self.attributes):
            payload["attributes"] = self.attributes
        if self._index_names is not None:
            payload["index_names"] = self._index_names
        return json.dumps(payload)

    @classmethod
    def from_json(
        cls,
        s: str,
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
    ) -> TimeSeriesTable:
        """Reconstruct from a JSON string produced by to_json()."""
        data = json.loads(s)
        raw_ts = data["timestamps"]

        if raw_ts and isinstance(raw_ts[0], list):
            timestamps = [
                tuple(datetime.fromisoformat(dt) for dt in row)
                for row in raw_ts
            ]
        else:
            timestamps = [datetime.fromisoformat(t) for t in raw_ts]

        values = np.array(data["values"], dtype=np.float64)
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

        if names is None:
            names = data.get("names") or data.get("column_names")
        if units is None:
            units = data.get("units")
        if descriptions is None:
            descriptions = data.get("descriptions")
        if data_types is None:
            raw_dt = data.get("data_types")
            if raw_dt:
                data_types = [DataType(d) if d else None for d in raw_dt]
        if timeseries_types is None:
            raw_tst = data.get("timeseries_types")
            if raw_tst:
                timeseries_types = [TimeSeriesType(t) for t in raw_tst]
        if attributes is None:
            attributes = data.get("attributes")

        return cls(
            freq,
            timezone=tz,
            timestamps=timestamps,
            values=values,
            names=names,
            units=units,
            descriptions=descriptions,
            data_types=data_types,
            locations=locations,
            timeseries_types=timeseries_types,
            attributes=attributes,
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
                val_cells = [
                    "" if np.isnan(v) else v for v in self._values[i]
                ]
                writer.writerow(ts_cells + val_cells)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        frequency: Frequency,
        *,
        timezone: str = "UTC",
        names: list[str | None] | None = None,
        units: list[str | None] | None = None,
        descriptions: list[str | None] | None = None,
        data_types: list[DataType | None] | None = None,
        locations: list[Location | None] | None = None,
        timeseries_types: list[TimeSeriesType] | None = None,
        attributes: list[dict[str, str]] | None = None,
    ) -> TimeSeriesTable:
        """Read a TimeSeriesTable from a CSV file produced by to_csv()."""
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

                rows.append([
                    np.nan if row[i] == "" else float(row[i])
                    for i in val_cols
                ])

        values = np.array(rows, dtype=np.float64)
        index_names = (
            [header[i] for i in idx_cols] if multi_index else None
        )

        if names is None:
            names = [header[i] for i in val_cols]

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
            index_names=index_names,
        )

    def validate(self) -> list[str]:
        """Return a list of validation warnings."""
        warnings = super().validate()
        n_values = self._values.shape[0]
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
    ) -> TimeSeriesTable:
        """Apply a pandas transformation, preserving metadata and auto-detecting frequency."""
        pd = _import_pandas()
        df = self.to_pandas_dataframe()
        result = func(df)
        new_freq, new_tz = self._infer_freq_tz(
            result, self.frequency, self.timezone
        )

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

        values = result.to_numpy(dtype=np.float64)
        new_names = [str(c) for c in result.columns]
        new_ncols = len(result.columns)

        def _carry_over(attr, default_factory):
            if len(attr) == 1 or len(attr) == new_ncols:
                return list(attr)
            return [default_factory()]

        return TimeSeriesTable(
            new_freq,
            timezone=new_tz,
            timestamps=timestamps,
            values=values,
            names=new_names,
            units=_carry_over(self.units, lambda: None),
            descriptions=_carry_over(self.descriptions, lambda: None),
            data_types=_carry_over(self.data_types, lambda: None),
            locations=_carry_over(self.locations, lambda: None),
            timeseries_types=_carry_over(
                self.timeseries_types, lambda: TimeSeriesType.FLAT
            ),
            attributes=_carry_over(self.attributes, dict),
            index_names=index_names,
        )

    def apply_numpy(
        self,
        func: Callable[[np.ndarray], np.ndarray],
    ) -> TimeSeriesTable:
        """Apply a numpy transformation to values, keeping timestamps and resolution unchanged."""
        arr = self.to_numpy()
        result = np.asarray(func(arr), dtype=np.float64)
        if result.shape[0] != len(self._timestamps):
            raise ValueError(
                f"apply_numpy: result length ({result.shape[0]}) must match "
                f"series length ({len(self._timestamps)})"
            )
        return self._clone_with(list(self._timestamps), result)


MultivariateTimeSeries = TimeSeriesTable
MultiTimeSeries = TimeSeriesTable
