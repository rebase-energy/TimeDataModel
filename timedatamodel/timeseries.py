from __future__ import annotations

import bisect
import csv
import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Callable, Iterator, NamedTuple, overload

import numpy as np
import pandas as pd

from .enums import Frequency
from .location import GeoArea, GeoLocation
from .metadata import Metadata
from .resolution import Resolution

_PANDAS_FREQ_MAP: dict[str, Frequency] = {
    # Modern pandas (>=2.0) aliases
    "YE": Frequency.P1Y, "YE-DEC": Frequency.P1Y, "A": Frequency.P1Y,
    "QE": Frequency.P3M, "QE-DEC": Frequency.P3M, "Q": Frequency.P3M,
    "ME": Frequency.P1M, "M": Frequency.P1M,
    "W": Frequency.P1W, "W-SUN": Frequency.P1W,
    "D": Frequency.P1D,
    "h": Frequency.PT1H, "H": Frequency.PT1H,
    "30min": Frequency.PT30M, "30T": Frequency.PT30M,
    "15min": Frequency.PT15M, "15T": Frequency.PT15M,
    "10min": Frequency.PT10M, "10T": Frequency.PT10M,
    "5min": Frequency.PT5M, "5T": Frequency.PT5M,
    "min": Frequency.PT1M, "T": Frequency.PT1M,
    "s": Frequency.PT1S, "S": Frequency.PT1S,
}

_MAX_PREVIEW = 3  # rows shown at head/tail in repr


class DataPoint(NamedTuple):
    timestamp: datetime
    value: float | None


@dataclass(slots=True)
class TimeSeries:
    resolution: Resolution
    metadata: Metadata = field(default_factory=Metadata)
    _timestamps: list[datetime] | list[tuple[datetime, ...]] = field(default_factory=list, repr=False)
    _values: list[float | None] | np.ndarray = field(default_factory=list, repr=False)
    _index_names: list[str] | None = field(default=None, repr=False)
    _column_names: list[str] | None = field(default=None, repr=False)

    def __init__(
        self,
        resolution: Resolution,
        metadata: Metadata | None = None,
        *,
        timestamps: list[datetime] | list[tuple[datetime, ...]] | None = None,
        values: list[float | None] | np.ndarray | None = None,
        data: list[DataPoint] | None = None,
        index_names: list[str] | None = None,
        column_names: list[str] | None = None,
    ) -> None:
        self.resolution = resolution
        self.metadata = metadata or Metadata()
        self._index_names = index_names
        self._column_names = column_names

        if data is not None:
            if timestamps is not None or values is not None:
                raise ValueError("cannot specify both 'data' and 'timestamps'/'values'")
            self._timestamps = [dp.timestamp for dp in data]
            self._values = [dp.value for dp in data]
        else:
            self._timestamps = timestamps or []
            self._values = values if values is not None else []

    @property
    def timestamps(self) -> list[datetime] | list[tuple[datetime, ...]]:
        return self._timestamps

    @property
    def values(self) -> list[float | None] | np.ndarray:
        return self._values

    @property
    def is_multi_index(self) -> bool:
        """True if timestamps are tuples of datetimes."""
        return len(self._timestamps) > 0 and isinstance(self._timestamps[0], tuple)

    @property
    def is_multivariate(self) -> bool:
        """True if values is a 2D ndarray."""
        return isinstance(self._values, np.ndarray) and self._values.ndim == 2

    @property
    def ndim(self) -> int:
        """Number of value columns (1 for univariate)."""
        if self.is_multivariate:
            return self._values.shape[1]
        return 1

    @property
    def index_names(self) -> tuple[str, ...]:
        """Labels for the index dimensions."""
        if self._index_names is not None:
            return tuple(self._index_names)
        if self.is_multi_index and self._timestamps:
            return tuple(f"index_{i}" for i in range(len(self._timestamps[0])))
        return ("timestamp",)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Labels for the value columns."""
        if self._column_names is not None:
            return tuple(self._column_names)
        if self.is_multivariate:
            return tuple(f"value_{i}" for i in range(self._values.shape[1]))
        return (self.metadata.name or "value",)

    def __len__(self) -> int:
        return len(self._timestamps)

    @overload
    def __getitem__(self, index: int) -> DataPoint | tuple: ...
    @overload
    def __getitem__(self, index: slice) -> list[DataPoint] | list[tuple]: ...

    def __getitem__(self, index: int | slice) -> DataPoint | tuple | list[DataPoint] | list[tuple]:
        use_datapoint = not self.is_multi_index and not self.is_multivariate
        if isinstance(index, slice):
            vals = self._values[index]
            if use_datapoint:
                return [
                    DataPoint(t, v)
                    for t, v in zip(self._timestamps[index], vals)
                ]
            return [
                (t, v) for t, v in zip(
                    self._timestamps[index],
                    vals if isinstance(vals, np.ndarray) else vals,
                )
            ]
        val = self._values[index]
        if isinstance(val, np.ndarray):
            val = val.tolist()
        if use_datapoint:
            return DataPoint(self._timestamps[index], val)
        return (self._timestamps[index], val)

    def __iter__(self) -> Iterator[DataPoint | tuple]:
        if not self.is_multi_index and not self.is_multivariate:
            return (DataPoint(t, v) for t, v in zip(self._timestamps, self._values))
        if self.is_multivariate:
            return (
                (t, self._values[i].tolist()) for i, t in enumerate(self._timestamps)
            )
        return ((t, v) for t, v in zip(self._timestamps, self._values))

    def __bool__(self) -> bool:
        return len(self._timestamps) > 0

    @property
    def begin(self) -> datetime | tuple[datetime, ...] | None:
        return self._timestamps[0] if self._timestamps else None

    @property
    def end(self) -> datetime | tuple[datetime, ...] | None:
        return self._timestamps[-1] if self._timestamps else None

    # ---- Tier 1 — trivial properties/methods -------------------------

    @property
    def duration(self) -> timedelta | None:
        """Time span from begin to end; None if empty."""
        if not self._timestamps:
            return None
        first = self._timestamps[0]
        last = self._timestamps[-1]
        if isinstance(first, tuple):
            return last[0] - first[0]
        return last - first

    @property
    def has_missing(self) -> bool:
        """True if any value is None (list) or NaN (ndarray)."""
        if isinstance(self._values, np.ndarray):
            return bool(np.isnan(self._values).any())
        return any(v is None for v in self._values)

    def __contains__(self, dt: datetime) -> bool:
        """Return True if *dt* appears in the timestamps."""
        i = bisect.bisect_left(self._timestamps, dt)
        return i < len(self._timestamps) and self._timestamps[i] == dt

    def head(self, n: int = 5) -> TimeSeries:
        """Return a new TimeSeries with the first *n* points."""
        return TimeSeries(
            self.resolution,
            self.metadata,
            timestamps=self._timestamps[:n],
            values=self._values[:n],
            index_names=self._index_names,
            column_names=self._column_names,
        )

    def tail(self, n: int = 5) -> TimeSeries:
        """Return a new TimeSeries with the last *n* points."""
        if n == 0:
            vals: list[float | None] | np.ndarray = (
                self._values[:0] if isinstance(self._values, np.ndarray) else []
            )
            return TimeSeries(
                self.resolution, self.metadata, timestamps=[], values=vals,
                index_names=self._index_names, column_names=self._column_names,
            )
        return TimeSeries(
            self.resolution,
            self.metadata,
            timestamps=self._timestamps[-n:],
            values=self._values[-n:],
            index_names=self._index_names,
            column_names=self._column_names,
        )

    def copy(self) -> TimeSeries:
        """Return a shallow copy (timestamps and values lists/arrays are new)."""
        if isinstance(self._values, np.ndarray):
            new_values = self._values.copy()
        else:
            new_values = list(self._values)
        return TimeSeries(
            self.resolution,
            self.metadata,
            timestamps=list(self._timestamps),
            values=new_values,
            index_names=self._index_names,
            column_names=self._column_names,
        )

    # ---- Tier 5 — scalar arithmetic operators ------------------------

    def _result_values(self, arr: np.ndarray) -> list[float | None] | np.ndarray:
        """Return ndarray for multivariate, list for univariate."""
        if self.is_multivariate:
            return arr
        return self._from_float_array(arr)

    def _apply_scalar(self, func) -> TimeSeries:
        arr = self._to_float_array()
        return TimeSeries(
            self.resolution,
            self.metadata,
            timestamps=list(self._timestamps),
            values=self._result_values(func(arr)),
            index_names=self._index_names,
            column_names=self._column_names,
        )

    def __add__(self, scalar: float) -> TimeSeries:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: v + scalar)

    def __radd__(self, scalar: float) -> TimeSeries:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: scalar + v)

    def __sub__(self, scalar: float) -> TimeSeries:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: v - scalar)

    def __rsub__(self, scalar: float) -> TimeSeries:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: scalar - v)

    def __mul__(self, scalar: float) -> TimeSeries:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: v * scalar)

    def __rmul__(self, scalar: float) -> TimeSeries:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: scalar * v)

    def __truediv__(self, scalar: float) -> TimeSeries:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: v / scalar)

    def __neg__(self) -> TimeSeries:
        return self._apply_scalar(lambda v: -v)

    def __abs__(self) -> TimeSeries:
        return self._apply_scalar(abs)

    def __round__(self, n: int = 0) -> TimeSeries:
        arr = self._to_float_array()
        return TimeSeries(
            self.resolution,
            self.metadata,
            timestamps=list(self._timestamps),
            values=self._result_values(np.round(arr, n)),
            index_names=self._index_names,
            column_names=self._column_names,
        )

    # ---- repr helpers ------------------------------------------------

    @staticmethod
    def _fmt_value(v: float | None) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "NaN"
        if v == int(v):
            return f"{v:.1f}"
        return f"{v:g}"

    @staticmethod
    def _fmt_location(loc: GeoLocation | GeoArea | None) -> str:
        if loc is None:
            return ""
        if isinstance(loc, GeoLocation):
            return f"{loc.latitude}\u00b0N, {loc.longitude}\u00b0E"
        name = loc.name or "unnamed"
        c = loc.centroid
        return f"{name} (centroid {c.latitude}\u00b0N, {c.longitude}\u00b0E)"

    def _fmt_row_value(self, i: int) -> str:
        """Format value at row *i* for repr."""
        if self.is_multivariate:
            row = self._values[i]
            return "  ".join(self._fmt_value(float(v)) for v in row)
        v = self._values[i]
        return self._fmt_value(v)

    def __repr__(self) -> str:
        m = self.metadata
        name = m.name or "unnamed"
        n = len(self._timestamps)

        header = f"<TimeSeries '{name}' ({n} points)>"
        lines = [header]

        # Resolution
        r = self.resolution
        lines.append(f"Resolution:  {r.frequency} ({r.timezone})")

        # Metadata fields (skip None)
        if m.unit:
            lines.append(f"Unit:        {m.unit}")
        if m.data_type:
            lines.append(f"Data type:   {m.data_type}")
        if m.location:
            lines.append(f"Location:    {self._fmt_location(m.location)}")
        if m.storage_type and m.storage_type != "FLAT":
            lines.append(f"Storage:     {m.storage_type}")

        # Data preview
        if n == 0:
            lines.append("Data:        (empty)")
        else:
            lines.append("Data:")
            show_all = n <= _MAX_PREVIEW * 2 + 1
            if show_all:
                rows = range(n)
            else:
                rows = list(range(_MAX_PREVIEW)) + list(range(n - _MAX_PREVIEW, n))

            # Compute column widths from visible rows
            ts_strs = {i: str(self._timestamps[i]) for i in rows}
            val_strs = {i: self._fmt_row_value(i) for i in rows}
            ts_w = max(len(s) for s in ts_strs.values())
            val_w = max(len(s) for s in val_strs.values())

            for i in rows:
                if not show_all and i == n - _MAX_PREVIEW:
                    lines.append(f"    {'...':<{ts_w}}  {'...':<{val_w}}")
                lines.append(
                    f"    {ts_strs[i]:<{ts_w}}  {val_strs[i]:>{val_w}}"
                )

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        m = self.metadata
        name = escape(m.name or "unnamed")
        n = len(self._timestamps)

        # --- CSS (scoped via unique class) ---
        css = """\
<style>
.ts-repr { font-family: monospace; font-size: 13px; max-width: 640px; }
.ts-repr .ts-header {
  font-weight: bold; font-size: 14px;
  padding: 6px 10px; border-bottom: 2px solid #4a4a4a;
  background: #f0f0f0; color: #1a1a1a;
}
.ts-repr .ts-meta { padding: 6px 10px; background: #fafafa; }
.ts-repr .ts-meta table { border-collapse: collapse; }
.ts-repr .ts-meta td { padding: 1px 8px 1px 0; white-space: nowrap; }
.ts-repr .ts-meta td:first-child { color: #666; font-weight: 600; }
.ts-repr .ts-data { padding: 6px 10px; }
.ts-repr .ts-data table {
  border-collapse: collapse; width: 100%; text-align: right;
}
.ts-repr .ts-data th {
  text-align: left; padding: 3px 10px; border-bottom: 1px solid #ccc;
  color: #555; font-weight: 600;
}
.ts-repr .ts-data td { padding: 2px 10px; }
.ts-repr .ts-data tr:hover { background: #f5f5f5; }
.ts-repr .ts-data td:first-child { text-align: left; color: #333; }
.ts-repr .ts-ellipsis { text-align: center !important; color: #999; }
</style>"""

        # --- Header ---
        html = [css, '<div class="ts-repr">']
        html.append(
            f'<div class="ts-header">'
            f'TimeSeries &mdash; <em>{name}</em> &nbsp; ({n:,} points)'
            f'</div>'
        )

        # --- Metadata table ---
        html.append('<div class="ts-meta"><table>')
        r = self.resolution
        html.append(
            f"<tr><td>Frequency</td><td>{escape(str(r.frequency))}</td></tr>"
        )
        html.append(f"<tr><td>Timezone</td><td>{escape(r.timezone)}</td></tr>")
        if m.unit:
            html.append(f"<tr><td>Unit</td><td>{escape(m.unit)}</td></tr>")
        if m.data_type:
            html.append(
                f"<tr><td>Data type</td><td>{escape(str(m.data_type))}</td></tr>"
            )
        if m.location:
            html.append(
                f"<tr><td>Location</td>"
                f"<td>{escape(self._fmt_location(m.location))}</td></tr>"
            )
        if m.description:
            html.append(
                f"<tr><td>Description</td>"
                f"<td>{escape(m.description)}</td></tr>"
            )
        html.append("</table></div>")

        # --- Data table ---
        idx_names = self.index_names
        col_names = self.column_names
        n_cols = len(idx_names) + len(col_names)
        html.append('<div class="ts-data"><table>')
        header_cells = "".join(f"<th>{escape(c)}</th>" for c in idx_names)
        header_cells += "".join(f"<th>{escape(c)}</th>" for c in col_names)
        html.append(f"<tr>{header_cells}</tr>")

        def _html_row(i: int) -> str:
            ts = self._timestamps[i]
            if isinstance(ts, tuple):
                ts_cells = "".join(f"<td>{escape(str(t))}</td>" for t in ts)
            else:
                ts_cells = f"<td>{escape(str(ts))}</td>"
            if self.is_multivariate:
                val_cells = "".join(
                    f"<td>{escape(self._fmt_value(float(v)))}</td>"
                    for v in self._values[i]
                )
            else:
                val_cells = f"<td>{escape(self._fmt_value(self._values[i]))}</td>"
            return f"<tr>{ts_cells}{val_cells}</tr>"

        if n == 0:
            html.append(
                f'<tr><td colspan="{n_cols}" class="ts-ellipsis">(empty)</td></tr>'
            )
        else:
            show_all = n <= _MAX_PREVIEW * 2 + 1
            head_rows = range(min(_MAX_PREVIEW, n))
            tail_rows = range(max(n - _MAX_PREVIEW, _MAX_PREVIEW), n) if not show_all else range(0)

            for i in head_rows:
                html.append(_html_row(i))

            if not show_all:
                ellipsis_cells = "".join(
                    f'<td class="ts-ellipsis">&hellip;</td>' for _ in range(n_cols)
                )
                html.append(f"<tr>{ellipsis_cells}</tr>")
                for i in tail_rows:
                    html.append(_html_row(i))

        html.append("</table></div>")
        html.append("</div>")
        return "\n".join(html)

    def to_numpy(self) -> np.ndarray:
        """Return values as a numpy float64 array (None -> nan)."""
        return self._to_float_array()

    def to_pandas_dataframe(self) -> pd.DataFrame:
        """Return a pandas DataFrame with DatetimeIndex or MultiIndex."""
        arr = self.to_numpy()
        if self.is_multi_index:
            idx_names = list(self.index_names)
            index = pd.MultiIndex.from_tuples(self._timestamps, names=idx_names)
        else:
            index = pd.DatetimeIndex(self._timestamps, name=self.index_names[0])

        if self.is_multivariate:
            cols = list(self.column_names)
            return pd.DataFrame(arr, index=index, columns=cols)
        col_name = self.metadata.name or "value"
        return pd.DataFrame({col_name: arr}, index=index)

    def to_pd_df(self) -> pd.DataFrame:
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

        data = {}
        if self.is_multi_index:
            for i, name in enumerate(self.index_names):
                data[name] = [t[i] for t in self._timestamps]
        else:
            data[self.index_names[0]] = self._timestamps

        arr = self._to_float_array()
        if self.is_multivariate:
            for j, name in enumerate(self.column_names):
                data[name] = arr[:, j]
        else:
            col_name = self.metadata.name or "value"
            data[col_name] = arr

        return pl.DataFrame(data)

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        resolution: Resolution | None = None,
        metadata: Metadata | None = None,
        value_column: str | None = None,
    ) -> TimeSeries:
        """Create a TimeSeries from a pandas DataFrame.

        Supports DatetimeIndex (single-index) and MultiIndex (multi-index).
        Multiple value columns produce a multivariate TimeSeries.
        If *resolution* is ``None``, the frequency and timezone are
        auto-inferred from the DataFrame's DatetimeIndex.
        """
        # Determine timestamps
        if isinstance(df.index, pd.MultiIndex):
            timestamps = [
                tuple(
                    lvl.to_pydatetime() if hasattr(lvl, 'to_pydatetime') else lvl
                    for lvl in row
                )
                for row in df.index
            ]
            index_names = list(df.index.names)
        elif isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index.to_pydatetime().tolist()
            index_names = None
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or MultiIndex")

        # Determine values
        if value_column is not None:
            cols = [value_column]
        else:
            cols = list(df.columns)

        if len(cols) == 1:
            arr = df[cols[0]].to_numpy(dtype=np.float64, na_value=np.nan)
            values = cls._from_float_array(arr)
            column_names = None
        else:
            values = df[cols].to_numpy(dtype=np.float64)
            column_names = cols

        meta = metadata or Metadata(name=cols[0] if len(cols) == 1 else None)

        if resolution is None:
            if isinstance(df.index, pd.DatetimeIndex):
                resolution = cls._infer_resolution(df, Resolution(Frequency.NONE))
            else:
                resolution = Resolution(Frequency.NONE)

        return cls(
            resolution, meta, timestamps=timestamps, values=values,
            index_names=index_names, column_names=column_names,
        )

    @classmethod
    def from_polars(
        cls,
        df,
        resolution: Resolution,
        metadata: Metadata | None = None,
        timestamp_column: str = "timestamp",
        value_column: str | None = None,
    ) -> TimeSeries:
        """Create a TimeSeries from a polars DataFrame."""
        val_col = value_column or [c for c in df.columns if c != timestamp_column][0]
        timestamps = df[timestamp_column].to_list()
        arr = df[val_col].to_numpy(allow_copy=True)
        values = cls._from_float_array(arr)
        meta = metadata or Metadata(name=val_col)
        return cls(resolution, meta, timestamps=timestamps, values=values)

    def update_from_pandas(
        self,
        df: pd.DataFrame,
        value_column: str | None = None,
        inplace: bool = False,
    ) -> TimeSeries | None:
        """Update a TimeSeries from a pandas DataFrame.

        By default returns a **new** TimeSeries.  With ``inplace=True``
        the current instance is mutated and ``None`` is returned.
        """
        # Determine timestamps
        if isinstance(df.index, pd.MultiIndex):
            timestamps = [
                tuple(
                    lvl.to_pydatetime() if hasattr(lvl, 'to_pydatetime') else lvl
                    for lvl in row
                )
                for row in df.index
            ]
            index_names = list(df.index.names)
        elif isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index.to_pydatetime().tolist()
            index_names = None
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or MultiIndex")

        # Determine values
        if value_column is not None:
            cols = [value_column]
        else:
            cols = list(df.columns)

        if len(cols) == 1:
            arr = df[cols[0]].to_numpy(dtype=np.float64, na_value=np.nan)
            values = self._from_float_array(arr)
            column_names = None
        else:
            values = df[cols].to_numpy(dtype=np.float64)
            column_names = cols

        new_resolution = self._infer_resolution_from_df(df) if isinstance(df.index, pd.DatetimeIndex) else self.resolution
        new_metadata = self._infer_metadata_from_df(df)
        if inplace:
            self._timestamps = timestamps
            self._values = values
            self._index_names = index_names
            self._column_names = column_names
            self.resolution = new_resolution
            self.metadata = new_metadata
            return None
        return TimeSeries(
            new_resolution, new_metadata, timestamps=timestamps, values=values,
            index_names=index_names, column_names=column_names,
        )

    # ---- Tier 6 — serialization I/O ----------------------------------

    def to_json(self) -> str:
        """Serialize to a JSON string (timestamps as ISO-8601 strings)."""
        if self.is_multi_index:
            ts_json = [[dt.isoformat() for dt in tup] for tup in self._timestamps]
        else:
            ts_json = [t.isoformat() for t in self._timestamps]

        if self.is_multivariate:
            val_json = self._values.tolist()
        else:
            val_json = self._values if isinstance(self._values, list) else self._values.tolist()

        payload = {"timestamps": ts_json, "values": val_json}
        if self._index_names is not None:
            payload["index_names"] = self._index_names
        if self._column_names is not None:
            payload["column_names"] = self._column_names
        return json.dumps(payload)

    @classmethod
    def from_json(
        cls,
        s: str,
        resolution: Resolution,
        metadata: Metadata | None = None,
    ) -> TimeSeries:
        """Reconstruct a TimeSeries from a JSON string produced by to_json()."""
        data = json.loads(s)
        raw_ts = data["timestamps"]

        # Detect multi-index: first element is a list
        if raw_ts and isinstance(raw_ts[0], list):
            timestamps = [
                tuple(datetime.fromisoformat(dt) for dt in row) for row in raw_ts
            ]
        else:
            timestamps = [datetime.fromisoformat(t) for t in raw_ts]

        raw_vals = data["values"]
        # Detect multivariate: first element is a list
        if raw_vals and isinstance(raw_vals[0], list):
            values = np.array(raw_vals, dtype=np.float64)
        else:
            values: list[float | None] = raw_vals

        index_names = data.get("index_names")
        column_names = data.get("column_names")
        return cls(
            resolution, metadata, timestamps=timestamps, values=values,
            index_names=index_names, column_names=column_names,
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
                if self.is_multivariate:
                    val_cells = [
                        "" if np.isnan(v) else v for v in self._values[i]
                    ]
                else:
                    v = self._values[i]
                    val_cells = ["" if v is None else v]
                writer.writerow(ts_cells + val_cells)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        resolution: Resolution,
        metadata: Metadata | None = None,
    ) -> TimeSeries:
        """Read a TimeSeries from a CSV file produced by to_csv()."""
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

            # Detect how many index columns vs value columns
            # Convention: columns whose names end with "time" or equal "timestamp" are index columns
            idx_cols = []
            val_cols = []
            for i, name in enumerate(header):
                if name in ("timestamp",) or name.endswith("_time"):
                    idx_cols.append(i)
                else:
                    val_cols.append(i)
            # Fallback: if no index columns detected, treat first column as index
            if not idx_cols:
                idx_cols = [0]
                val_cols = list(range(1, len(header)))

            multi_index = len(idx_cols) > 1
            multivariate = len(val_cols) > 1

            timestamps: list = []
            rows: list = []
            for row in reader:
                if multi_index:
                    timestamps.append(
                        tuple(datetime.fromisoformat(row[i]) for i in idx_cols)
                    )
                else:
                    timestamps.append(datetime.fromisoformat(row[idx_cols[0]]))

                if multivariate:
                    rows.append([
                        np.nan if row[i] == "" else float(row[i]) for i in val_cols
                    ])
                else:
                    raw = row[val_cols[0]] if val_cols else ""
                    rows.append(None if raw == "" else float(raw))

        if multivariate:
            values = np.array(rows, dtype=np.float64)
            column_names = [header[i] for i in val_cols]
        else:
            values = rows
            column_names = None

        index_names = [header[i] for i in idx_cols] if multi_index else None
        val_col_name = header[val_cols[0]] if val_cols and not multivariate else None
        meta = metadata or Metadata(name=val_col_name)
        return cls(
            resolution, meta, timestamps=timestamps, values=values,
            index_names=index_names, column_names=column_names,
        )

    def validate(self) -> list[str]:
        """Return a list of validation warnings."""
        warnings: list[str] = []

        n_values = self._values.shape[0] if isinstance(self._values, np.ndarray) else len(self._values)
        if len(self._timestamps) != n_values:
            warnings.append(
                f"length mismatch: {len(self._timestamps)} timestamps "
                f"vs {n_values} values"
            )

        td = self.resolution.to_timedelta()
        check_order = True
        check_freq = td is not None
        multi = self.is_multi_index
        for i in range(1, len(self._timestamps)):
            if check_order and self._timestamps[i] <= self._timestamps[i - 1]:
                warnings.append(
                    f"timestamps not strictly increasing at index {i}: "
                    f"{self._timestamps[i-1]} >= {self._timestamps[i]}"
                )
                check_order = False
            if check_freq:
                cur = self._timestamps[i][0] if multi else self._timestamps[i]
                prev = self._timestamps[i - 1][0] if multi else self._timestamps[i - 1]
                if cur - prev != td:
                    warnings.append(
                        f"inconsistent frequency at index {i}: "
                        f"expected {td}, got {cur - prev}"
                    )
                    check_freq = False
            if not check_order and not check_freq:
                break

        return warnings

    # ---- Tier 7 — pandas bridge --------------------------------------

    @staticmethod
    def _infer_resolution(df: pd.DataFrame, fallback: Resolution) -> Resolution:
        """Infer Resolution from a DataFrame's DatetimeIndex, falling back to *fallback*."""
        new_tz = str(df.index.tz) if df.index.tz is not None else fallback.timezone

        freq_str: str | None = None
        if df.index.freq is not None:
            freq_str = df.index.freqstr
        elif len(df.index) >= 3:
            try:
                freq_str = pd.infer_freq(df.index)
            except Exception:
                pass

        new_freq = _PANDAS_FREQ_MAP.get(freq_str, fallback.frequency) if freq_str else fallback.frequency

        if new_freq == fallback.frequency and new_tz == fallback.timezone:
            return fallback
        return Resolution(new_freq, new_tz)

    def _infer_resolution_from_df(self, result: pd.DataFrame) -> Resolution:
        return self._infer_resolution(result, self.resolution)

    def _infer_metadata_from_df(self, result: pd.DataFrame) -> Metadata:
        original_col = self.metadata.name or "value"
        new_col = result.columns[0] if len(result.columns) > 0 else original_col
        return (
            dataclasses.replace(self.metadata, name=new_col)
            if new_col != original_col
            else self.metadata
        )

    def apply_pandas(
        self,
        func: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> TimeSeries:
        """Apply a pandas transformation, preserving metadata and auto-detecting resolution."""
        df = self.to_pandas_dataframe()
        result = func(df)
        new_resolution = self._infer_resolution_from_df(result)
        new_metadata = self._infer_metadata_from_df(result)
        return TimeSeries.from_pandas(result, new_resolution, new_metadata)

    def apply_numpy(
        self,
        func: Callable[[np.ndarray], np.ndarray],
    ) -> TimeSeries:
        """Apply a numpy transformation to values, keeping timestamps and resolution unchanged."""
        arr = self.to_numpy()
        result = np.asarray(func(arr), dtype=np.float64)
        if result.shape[0] != len(self._timestamps):
            raise ValueError(
                f"apply_numpy: result length ({result.shape[0]}) must match "
                f"series length ({len(self._timestamps)})"
            )
        return TimeSeries(
            self.resolution,
            self.metadata,
            timestamps=list(self._timestamps),
            values=self._result_values(result),
            index_names=self._index_names,
            column_names=self._column_names,
        )

    # ---- Private array helpers ---------------------------------------

    def _to_float_array(self) -> np.ndarray:
        """Convert _values to float64 ndarray — None → NaN. Vectorised via pandas nullable array."""
        if isinstance(self._values, np.ndarray):
            return self._values.astype(np.float64, copy=True)
        return pd.array(self._values, dtype="Float64").to_numpy(dtype=np.float64, na_value=np.nan)

    @staticmethod
    def _from_float_array(arr: np.ndarray) -> list[float | None]:
        """Convert float64 ndarray to list[float|None] — NaN → None."""
        values: list[float | None] = arr.tolist()  # C-level, fast
        for i in np.where(np.isnan(arr))[0]:       # iterate only NaN positions
            values[i] = None
        return values
