from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from typing import Iterator, NamedTuple, overload

import numpy as np
import pandas as pd

from .location import GeoArea, GeoLocation
from .metadata import Metadata
from .resolution import Resolution

_MAX_PREVIEW = 3  # rows shown at head/tail in repr


class DataPoint(NamedTuple):
    timestamp: datetime
    value: float | None


@dataclass(slots=True)
class TimeSeries:
    resolution: Resolution
    metadata: Metadata = field(default_factory=Metadata)
    _timestamps: list[datetime] = field(default_factory=list, repr=False)
    _values: list[float | None] = field(default_factory=list, repr=False)

    def __init__(
        self,
        resolution: Resolution,
        metadata: Metadata | None = None,
        *,
        timestamps: list[datetime] | None = None,
        values: list[float | None] | None = None,
        data: list[DataPoint] | None = None,
    ) -> None:
        self.resolution = resolution
        self.metadata = metadata or Metadata()

        if data is not None:
            if timestamps is not None or values is not None:
                raise ValueError("cannot specify both 'data' and 'timestamps'/'values'")
            self._timestamps = [dp.timestamp for dp in data]
            self._values = [dp.value for dp in data]
        else:
            self._timestamps = timestamps or []
            self._values = values or []

    @property
    def timestamps(self) -> list[datetime]:
        return self._timestamps

    @property
    def values(self) -> list[float | None]:
        return self._values

    def __len__(self) -> int:
        return len(self._timestamps)

    @overload
    def __getitem__(self, index: int) -> DataPoint: ...
    @overload
    def __getitem__(self, index: slice) -> list[DataPoint]: ...

    def __getitem__(self, index: int | slice) -> DataPoint | list[DataPoint]:
        if isinstance(index, slice):
            return [
                DataPoint(t, v)
                for t, v in zip(self._timestamps[index], self._values[index])
            ]
        return DataPoint(self._timestamps[index], self._values[index])

    def __iter__(self) -> Iterator[DataPoint]:
        return (DataPoint(t, v) for t, v in zip(self._timestamps, self._values))

    def __bool__(self) -> bool:
        return len(self._timestamps) > 0

    # ---- repr helpers ------------------------------------------------

    @staticmethod
    def _fmt_value(v: float | None) -> str:
        if v is None:
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
            val_strs = {i: self._fmt_value(self._values[i]) for i in rows}
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
        col_name = escape(m.name or "value")
        html.append('<div class="ts-data"><table>')
        html.append(f"<tr><th>timestamp</th><th>{col_name}</th></tr>")

        if n == 0:
            html.append(
                '<tr><td colspan="2" class="ts-ellipsis">(empty)</td></tr>'
            )
        else:
            show_all = n <= _MAX_PREVIEW * 2 + 1
            head = range(min(_MAX_PREVIEW, n))
            tail = range(max(n - _MAX_PREVIEW, _MAX_PREVIEW), n) if not show_all else range(0)

            for i in head:
                ts_s = escape(str(self._timestamps[i]))
                v_s = escape(self._fmt_value(self._values[i]))
                html.append(f"<tr><td>{ts_s}</td><td>{v_s}</td></tr>")

            if not show_all:
                html.append(
                    '<tr><td class="ts-ellipsis">&hellip;</td>'
                    '<td class="ts-ellipsis">&hellip;</td></tr>'
                )
                for i in tail:
                    ts_s = escape(str(self._timestamps[i]))
                    v_s = escape(self._fmt_value(self._values[i]))
                    html.append(f"<tr><td>{ts_s}</td><td>{v_s}</td></tr>")

        html.append("</table></div>")
        html.append("</div>")
        return "\n".join(html)

    def to_numpy(self) -> np.ndarray:
        """Return values as a numpy float64 array (None -> nan)."""
        return np.array(
            [v if v is not None else float("nan") for v in self._values],
            dtype=np.float64,
        )

    def to_pandas_dataframe(self) -> pd.DataFrame:
        """Return a pandas DataFrame with DatetimeIndex."""
        col_name = self.metadata.name or "value"
        index = pd.DatetimeIndex(self._timestamps, name="timestamp")
        return pd.DataFrame({col_name: self.to_numpy()}, index=index)

    def to_polars_dataframe(self):
        """Return a polars DataFrame with 'timestamp' and value columns."""
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError(
                "polars is required for to_polars_dataframe(). "
                "Install it with: pip install timedatamodel[polars]"
            ) from e

        col_name = self.metadata.name or "value"
        return pl.DataFrame(
            {
                "timestamp": self._timestamps,
                col_name: [v if v is not None else float("nan") for v in self._values],
            }
        )

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        resolution: Resolution,
        metadata: Metadata | None = None,
        value_column: str | None = None,
    ) -> TimeSeries:
        """Create a TimeSeries from a pandas DataFrame with DatetimeIndex."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")
        col = value_column or df.columns[0]
        timestamps = df.index.to_pydatetime().tolist()
        values: list[float | None] = [
            None if pd.isna(v) else float(v) for v in df[col]
        ]
        meta = metadata or Metadata(name=col)
        return cls(resolution, meta, timestamps=timestamps, values=values)

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
        raw_values = df[val_col].to_list()
        values: list[float | None] = [
            None if v != v else float(v) for v in raw_values  # NaN != NaN
        ]
        meta = metadata or Metadata(name=val_col)
        return cls(resolution, meta, timestamps=timestamps, values=values)

    def validate(self) -> list[str]:
        """Return a list of validation warnings."""
        warnings: list[str] = []

        if len(self._timestamps) != len(self._values):
            warnings.append(
                f"length mismatch: {len(self._timestamps)} timestamps "
                f"vs {len(self._values)} values"
            )

        # Check ordering
        for i in range(1, len(self._timestamps)):
            if self._timestamps[i] <= self._timestamps[i - 1]:
                warnings.append(
                    f"timestamps not strictly increasing at index {i}: "
                    f"{self._timestamps[i-1]} >= {self._timestamps[i]}"
                )
                break

        # Check frequency consistency for fixed-duration frequencies
        td = self.resolution.to_timedelta()
        if td is not None and len(self._timestamps) > 1:
            for i in range(1, len(self._timestamps)):
                actual = self._timestamps[i] - self._timestamps[i - 1]
                if actual != td:
                    warnings.append(
                        f"inconsistent frequency at index {i}: "
                        f"expected {td}, got {actual}"
                    )
                    break

        return warnings
