from __future__ import annotations

import bisect
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Callable, Iterator, NamedTuple, overload

import numpy as np
import pandas as pd

from .enums import DataType, Frequency, TimeSeriesType
from .location import GeoArea, GeoLocation, Location
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


# ---------------------------------------------------------------------------
# CoverageBar — visual coverage indicator
# ---------------------------------------------------------------------------


def _fmt_short_date(dt: datetime) -> str:
    """Format a datetime concisely: omit time when midnight, omit tzinfo."""
    dt_naive = dt.replace(tzinfo=None)
    if dt_naive.hour == 0 and dt_naive.minute == 0 and dt_naive.second == 0:
        return dt_naive.strftime("%Y-%m-%d")
    return dt_naive.strftime("%Y-%m-%d %H:%M")


class CoverageBar:
    """Displayable coverage bar for TimeSeries objects."""

    _TERM_BINS = 40
    _SVG_BINS = 60

    def __init__(
        self,
        masks: list[tuple[str, list[bool]]],
        begin: datetime | None,
        end: datetime | None,
    ) -> None:
        self._masks = masks
        self._begin = begin
        self._end = end

    @staticmethod
    def _bin_coverage(mask: list[bool], n_bins: int) -> list[bool]:
        n = len(mask)
        if n == 0:
            return [False] * n_bins
        actual_bins = min(n_bins, n)
        bins: list[bool] = []
        for i in range(actual_bins):
            lo = i * n // actual_bins
            hi = (i + 1) * n // actual_bins
            bins.append(any(mask[lo:hi]))
        return bins

    def __repr__(self) -> str:
        if not self._masks:
            return ""
        n_bins = self._TERM_BINS
        label_w = max(len(name) for name, _ in self._masks) + 2

        lines: list[str] = []
        bar_len = 0
        for name, mask in self._masks:
            binned = self._bin_coverage(mask, n_bins)
            bar = "".join("\u2588" if b else "\u2591" for b in binned)
            bar_len = len(binned)
            lines.append(f"{name:<{label_w}}{bar}")

        start_str = _fmt_short_date(self._begin) if self._begin else ""
        end_str = _fmt_short_date(self._end) if self._end else ""
        date_line = f"{start_str:<{bar_len // 2}}{end_str:>{bar_len - bar_len // 2}}"
        lines.append(f"{'':<{label_w}}{date_line}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        if not self._masks:
            return ""
        n_bins = self._SVG_BINS
        # Use actual bin count (may be less than n_bins for short series)
        max_mask_len = max(len(m) for _, m in self._masks) if self._masks else 0
        actual_bins = min(n_bins, max_mask_len) if max_mask_len > 0 else n_bins
        label_w = 120  # px reserved for labels
        bar_w = 480  # px for the bar area
        row_h = 22
        n_rows = len(self._masks)
        date_h = 18
        total_h = n_rows * row_h + date_h + 4

        parts: list[str] = []
        parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {label_w + bar_w} {total_h}" '
            f'width="100%" style="max-width:{label_w + bar_w}px;'
            f'font-family:monospace;font-size:12px;">'
        )

        for row_idx, (name, mask) in enumerate(self._masks):
            y = row_idx * row_h
            # label
            parts.append(
                f'<text x="{label_w - 6}" y="{y + 15}" '
                f'text-anchor="end" fill="#333">{escape(name)}</text>'
            )
            # bar segments
            binned = self._bin_coverage(mask, n_bins)
            seg_w = bar_w / len(binned) if binned else bar_w
            for i, b in enumerate(binned):
                color = "#4CAF50" if b else "#e0e0e0"
                x = label_w + i * seg_w
                parts.append(
                    f'<rect x="{x:.1f}" y="{y + 2}" '
                    f'width="{seg_w:.2f}" height="{row_h - 4}" '
                    f'fill="{color}" />'
                )

        # date labels
        date_y = n_rows * row_h + date_h
        if self._begin:
            parts.append(
                f'<text x="{label_w}" y="{date_y}" '
                f'text-anchor="start" fill="#666">'
                f'{escape(_fmt_short_date(self._begin))}</text>'
            )
        if self._end:
            parts.append(
                f'<text x="{label_w + bar_w}" y="{date_y}" '
                f'text-anchor="end" fill="#666">'
                f'{escape(_fmt_short_date(self._end))}</text>'
            )

        parts.append("</svg>")
        return "\n".join(parts)


class DataPoint(NamedTuple):
    timestamp: datetime
    value: float | None


# ---------------------------------------------------------------------------
# _TimeSeriesBase — shared logic for univariate and multivariate
# ---------------------------------------------------------------------------


class _TimeSeriesBase:
    __slots__ = ()

    @property
    def timestamps(self) -> list[datetime] | list[tuple[datetime, ...]]:
        return self._timestamps

    @property
    def is_multi_index(self) -> bool:
        """True if timestamps are tuples of datetimes."""
        return len(self._timestamps) > 0 and isinstance(self._timestamps[0], tuple)

    @property
    def index_names(self) -> tuple[str, ...]:
        """Labels for the index dimensions."""
        if self._index_names is not None:
            return tuple(self._index_names)
        if self.is_multi_index and self._timestamps:
            return tuple(f"index_{i}" for i in range(len(self._timestamps[0])))
        return ("timestamp",)

    def __len__(self) -> int:
        return len(self._timestamps)

    def __bool__(self) -> bool:
        return len(self._timestamps) > 0

    def __contains__(self, dt: datetime) -> bool:
        """Return True if *dt* appears in the timestamps."""
        i = bisect.bisect_left(self._timestamps, dt)
        return i < len(self._timestamps) and self._timestamps[i] == dt

    @property
    def begin(self) -> datetime | tuple[datetime, ...] | None:
        return self._timestamps[0] if self._timestamps else None

    @property
    def end(self) -> datetime | tuple[datetime, ...] | None:
        return self._timestamps[-1] if self._timestamps else None

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

    def validate(self) -> list[str]:
        """Return a list of validation warnings (timestamp ordering and frequency)."""
        warnings: list[str] = []
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

    # ---- repr (shared) ---------------------------------------------------

    def __repr__(self) -> str:
        class_name = type(self).__name__
        meta_lines = self._repr_meta_lines()
        n = len(self._timestamps)

        # Compute preview row indices
        if n == 0:
            indices: list[int] = []
            truncated = False
        elif n <= _MAX_PREVIEW * 2 + 1:
            indices = list(range(n))
            truncated = False
        else:
            indices = list(range(_MAX_PREVIEW)) + list(
                range(n - _MAX_PREVIEW, n)
            )
            truncated = True

        data_rows = self._repr_data_rows(indices) if indices else []
        col_names = list(self.column_names)
        show_col_header = len(col_names) > 1

        # Build all content lines (without box chars)
        content_lines: list[str] = []
        # Meta section
        for ml in meta_lines:
            content_lines.append(ml)

        # Data section
        if n == 0:
            content_lines.append(None)  # separator marker
            content_lines.append("(empty)")
        else:
            # Compute column widths for data rows
            all_rows: list[list[str]] = []
            if show_col_header:
                all_rows.append([""] + col_names)
            all_rows.extend(data_rows)

            ncols_data = len(all_rows[0]) if all_rows else 0
            col_widths = [0] * ncols_data
            for row in all_rows:
                for j, cell in enumerate(row):
                    col_widths[j] = max(col_widths[j], len(cell))

            # Also account for ellipsis row
            if truncated:
                for j in range(ncols_data):
                    col_widths[j] = max(col_widths[j], 3)

            def _format_row(row: list[str]) -> str:
                parts: list[str] = []
                for j, cell in enumerate(row):
                    if j == 0:
                        parts.append(f"{cell:<{col_widths[j]}}")
                    else:
                        parts.append(f"{cell:>{col_widths[j]}}")
                return "  ".join(parts)

            content_lines.append(None)  # separator marker

            if show_col_header:
                content_lines.append(_format_row([""] + col_names))

            head_data = data_rows[:_MAX_PREVIEW] if truncated else data_rows
            tail_data = data_rows[_MAX_PREVIEW:] if truncated else []

            for row in head_data:
                content_lines.append(_format_row(row))
            if truncated:
                ellipsis_row = ["..."] * ncols_data
                content_lines.append(_format_row(ellipsis_row))
                for row in tail_data:
                    content_lines.append(_format_row(row))

        # Compute box width
        padding = 2
        max_content_width = max(
            (len(line) for line in content_lines if line is not None), default=0
        )
        box_inner = max_content_width + padding * 2

        # Build output
        lines: list[str] = [class_name]
        top = "\u250c" + "\u2500" * box_inner + "\u2510"
        bot = "\u2514" + "\u2500" * box_inner + "\u2518"
        sep = "\u251c" + "\u2500" * box_inner + "\u2524"
        lines.append(top)
        for line in content_lines:
            if line is None:
                lines.append(sep)
            else:
                lines.append(
                    "\u2502" + " " * padding + line.ljust(max_content_width) + " " * padding + "\u2502"
                )
        lines.append(bot)
        return "\n".join(lines)

    # ---- static helpers --------------------------------------------------

    @staticmethod
    def _from_float_array(arr: np.ndarray) -> list[float | None]:
        """Convert float64 ndarray to list[float|None] — NaN → None."""
        values: list[float | None] = arr.tolist()
        for i in np.where(np.isnan(arr))[0]:
            values[i] = None
        return values

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

    @staticmethod
    def _infer_resolution(df: pd.DataFrame, fallback: Resolution) -> Resolution:
        new_tz = str(df.index.tz) if df.index.tz is not None else fallback.timezone
        freq_str: str | None = None
        if df.index.freq is not None:
            freq_str = df.index.freqstr
        elif len(df.index) >= 3:
            try:
                freq_str = pd.infer_freq(df.index)
            except Exception:
                pass
        new_freq = (
            _PANDAS_FREQ_MAP.get(freq_str, fallback.frequency)
            if freq_str
            else fallback.frequency
        )
        if new_freq == fallback.frequency and new_tz == fallback.timezone:
            return fallback
        return Resolution(new_freq, new_tz)


# ---------------------------------------------------------------------------
# TimeSeries — univariate (single value column)
# ---------------------------------------------------------------------------


@dataclass(slots=True, repr=False)
class TimeSeries(_TimeSeriesBase):
    resolution: Resolution
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
        resolution: Resolution,
        *,
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
        self.resolution = resolution
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
        import pint

        ureg = pint.UnitRegistry()
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
            self.resolution,
            timestamps=self._timestamps[:n],
            values=self._values[:n],
            **self._meta_kwargs(),
        )

    def tail(self, n: int = 5) -> TimeSeries:
        """Return a new TimeSeries with the last *n* points."""
        if n == 0:
            return TimeSeries(
                self.resolution, timestamps=[], values=[], **self._meta_kwargs()
            )
        return TimeSeries(
            self.resolution,
            timestamps=self._timestamps[-n:],
            values=self._values[-n:],
            **self._meta_kwargs(),
        )

    def copy(self) -> TimeSeries:
        """Return a shallow copy (timestamps and values lists are new)."""
        return TimeSeries(
            self.resolution,
            timestamps=list(self._timestamps),
            values=list(self._values),
            **self._meta_kwargs(),
        )

    # ---- scalar arithmetic -----------------------------------------------

    def _to_float_array(self) -> np.ndarray:
        """Convert _values to float64 ndarray — None → NaN."""
        return pd.array(self._values, dtype="Float64").to_numpy(
            dtype=np.float64, na_value=np.nan
        )

    def _apply_scalar(self, func) -> TimeSeries:
        arr = self._to_float_array()
        return TimeSeries(
            self.resolution,
            timestamps=list(self._timestamps),
            values=self._from_float_array(func(arr)),
            **self._meta_kwargs(),
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
        r = self.resolution
        lines.append(f"{'Resolution:':<{label_w}}{r.frequency} ({r.timezone})")
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

        html = [css, '<div class="ts-repr">']
        html.append(
            f'<div class="ts-header">TimeSeries</div>'
        )

        html.append('<div class="ts-meta"><table>')
        html.append(
            f"<tr><td>Columns</td><td>{disp_name}</td></tr>"
        )
        html.append(
            f"<tr><td>Shape</td><td>({n:,},)</td></tr>"
        )
        r = self.resolution
        html.append(
            f"<tr><td>Frequency</td><td>{escape(str(r.frequency))}</td></tr>"
        )
        html.append(f"<tr><td>Timezone</td><td>{escape(r.timezone)}</td></tr>")
        if self.unit:
            html.append(f"<tr><td>Unit</td><td>{escape(self.unit)}</td></tr>")
        if self.data_type:
            html.append(
                f"<tr><td>Data type</td>"
                f"<td>{escape(str(self.data_type))}</td></tr>"
            )
        if self.location:
            html.append(
                f"<tr><td>Location</td>"
                f"<td>{escape(self._fmt_location(self.location))}</td></tr>"
            )
        if self.description:
            html.append(
                f"<tr><td>Description</td>"
                f"<td>{escape(self.description)}</td></tr>"
            )
        html.append("</table></div>")

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
                ts_cells = "".join(
                    f"<td>{escape(str(t))}</td>" for t in ts
                )
            else:
                ts_cells = f"<td>{escape(str(ts))}</td>"
            val_cells = (
                f"<td>{escape(self._fmt_value(self._values[i]))}</td>"
            )
            return f"<tr>{ts_cells}{val_cells}</tr>"

        if n == 0:
            html.append(
                f'<tr><td colspan="{n_cols}" class="ts-ellipsis">'
                f"(empty)</td></tr>"
            )
        else:
            show_all = n <= _MAX_PREVIEW * 2 + 1
            head_rows = range(min(_MAX_PREVIEW, n))
            tail_rows = (
                range(max(n - _MAX_PREVIEW, _MAX_PREVIEW), n)
                if not show_all
                else range(0)
            )

            for i in head_rows:
                html.append(_html_row(i))

            if not show_all:
                ellipsis_cells = "".join(
                    f'<td class="ts-ellipsis">&hellip;</td>'
                    for _ in range(n_cols)
                )
                html.append(f"<tr>{ellipsis_cells}</tr>")
                for i in tail_rows:
                    html.append(_html_row(i))

        html.append("</table></div>")
        html.append("</div>")
        return "\n".join(html)

    # ---- numpy / pandas / polars -----------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Return values as a 1D numpy float64 array (None -> nan)."""
        return self._to_float_array()

    def to_pandas_dataframe(self) -> pd.DataFrame:
        """Return a pandas DataFrame with DatetimeIndex or MultiIndex."""
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

        data: dict = {}
        if self.is_multi_index:
            for i, iname in enumerate(self.index_names):
                data[iname] = [t[i] for t in self._timestamps]
        else:
            data[self.index_names[0]] = self._timestamps

        col_name = self.name or "value"
        data[col_name] = self._to_float_array()
        return pl.DataFrame(data)

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        resolution: Resolution | None = None,
        value_column: str | None = None,
        *,
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
        If *resolution* is ``None``, frequency and timezone are auto-inferred.
        """
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

        if resolution is None:
            if isinstance(df.index, pd.DatetimeIndex):
                resolution = cls._infer_resolution(
                    df, Resolution(Frequency.NONE)
                )
            else:
                resolution = Resolution(Frequency.NONE)

        return cls(
            resolution,
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
        resolution: Resolution,
        timestamp_column: str = "timestamp",
        value_column: str | None = None,
        *,
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
            resolution,
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
        df: pd.DataFrame,
        value_column: str | None = None,
        inplace: bool = False,
    ) -> TimeSeries | None:
        """Update a TimeSeries from a pandas DataFrame.

        By default returns a **new** TimeSeries.  With ``inplace=True``
        the current instance is mutated and ``None`` is returned.
        """
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

        new_resolution = (
            self._infer_resolution(df, self.resolution)
            if isinstance(df.index, pd.DatetimeIndex)
            else self.resolution
        )
        original_col = self.name or "value"
        new_name = str(col) if str(col) != original_col else self.name

        if inplace:
            self._timestamps = timestamps
            self._values = values
            self._index_names = index_names
            self.resolution = new_resolution
            self.name = new_name
            return None
        return TimeSeries(
            new_resolution,
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

        payload: dict = {"timestamps": ts_json, "values": val_json}
        if self._index_names is not None:
            payload["index_names"] = self._index_names
        return json.dumps(payload)

    @classmethod
    def from_json(
        cls,
        s: str,
        resolution: Resolution,
        *,
        name: str | None = None,
        unit: str | None = None,
        description: str | None = None,
        data_type: DataType | None = None,
        location: Location | None = None,
        timeseries_type: TimeSeriesType = TimeSeriesType.FLAT,
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

        return cls(
            resolution,
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
        resolution: Resolution,
        *,
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
            resolution,
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
        warnings = super().validate()
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
        func: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> TimeSeries:
        """Apply a pandas transformation, preserving metadata and auto-detecting resolution."""
        df = self.to_pandas_dataframe()
        result = func(df)
        new_resolution = self._infer_resolution(result, self.resolution)

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
            new_resolution,
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
            timestamps=list(self._timestamps),
            values=self._from_float_array(result),
            **self._meta_kwargs(),
        )

    @staticmethod
    def merge(series: list[TimeSeries]) -> MultivariateTimeSeries:
        """Combine multiple univariate TimeSeries into a MultivariateTimeSeries.

        All series must share the same timestamps (same length and values).

        Args:
            series: List of univariate TimeSeries to merge.

        Returns:
            A new MultivariateTimeSeries with one column per input series.

        Raises:
            ValueError: If the list is empty or timestamps don't match.
        """
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

        return MultivariateTimeSeries(
            series[0].resolution,
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


# ---------------------------------------------------------------------------
# MultivariateTimeSeries — multiple value columns
# ---------------------------------------------------------------------------


@dataclass(slots=True, repr=False)
class MultivariateTimeSeries(_TimeSeriesBase):
    resolution: Resolution
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
        resolution: Resolution,
        *,
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
        self.resolution = resolution
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
    ) -> MultivariateTimeSeries:
        return MultivariateTimeSeries(
            self.resolution,
            timestamps=timestamps,
            values=values,
            **self._list_meta_kwargs(),
        )

    # ---- column extraction ------------------------------------------------

    def select_column(self, col: int | str) -> TimeSeries:
        """Extract a single column as a univariate TimeSeries.

        Args:
            col: Column index (int) or column name (str).

        Returns:
            A new TimeSeries with that column's values and resolved metadata.
        """
        if isinstance(col, str):
            names = self.column_names
            if col not in names:
                raise KeyError(f"Column '{col}' not found. Available: {names}")
            col = names.index(col)

        arr = self._values[:, col]
        values = self._from_float_array(arr)

        return TimeSeries(
            self.resolution,
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

    def to_univariate_list(self) -> list[TimeSeries]:
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

    def head(self, n: int = 5) -> MultivariateTimeSeries:
        """Return a new MultivariateTimeSeries with the first *n* points."""
        return self._clone_with(self._timestamps[:n], self._values[:n])

    def tail(self, n: int = 5) -> MultivariateTimeSeries:
        """Return a new MultivariateTimeSeries with the last *n* points."""
        if n == 0:
            return self._clone_with([], self._values[:0])
        return self._clone_with(self._timestamps[-n:], self._values[-n:])

    def copy(self) -> MultivariateTimeSeries:
        """Return a shallow copy (timestamps list and values array are new)."""
        return self._clone_with(
            list(self._timestamps), self._values.copy()
        )

    # ---- scalar arithmetic -----------------------------------------------

    def _apply_scalar(self, func) -> MultivariateTimeSeries:
        arr = self._values.astype(np.float64, copy=True)
        return self._clone_with(list(self._timestamps), func(arr))

    def __add__(self, scalar: float) -> MultivariateTimeSeries:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: v + scalar)

    def __radd__(self, scalar: float) -> MultivariateTimeSeries:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: scalar + v)

    def __sub__(self, scalar: float) -> MultivariateTimeSeries:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: v - scalar)

    def __rsub__(self, scalar: float) -> MultivariateTimeSeries:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: scalar - v)

    def __mul__(self, scalar: float) -> MultivariateTimeSeries:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: v * scalar)

    def __rmul__(self, scalar: float) -> MultivariateTimeSeries:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: scalar * v)

    def __truediv__(self, scalar: float) -> MultivariateTimeSeries:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self._apply_scalar(lambda v: v / scalar)

    def __neg__(self) -> MultivariateTimeSeries:
        return self._apply_scalar(lambda v: -v)

    def __abs__(self) -> MultivariateTimeSeries:
        return self._apply_scalar(abs)

    def __round__(self, n: int = 0) -> MultivariateTimeSeries:
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
        r = self.resolution
        lines.append(f"{'Resolution:':<{label_w}}{r.frequency} ({r.timezone})")

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

        label = ", ".join(escape(c) for c in cn)
        html = [css, '<div class="ts-repr">']
        html.append(
            f'<div class="ts-header">MultivariateTimeSeries</div>'
        )

        html.append('<div class="ts-meta"><table>')
        html.append(
            f"<tr><td>Columns</td><td>{label}</td></tr>"
        )
        html.append(
            f"<tr><td>Shape</td><td>({n:,}, {ncols})</td></tr>"
        )
        r = self.resolution
        html.append(
            f"<tr><td>Frequency</td>"
            f"<td>{escape(str(r.frequency))}</td></tr>"
        )
        html.append(
            f"<tr><td>Timezone</td><td>{escape(r.timezone)}</td></tr>"
        )
        html.append("</table></div>")

        idx_names = self.index_names
        total_cols = len(idx_names) + ncols
        html.append('<div class="ts-data"><table>')
        header_cells = "".join(
            f"<th>{escape(c)}</th>" for c in idx_names
        )
        header_cells += "".join(f"<th>{escape(c)}</th>" for c in cn)
        html.append(f"<tr>{header_cells}</tr>")

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

        if n == 0:
            html.append(
                f'<tr><td colspan="{total_cols}" class="ts-ellipsis">'
                f"(empty)</td></tr>"
            )
        else:
            show_all = n <= _MAX_PREVIEW * 2 + 1
            head_rows = range(min(_MAX_PREVIEW, n))
            tail_rows = (
                range(max(n - _MAX_PREVIEW, _MAX_PREVIEW), n)
                if not show_all
                else range(0)
            )

            for i in head_rows:
                html.append(_html_row(i))

            if not show_all:
                ellipsis_cells = "".join(
                    f'<td class="ts-ellipsis">&hellip;</td>'
                    for _ in range(total_cols)
                )
                html.append(f"<tr>{ellipsis_cells}</tr>")
                for i in tail_rows:
                    html.append(_html_row(i))

        html.append("</table></div>")
        html.append("</div>")
        return "\n".join(html)

    # ---- numpy / pandas / polars -----------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Return values as a 2D numpy float64 array."""
        return self._values.astype(np.float64, copy=True)

    def to_pandas_dataframe(self) -> pd.DataFrame:
        """Return a pandas DataFrame with multiple value columns."""
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
        df: pd.DataFrame,
        resolution: Resolution | None = None,
        *,
        names: list[str | None] | None = None,
        units: list[str | None] | None = None,
        descriptions: list[str | None] | None = None,
        data_types: list[DataType | None] | None = None,
        locations: list[Location | None] | None = None,
        timeseries_types: list[TimeSeriesType] | None = None,
        attributes: list[dict[str, str]] | None = None,
    ) -> MultivariateTimeSeries:
        """Create a MultivariateTimeSeries from a pandas DataFrame.

        All numeric columns become value columns.  Column names become
        ``names`` if not explicitly provided.
        """
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

        if resolution is None:
            if isinstance(df.index, pd.DatetimeIndex):
                resolution = cls._infer_resolution(
                    df, Resolution(Frequency.NONE)
                )
            else:
                resolution = Resolution(Frequency.NONE)

        return cls(
            resolution,
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
        }
        if self._index_names is not None:
            payload["index_names"] = self._index_names
        return json.dumps(payload)

    @classmethod
    def from_json(
        cls,
        s: str,
        resolution: Resolution,
        *,
        names: list[str | None] | None = None,
        units: list[str | None] | None = None,
        descriptions: list[str | None] | None = None,
        data_types: list[DataType | None] | None = None,
        locations: list[Location | None] | None = None,
        timeseries_types: list[TimeSeriesType] | None = None,
        attributes: list[dict[str, str]] | None = None,
    ) -> MultivariateTimeSeries:
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

        if names is None:
            cn = data.get("column_names")
            if cn:
                names = cn

        return cls(
            resolution,
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
        resolution: Resolution,
        *,
        names: list[str | None] | None = None,
        units: list[str | None] | None = None,
        descriptions: list[str | None] | None = None,
        data_types: list[DataType | None] | None = None,
        locations: list[Location | None] | None = None,
        timeseries_types: list[TimeSeriesType] | None = None,
        attributes: list[dict[str, str]] | None = None,
    ) -> MultivariateTimeSeries:
        """Read a MultivariateTimeSeries from a CSV file produced by to_csv()."""
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
            resolution,
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
        func: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> MultivariateTimeSeries:
        """Apply a pandas transformation, preserving metadata and auto-detecting resolution."""
        df = self.to_pandas_dataframe()
        result = func(df)
        new_resolution = self._infer_resolution(result, self.resolution)

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

        return MultivariateTimeSeries(
            new_resolution,
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
    ) -> MultivariateTimeSeries:
        """Apply a numpy transformation to values, keeping timestamps and resolution unchanged."""
        arr = self.to_numpy()
        result = np.asarray(func(arr), dtype=np.float64)
        if result.shape[0] != len(self._timestamps):
            raise ValueError(
                f"apply_numpy: result length ({result.shape[0]}) must match "
                f"series length ({len(self._timestamps)})"
            )
        return self._clone_with(list(self._timestamps), result)


MultiTimeSeries = MultivariateTimeSeries
