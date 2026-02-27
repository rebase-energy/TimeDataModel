from __future__ import annotations

import bisect
import functools
from datetime import datetime, timedelta
from html import escape
from typing import Callable

import numpy as np

from .enums import DataType, Frequency, TimeSeriesType
from .location import GeoArea, GeoLocation, Location

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


def _fmt_short_date(dt: datetime) -> str:
    """Format a datetime concisely: omit time when midnight, omit tzinfo."""
    dt_naive = dt.replace(tzinfo=None)
    if dt_naive.hour == 0 and dt_naive.minute == 0 and dt_naive.second == 0:
        return dt_naive.strftime("%Y-%m-%d")
    return dt_naive.strftime("%Y-%m-%d %H:%M")


def _import_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError(
            "pandas is required for this operation. "
            "Install it with: pip install timedatamodel[pandas]"
        ) from None


@functools.lru_cache(maxsize=1)
def _get_pint_registry():
    import pint
    return pint.UnitRegistry()


# ---------------------------------------------------------------------------
# Shared _repr_html_ builder
# ---------------------------------------------------------------------------

_REPR_CSS = """\
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


def _build_repr_html(
    class_name: str,
    meta_rows: list[tuple[str, str]],
    index_names: tuple[str, ...],
    column_names: tuple[str, ...],
    n_rows: int,
    html_row_fn: Callable[[int], str],
    max_preview: int = _MAX_PREVIEW,
) -> str:
    """Build a shared HTML repr for TimeSeries and TimeSeriesTable."""
    total_cols = len(index_names) + len(column_names)

    html = [_REPR_CSS, '<div class="ts-repr">']
    html.append(f'<div class="ts-header">{escape(class_name)}</div>')

    # Meta section
    html.append('<div class="ts-meta"><table>')
    for label, value in meta_rows:
        html.append(f"<tr><td>{escape(label)}</td><td>{value}</td></tr>")
    html.append("</table></div>")

    # Data section
    html.append('<div class="ts-data"><table>')
    header_cells = "".join(f"<th>{escape(c)}</th>" for c in index_names)
    header_cells += "".join(f"<th>{escape(c)}</th>" for c in column_names)
    html.append(f"<tr>{header_cells}</tr>")

    if n_rows == 0:
        html.append(
            f'<tr><td colspan="{total_cols}" class="ts-ellipsis">'
            f"(empty)</td></tr>"
        )
    else:
        show_all = n_rows <= max_preview * 2 + 1
        head_rows = range(min(max_preview, n_rows))
        tail_rows = (
            range(max(n_rows - max_preview, max_preview), n_rows)
            if not show_all
            else range(0)
        )

        for i in head_rows:
            html.append(html_row_fn(i))

        if not show_all:
            ellipsis_cells = "".join(
                f'<td class="ts-ellipsis">&hellip;</td>'
                for _ in range(total_cols)
            )
            html.append(f"<tr>{ellipsis_cells}</tr>")
            for i in tail_rows:
                html.append(html_row_fn(i))

    html.append("</table></div>")
    html.append("</div>")
    return "\n".join(html)


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
        td = self.frequency.to_timedelta()
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
    def _infer_freq_tz(
        df: "pd.DataFrame", fallback_freq: Frequency, fallback_tz: str,
    ) -> tuple[Frequency, str]:
        pd = _import_pandas()
        new_tz = str(df.index.tz) if df.index.tz is not None else fallback_tz
        freq_str: str | None = None
        if df.index.freq is not None:
            freq_str = df.index.freqstr
        elif len(df.index) >= 3:
            try:
                freq_str = pd.infer_freq(df.index)
            except Exception:
                pass
        new_freq = (
            _PANDAS_FREQ_MAP.get(freq_str, fallback_freq)
            if freq_str
            else fallback_freq
        )
        return (new_freq, new_tz)
