from __future__ import annotations

import bisect
import functools
from datetime import datetime, timedelta
from html import escape
from typing import Callable

import numpy as np

from ._theme import THEME, get_theme_version
from .enums import DataType, Frequency, TimeSeriesType
from .location import GeoArea, GeoLocation, Location

_default_dataframe_backend: str = "pandas"
_default_repr_width: int | None = None  # None = no limit


def set_repr_width(width: int | None) -> None:
    """Set max repr box width. None = no limit."""
    global _default_repr_width
    if width is not None and width < 10:
        raise ValueError("repr width must be at least 10 or None")
    _default_repr_width = width


def get_repr_width() -> int | None:
    """Return current max repr width (None = no limit)."""
    return _default_repr_width


def set_default_df(backend: str) -> None:
    """Set the default DataFrame backend for the ``.df`` property.

    Parameters
    ----------
    backend : str
        ``"pandas"`` or ``"polars"``.
    """
    global _default_dataframe_backend
    if backend not in ("pandas", "polars"):
        raise ValueError(
            f"backend must be 'pandas' or 'polars', got {backend!r}"
        )
    _default_dataframe_backend = backend


def get_default_df() -> str:
    """Return the current default DataFrame backend (``"pandas"`` or ``"polars"``)."""
    return _default_dataframe_backend


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
_MAX_COL_PREVIEW = 4  # leaf columns shown at head/tail in array repr


def _truncate(s: str, max_len: int) -> str:
    """Return s[:max_len-3] + '...' if too long, else s."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _fmt_short_date(dt: datetime) -> str:
    """Format a datetime concisely: always include time, omit tzinfo."""
    dt_naive = dt.replace(tzinfo=None)
    return dt_naive.strftime("%Y-%m-%d %H:%M")


def _fmt_timestamp(ts: datetime | tuple[datetime, ...]) -> str:
    """Format a single or multi-index timestamp for terminal repr."""
    if isinstance(ts, tuple):
        return ", ".join(_fmt_short_date(t) for t in ts)
    return _fmt_short_date(ts)


def _fmt_timestamp_cells(ts: datetime | tuple[datetime, ...]) -> str:
    """Format a single or multi-index timestamp as HTML ``<td>`` cells."""
    if isinstance(ts, tuple):
        return "".join(f"<td>{escape(_fmt_short_date(t))}</td>" for t in ts)
    return f"<td>{escape(_fmt_short_date(ts))}</td>"


def _format_meta_lines(pairs: list[tuple[str, str]], label_w: int = 18) -> list[str]:
    """Convert (label, value) pairs to formatted terminal meta lines."""
    return [f"{label + ':':<{label_w}}{value}" for label, value in pairs]


def _fmt_tz_with_offset(tz_str: str, timestamps: list) -> str:
    """Return e.g. ``'UTC (+00:00)'`` when the first timestamp is tz-aware."""
    if timestamps:
        first = timestamps[0]
        if isinstance(first, tuple):
            first = first[0]
        if hasattr(first, 'utcoffset') and first.utcoffset() is not None:
            offset = first.utcoffset()
            total_seconds = int(offset.total_seconds())
            sign = '+' if total_seconds >= 0 else '-'
            hours, remainder = divmod(abs(total_seconds), 3600)
            minutes = remainder // 60
            return f"{tz_str} ({sign}{hours:02d}:{minutes:02d})"
    return tz_str


def _import_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError(
            "pandas is required for this operation. "
            "Install it with: pip install timedatamodel[pandas]"
        ) from None


def _import_polars():
    try:
        import polars as pl
        return pl
    except ImportError:
        raise ImportError(
            "polars is required for this operation. "
            "Install it with: pip install timedatamodel[polars]"
        ) from None


def _extract_timestamps_from_pandas_index(df):
    """Extract timestamps and index names from a pandas DataFrame index."""
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
    return timestamps, index_names


@functools.lru_cache(maxsize=1)
def _get_pint_registry():
    import pint
    return pint.UnitRegistry()


def _convert_unit_values(values: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    """Convert *values* from *from_unit* to *to_unit* using pint.

    Short-circuits when units are identical.  Wraps pint dimensionality
    errors into a plain ``ValueError`` and ``ImportError`` with install hints.
    """
    if from_unit == to_unit:
        return values
    try:
        import pint
    except ImportError:
        raise ImportError(
            "pint is required for unit conversion. "
            "Install it with: pip install timedatamodel[pint]"
        ) from None
    ureg = _get_pint_registry()
    try:
        factor = ureg.Quantity(1, from_unit).to(to_unit).magnitude
    except pint.errors.DimensionalityError:
        raise ValueError(
            f"cannot convert '{from_unit}' to '{to_unit}': incompatible dimensions"
        ) from None
    return values * factor


# ---------------------------------------------------------------------------
# Shared _repr_html_ builder
# ---------------------------------------------------------------------------

def _repr_css() -> str:
    lt = THEME["light"]
    dk = THEME["dark"]
    return f"""\
<style>
.ts-repr {{ font-family: monospace; font-size: 13px; max-width: 640px; display: inline-grid; }}
.ts-repr .ts-header {{
  font-weight: bold; font-size: 14px;
  padding: 6px 10px; border-bottom: 2px solid {lt["header_border"]};
  background: {lt["header_bg"]}; color: {lt["header_text"]};
}}
.ts-repr .ts-meta {{ padding: 6px 10px; background: {lt["meta_bg"]}; overflow: hidden; min-width: 0; }}
.ts-repr .ts-meta table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
.ts-repr .ts-meta td {{ padding: 1px 8px 1px 0; white-space: nowrap; }}
.ts-repr .ts-meta td:first-child {{ color: {lt["meta_label"]}; font-weight: 600; width: 90px; }}
.ts-repr .ts-meta td:last-child {{ color: {lt["meta_value"]}; overflow: hidden; text-overflow: ellipsis; }}
.ts-repr .ts-data {{ padding: 6px 10px; }}
.ts-repr .ts-data table {{
  border-collapse: collapse; text-align: right;
}}
.ts-repr .ts-data th {{
  text-align: right; padding: 3px 10px; border-bottom: 1px solid {lt["col_header_border"]};
  color: {lt["col_header_text"]}; font-weight: 600;
}}
.ts-repr .ts-data th.ts-idx {{ text-align: left; }}
.ts-repr .ts-data td {{ padding: 2px 10px; }}
.ts-repr .ts-data tr:hover {{ background: {lt["hover_bg"]}; }}
.ts-repr .ts-data td:first-child {{ text-align: left; color: {lt["index_text"]}; }}
.ts-repr .ts-data td.ts-idx {{ text-align: left; color: {lt["index_text"]}; }}
.ts-repr .ts-ellipsis {{ text-align: center !important; color: {lt["ellipsis"]}; }}
@media (prefers-color-scheme: dark) {{
  .ts-repr .ts-header {{ background: {dk["header_bg"]}; color: {dk["header_text"]}; border-color: {dk["header_border"]}; }}
  .ts-repr .ts-meta {{ background: {dk["meta_bg"]}; }}
  .ts-repr .ts-meta td:first-child {{ color: {dk["meta_label"]}; }}
  .ts-repr .ts-meta td:last-child {{ color: {dk["meta_value"]}; }}
  .ts-repr .ts-data th {{ color: {dk["col_header_text"]}; border-color: {dk["col_header_border"]}; }}
  .ts-repr .ts-data td {{ color: {dk["data_text"]}; }}
  .ts-repr .ts-data td:first-child {{ color: {dk["index_text"]}; }}
  .ts-repr .ts-data td.ts-idx {{ color: {dk["index_text"]}; }}
  .ts-repr .ts-data tr:hover {{ background: {dk["hover_bg"]}; }}
  .ts-repr .ts-ellipsis {{ color: {dk["ellipsis"]}; }}
}}
</style>"""

_css_cache: str | None = None
_css_cache_version: int = -1


def _get_repr_css() -> str:
    """Return the CSS string, regenerating only when the theme version changes."""
    global _css_cache, _css_cache_version
    version = get_theme_version()
    if _css_cache is not None and _css_cache_version == version:
        return _css_cache
    _css_cache = _repr_css()
    _css_cache_version = version
    return _css_cache


def _build_repr_html(
    class_name: str,
    meta_rows: list[tuple[str, str]],
    index_names: tuple[str, ...],
    column_names: tuple[str, ...],
    n_rows: int,
    html_row_fn: Callable[[int], str],
    max_preview: int = _MAX_PREVIEW,
) -> str:
    """Build a shared HTML repr for TimeSeriesList and TimeSeriesTable."""
    total_cols = len(index_names) + len(column_names)

    html = [_get_repr_css(), '<div class="ts-repr">']
    html.append(f'<div class="ts-header">{escape(class_name)}</div>')

    # Meta section
    html.append('<div class="ts-meta"><table>')
    for label, value in meta_rows:
        html.append(f"<tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>")
    html.append("</table></div>")

    # Data section
    html.append('<div class="ts-data"><table>')
    header_cells = "".join(f'<th class="ts-idx">{escape(c)}</th>' for c in index_names)
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

    def _repr_meta_lines(self) -> list[str]:
        """Build terminal meta lines from ``_repr_meta_pairs()``."""
        return _format_meta_lines(self._repr_meta_pairs())

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

        return _render_box(class_name, content_lines)

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


def _xarray_labels_to_list(raw) -> list:
    """Convert xarray coord values (numpy) to Python datetime/float/str."""
    import pandas as pd
    out = []
    for v in raw:
        if isinstance(v, (np.datetime64, pd.Timestamp)):
            out.append(pd.Timestamp(v).to_pydatetime())
        elif isinstance(v, (np.floating, float)):
            out.append(float(v))
        elif isinstance(v, np.integer):
            out.append(float(v))
        else:
            out.append(str(v))
    return out


def _render_box(
    class_name: str,
    content_lines: list[str | None],
    padding: int = 2,
    max_width: int | None = None,
) -> str:
    """Render a Unicode box around content lines.

    ``None`` entries in *content_lines* are drawn as horizontal separators.
    *max_width* caps the total box width (border + padding + content).
    Defaults to the global ``get_repr_width()`` setting when ``None``.
    """
    if max_width is None:
        max_width = get_repr_width()

    max_w = max((len(l) for l in content_lines if l is not None), default=0)

    # Cap content width when a max_width is active
    # Total width = 2 (borders) + 2*padding + content
    if max_width is not None:
        max_content = max_width - 2 * padding - 2
        if max_content < 1:
            max_content = 1
        max_w = min(max_w, max_content)

    box_inner = max_w + padding * 2
    top = "\u250c" + "\u2500" * box_inner + "\u2510"
    bot = "\u2514" + "\u2500" * box_inner + "\u2518"
    sep = "\u251c" + "\u2500" * box_inner + "\u2524"
    lines = [class_name, top]
    for cl in content_lines:
        if cl is None:
            lines.append(sep)
        else:
            if max_width is not None and len(cl) > max_w:
                cl = _truncate(cl, max_w)
            lines.append(
                "\u2502" + " " * padding + cl.ljust(max_w) + " " * padding + "\u2502"
            )
    lines.append(bot)
    return "\n".join(lines)


class _DataFrameMixin:
    """Mixin providing the ``df`` shorthand property."""
    __slots__ = ()

    @property
    def df(self):
        """Shorthand for the default DataFrame backend (pandas or polars)."""
        if _default_dataframe_backend == "polars":
            return self.to_polars_dataframe()
        return self.to_pandas_dataframe()


def _validate_timestamp_sequence(
    timestamps: list[datetime] | list[tuple[datetime, ...]],
) -> None:
    """Validate timestamp container shape and element types."""
    if not timestamps:
        return

    first = timestamps[0]

    if isinstance(first, tuple):
        tuple_len = len(first)
        if tuple_len == 0:
            raise ValueError("multi-index timestamp tuples must not be empty")

        for i, ts in enumerate(timestamps):
            if not isinstance(ts, tuple):
                raise TypeError(
                    "timestamps must be homogeneous: all datetime values "
                    "or all tuples of datetime values"
                )
            if len(ts) != tuple_len:
                raise ValueError(
                    f"inconsistent multi-index tuple length at index {i}: "
                    f"expected {tuple_len}, got {len(ts)}"
                )
            for j, dt in enumerate(ts):
                if not isinstance(dt, datetime):
                    raise TypeError(
                        f"timestamp tuple element at index {i}[{j}] must be datetime, "
                        f"got {type(dt).__name__}"
                    )
        return

    if not isinstance(first, datetime):
        raise TypeError(
            f"timestamp at index 0 must be datetime, got {type(first).__name__}"
        )

    for i, ts in enumerate(timestamps):
        if isinstance(ts, tuple):
            raise TypeError(
                "timestamps must be homogeneous: all datetime values "
                "or all tuples of datetime values"
            )
        if not isinstance(ts, datetime):
            raise TypeError(
                f"timestamp at index {i} must be datetime, got {type(ts).__name__}"
            )
