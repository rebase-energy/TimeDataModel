"""Private helpers for TimeSeries — display, Arrow, pandas, polars, unit conversion."""
from __future__ import annotations

import functools
from datetime import datetime, timezone as _dt_timezone
from html import escape
from typing import Callable

import numpy as np
import pyarrow as pa

from .enums import Frequency
from .location import GeoArea, GeoLocation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_PREVIEW = 3

_PANDAS_FREQ_MAP: dict[str, Frequency] = {
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

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _fmt_short_date(dt: datetime) -> str:
    """Format a datetime concisely: omit time when midnight, omit tzinfo."""
    dt_naive = dt.replace(tzinfo=None)
    if dt_naive.hour == 0 and dt_naive.minute == 0 and dt_naive.second == 0:
        return dt_naive.strftime("%Y-%m-%d")
    return dt_naive.strftime("%Y-%m-%d %H:%M")


def _fmt_value(v: float | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NaN"
    if v == int(v):
        return f"{v:.1f}"
    return f"{v:g}"


def _fmt_location(loc: GeoLocation | GeoArea | None) -> str:
    if loc is None:
        return ""
    if isinstance(loc, GeoLocation):
        return f"{loc.latitude}\u00b0N, {loc.longitude}\u00b0E"
    name = loc.name or "unnamed"
    c = loc.centroid
    return f"{name} (centroid {c.latitude}\u00b0N, {c.longitude}\u00b0E)"


def _render_box(class_name: str, content_lines: list[str | None], padding: int = 2) -> str:
    """Render a Unicode box around content lines.

    ``None`` entries in *content_lines* are drawn as horizontal separators.
    """
    max_w = max((len(l) for l in content_lines if l is not None), default=0)
    box_inner = max_w + padding * 2
    top = "\u250c" + "\u2500" * box_inner + "\u2510"
    bot = "\u2514" + "\u2500" * box_inner + "\u2518"
    sep = "\u251c" + "\u2500" * box_inner + "\u2524"
    lines = [class_name, top]
    for cl in content_lines:
        if cl is None:
            lines.append(sep)
        else:
            lines.append(
                "\u2502" + " " * padding + cl.ljust(max_w) + " " * padding + "\u2502"
            )
    lines.append(bot)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML repr
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
.ts-repr .ts-data td.ts-idx { text-align: left; color: #333; }
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
    """Build a shared HTML repr for TimeSeries."""
    total_cols = len(index_names) + len(column_names)

    html = [_REPR_CSS, '<div class="ts-repr">']
    html.append(f'<div class="ts-header">{escape(class_name)}</div>')

    html.append('<div class="ts-meta"><table>')
    for label, value in meta_rows:
        html.append(f"<tr><td>{escape(label)}</td><td>{value}</td></tr>")
    html.append("</table></div>")

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
# Lazy imports
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _get_pint_registry():
    import pint
    return pint.UnitRegistry()


def _convert_unit_values(values: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    """Convert *values* from *from_unit* to *to_unit* using pint."""
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
# xarray helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Pandas helpers
# ---------------------------------------------------------------------------


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


def _extract_pandas_index(df) -> tuple[list, list | None]:
    """Extract timestamps and index_names from a pandas DataFrame."""
    pd = _import_pandas()
    if isinstance(df.index, pd.MultiIndex):
        timestamps = [
            tuple(
                lvl.to_pydatetime() if hasattr(lvl, "to_pydatetime") else lvl
                for lvl in row
            )
            for row in df.index
        ]
        return timestamps, list(df.index.names)
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.to_pydatetime().tolist(), None
    raise ValueError("DataFrame must have a DatetimeIndex or MultiIndex")


def _from_float_array(arr: np.ndarray) -> list[float | None]:
    """Convert float64 ndarray to list[float|None] — NaN → None."""
    values: list[float | None] = arr.tolist()
    for i in np.where(np.isnan(arr))[0]:
        values[i] = None
    return values


# ---------------------------------------------------------------------------
# Arrow helpers
# ---------------------------------------------------------------------------


def _ensure_utc(dt: datetime) -> datetime:
    """Return timezone-aware UTC datetime; naive datetimes are assumed UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_dt_timezone.utc)
    return dt


def _build_arrow_table(
    timestamps: list,
    values: list[float | None],
    index_names: list[str] | None = None,
) -> pa.Table:
    """Convert Python lists to Arrow Table with schema [time_cols..., value].

    The first (or only) time column is named "valid_time" by default.
    For multi-index, additional columns are named after *index_names* or
    auto-generated as "valid_time", "index_1", "index_2", ...
    """
    if not timestamps:
        default_ts_name = (index_names[0] if index_names else None) or "valid_time"
        schema = pa.schema([
            pa.field(default_ts_name, pa.timestamp("us", tz="UTC")),
            pa.field("value", pa.float64()),
        ])
        return pa.table(
            {default_ts_name: pa.array([], type=pa.timestamp("us", tz="UTC")),
             "value": pa.array([], type=pa.float64())},
            schema=schema,
        )

    is_multi = isinstance(timestamps[0], tuple)
    arrays: dict[str, pa.Array] = {}

    if is_multi:
        n_dims = len(timestamps[0])
        if index_names and len(index_names) == n_dims:
            names = list(index_names)
        else:
            names = ["valid_time"] + [f"index_{i}" for i in range(1, n_dims)]

        for i, col_name in enumerate(names):
            col_vals = [_ensure_utc(t[i]) for t in timestamps]
            arrays[col_name] = pa.array(col_vals, type=pa.timestamp("us", tz="UTC"))
    else:
        col_name = (index_names[0] if index_names else None) or "valid_time"
        arrays[col_name] = pa.array(
            [_ensure_utc(t) for t in timestamps],
            type=pa.timestamp("us", tz="UTC"),
        )

    arrays["value"] = pa.array(
        [float(v) if v is not None else None for v in values],
        type=pa.float64(),
    )
    return pa.table(arrays)


def _arrow_table_to_timestamps(table: pa.Table) -> list:
    """Extract timestamps from Arrow Table as Python datetimes or tuples."""
    ts_cols = [n for n in table.schema.names if n != "value"]
    if len(ts_cols) == 1:
        return table.column(ts_cols[0]).to_pylist()
    cols = [table.column(n).to_pylist() for n in ts_cols]
    return [tuple(row) for row in zip(*cols)]
