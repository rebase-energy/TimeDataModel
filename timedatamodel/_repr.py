"""Centralized display / repr logic for timedatamodel data classes.

This module contains:
- Module-level formatting helpers
- CSS infrastructure for HTML reprs
- Mixin classes providing __repr__ / _repr_html_ for each data class
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime
from html import escape

from ._theme import THEME, get_theme_version

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

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


_MAX_PREVIEW = 3  # rows shown at head/tail in repr
_MAX_COL_PREVIEW = 4  # leaf columns shown at head/tail in array repr

# ---------------------------------------------------------------------------
# Pure formatting helpers
# ---------------------------------------------------------------------------


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
        if hasattr(first, "utcoffset") and first.utcoffset() is not None:
            offset = first.utcoffset()
            total_seconds = int(offset.total_seconds())
            sign = "+" if total_seconds >= 0 else "-"
            hours, remainder = divmod(abs(total_seconds), 3600)
            minutes = remainder // 60
            return f"{tz_str} ({sign}{hours:02d}:{minutes:02d})"
    return tz_str


def _fmt_value(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "NaN"
    if v == int(v):
        return f"{v:.1f}"
    return f"{v:g}"


# ---------------------------------------------------------------------------
# CoverageBar
# ---------------------------------------------------------------------------


class CoverageBar:
    """Displayable coverage bar showing value presence over time.

    Each ``(name, mask)`` pair in *masks* produces one row; ``True`` = value
    present, ``False`` = null/missing.  *begin* and *end* are Python
    ``datetime`` objects used for the date axis labels.

    Produced by :meth:`TimeSeries.coverage_bar`.
    """

    _TERM_BINS = 60
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
        for name, mask in self._masks:
            binned = self._bin_coverage(mask, n_bins)
            bar = "".join("█" if b else "░" for b in binned)
            lines.append(f"{name:<{label_w}}{bar}")

        bar_len = len(self._bin_coverage(self._masks[0][1], n_bins))
        start_str = _fmt_short_date(self._begin) if self._begin else ""
        end_str = _fmt_short_date(self._end) if self._end else ""
        gap = bar_len - len(start_str) - len(end_str)
        if gap < 2:
            gap = 2
        date_line = start_str + " " * gap + end_str
        lines.append(f"{'':<{label_w}}{date_line}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        if not self._masks:
            return ""
        n_bins = self._SVG_BINS
        label_w = 120
        bar_w = 480
        row_h = 22
        n_rows = len(self._masks)
        date_h = 18
        total_h = n_rows * row_h + date_h + 4

        lt = THEME["light"]
        parts: list[str] = []
        parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {label_w + bar_w} {total_h}" '
            f'width="100%" style="max-width:{label_w + bar_w}px;'
            f'font-family:monospace;font-size:12px;">'
        )

        for row_idx, (name, mask) in enumerate(self._masks):
            y = row_idx * row_h
            parts.append(
                f'<text x="{label_w - 6}" y="{y + 15}" '
                f'text-anchor="end" fill="{lt["coverage_label"]}">{escape(name)}</text>'
            )
            binned = self._bin_coverage(mask, n_bins)
            seg_w = bar_w / len(binned) if binned else bar_w
            for i, b in enumerate(binned):
                color = lt["coverage_present"] if b else lt["coverage_absent"]
                x = label_w + i * seg_w
                parts.append(
                    f'<rect x="{x:.1f}" y="{y + 2}" width="{seg_w:.2f}" height="{row_h - 4}" fill="{color}" />'
                )

        date_y = n_rows * row_h + date_h
        if self._begin:
            parts.append(
                f'<text x="{label_w}" y="{date_y}" '
                f'text-anchor="start" fill="{lt["coverage_date"]}">'
                f"{escape(_fmt_short_date(self._begin))}</text>"
            )
        if self._end:
            parts.append(
                f'<text x="{label_w + bar_w}" y="{date_y}" '
                f'text-anchor="end" fill="{lt["coverage_date"]}">'
                f"{escape(_fmt_short_date(self._end))}</text>"
            )
        parts.append("</svg>")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# CSS infrastructure
# ---------------------------------------------------------------------------

_css_cache: str | None = None
_css_cache_version: int = -1


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


def _get_repr_css() -> str:
    """Return the CSS string, regenerating only when the theme version changes."""
    global _css_cache, _css_cache_version
    version = get_theme_version()
    if _css_cache is not None and _css_cache_version == version:
        return _css_cache
    _css_cache = _repr_css()
    _css_cache_version = version
    return _css_cache


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_repr_html(
    class_name: str,
    meta_rows: list[tuple[str, str]],
    index_names: tuple[str, ...],
    column_names: tuple[str, ...],
    n_rows: int,
    html_row_fn: Callable[[int], str],
    max_preview: int = _MAX_PREVIEW,
) -> str:
    """Build a shared HTML repr for TimeSeries-style displays."""
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
        html.append(f'<tr><td colspan="{total_cols}" class="ts-ellipsis">(empty)</td></tr>')
    else:
        show_all = n_rows <= max_preview * 2 + 1
        head_rows = range(min(max_preview, n_rows))
        tail_rows = range(max(n_rows - max_preview, max_preview), n_rows) if not show_all else range(0)

        for i in head_rows:
            html.append(html_row_fn(i))

        if not show_all:
            ellipsis_cells = "".join('<td class="ts-ellipsis">&hellip;</td>' for _ in range(total_cols))
            html.append(f"<tr>{ellipsis_cells}</tr>")
            for i in tail_rows:
                html.append(html_row_fn(i))

    html.append("</table></div>")
    html.append("</div>")
    return "\n".join(html)


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

    max_w = max((len(line) for line in content_lines if line is not None), default=0)

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
            lines.append("\u2502" + " " * padding + cl.ljust(max_w) + " " * padding + "\u2502")
    lines.append(bot)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DataPoint repr functions (standalone, monkey-patched onto NamedTuple)
# ---------------------------------------------------------------------------


def _datapoint_repr(self) -> str:
    meta_lines: list[str] = []

    ts_str = _fmt_short_date(self.timestamp)
    meta_lines.append(f"Timestamp:  {ts_str}")

    if hasattr(self.timestamp, "utcoffset") and self.timestamp.utcoffset() is not None:
        tz_str = str(self.timestamp.tzinfo)
        tz_display = _fmt_tz_with_offset(tz_str, [self.timestamp])
        meta_lines.append(f"Timezone:   {tz_display}")

    meta_lines.append(f"Value:      {_fmt_value(self.value)}")

    return _render_box("DataPoint", meta_lines)


def _datapoint_repr_html(self) -> str:
    ts_str = escape(_fmt_short_date(self.timestamp))
    val_str = escape(_fmt_value(self.value))

    meta_rows = [("Timestamp", ts_str)]

    if hasattr(self.timestamp, "utcoffset") and self.timestamp.utcoffset() is not None:
        tz_str = str(self.timestamp.tzinfo)
        tz_display = escape(_fmt_tz_with_offset(tz_str, [self.timestamp]))
        meta_rows.append(("Timezone", tz_display))

    meta_rows.append(("Value", val_str))

    html = [_get_repr_css(), '<div class="ts-repr">']
    html.append('<div class="ts-header">DataPoint</div>')
    html.append('<div class="ts-meta"><table>')
    for label, value in meta_rows:
        html.append(f"<tr><td>{label}</td><td>{value}</td></tr>")
    html.append("</table></div>")
    html.append("</div>")
    return "\n".join(html)


#: Maps DataShape.value → ordered index column names for that shape.
_SHAPE_INDEX_COLS: dict[str, tuple[str, ...]] = {
    "SIMPLE": ("valid_time",),
    "VERSIONED": ("knowledge_time", "valid_time"),
    "CORRECTED": ("valid_time", "change_time"),
    "AUDIT": ("knowledge_time", "change_time", "valid_time"),
}


# ---------------------------------------------------------------------------
# Mixin: _TimeSeriesReprMixin
# ---------------------------------------------------------------------------


class _TimeSeriesReprMixin:
    """Repr mixin for the Polars-backed TimeSeries class (timeseries.py)."""

    __slots__ = ()

    def _repr_meta_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = [("Name", self.name)]
        if self.shape is not None:
            pairs.append(("Shape", self.shape.value))
            pairs.append(("Rows", str(self.num_rows)))
        if self.frequency:
            pairs.append(("Frequency", str(self.frequency)))
        pairs.append(("Timezone", self.timezone))
        if self.unit and self.unit != "dimensionless":
            pairs.append(("Unit", self.unit))
        if self.data_type:
            pairs.append(("Data type", str(self.data_type)))
        if self.description:
            pairs.append(("Description", self.description))
        if self.timeseries_type and str(self.timeseries_type) != "FLAT":
            pairs.append(("Timeseries type", str(self.timeseries_type)))
        return pairs

    def _repr_data_rows(self, indices: list[int]) -> list[list[str]]:
        """Return one [timestamp_str, value_str] row per index, reading from Polars."""
        assert self.shape is not None  # caller guards against metadata-only
        idx_cols = _SHAPE_INDEX_COLS[self.shape.value]
        rows: list[list[str]] = []
        for i in indices:
            ts_parts = [_fmt_short_date(self._df[col][i]) for col in idx_cols]
            val = self._df["value"][i]
            rows.append([", ".join(ts_parts), _fmt_value(val)])
        return rows

    def __repr__(self) -> str:
        meta_lines = _format_meta_lines(self._repr_meta_pairs())

        if self.shape is None:
            return _render_box("TimeSeries", list(meta_lines))

        n = self.num_rows
        col_names = [self.name]

        if n == 0:
            indices: list[int] = []
            truncated = False
        elif n <= _MAX_PREVIEW * 2 + 1:
            indices = list(range(n))
            truncated = False
        else:
            indices = list(range(_MAX_PREVIEW)) + list(range(n - _MAX_PREVIEW, n))
            truncated = True

        data_rows = self._repr_data_rows(indices) if indices else []

        content_lines: list[str | None] = []
        for ml in meta_lines:
            content_lines.append(ml)

        if n == 0:
            content_lines.append(None)
            content_lines.append("(empty)")
        else:
            all_rows: list[list[str]] = [[""] + col_names] + data_rows
            ncols_data = len(all_rows[0])
            col_widths = [0] * ncols_data
            for row in all_rows:
                for j, cell in enumerate(row):
                    col_widths[j] = max(col_widths[j], len(cell))
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

            content_lines.append(None)
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

        return _render_box("TimeSeries", content_lines)

    def _repr_html_(self) -> str:
        if self.shape is None:
            html = [_get_repr_css(), '<div class="ts-repr">']
            html.append('<div class="ts-header">TimeSeries</div>')
            html.append('<div class="ts-meta"><table>')
            for label, value in self._repr_meta_pairs():
                html.append(f"<tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>")
            html.append("</table></div>")
            html.append("</div>")
            return "\n".join(html)

        idx_cols = _SHAPE_INDEX_COLS[self.shape.value]
        col_names = (self.name,)

        def _html_row(i: int) -> str:
            idx_cells = "".join(f"<td>{escape(_fmt_short_date(self._df[col][i]))}</td>" for col in idx_cols)
            val = self._df["value"][i]
            return f"<tr>{idx_cells}<td>{escape(_fmt_value(val))}</td></tr>"

        return _build_repr_html(
            class_name="TimeSeries",
            meta_rows=self._repr_meta_pairs(),
            index_names=idx_cols,
            column_names=col_names,
            n_rows=self.num_rows,
            html_row_fn=_html_row,
        )
