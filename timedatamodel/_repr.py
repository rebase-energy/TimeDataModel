"""Centralized display / repr logic for all timedatamodel data classes.

This module contains:
- Module-level formatting helpers (moved from _base.py)
- CSS infrastructure for HTML reprs
- CoverageBar class (moved from coverage.py)
- HierarchyTree class (moved from hierarchy.py)
- DataPoint repr functions (standalone, monkey-patched)
- Mixin classes providing __repr__ / _repr_html_ for each data class
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from itertools import product
from typing import TYPE_CHECKING, Callable

import numpy as np

from ._theme import THEME, get_theme_version
from .location import GeoArea, GeoLocation

if TYPE_CHECKING:
    from .hierarchy import HierarchyNode

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
        if hasattr(first, 'utcoffset') and first.utcoffset() is not None:
            offset = first.utcoffset()
            total_seconds = int(offset.total_seconds())
            sign = '+' if total_seconds >= 0 else '-'
            hours, remainder = divmod(abs(total_seconds), 3600)
            minutes = remainder // 60
            return f"{tz_str} ({sign}{hours:02d}:{minutes:02d})"
    return tz_str


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
                '<td class="ts-ellipsis">&hellip;</td>'
                for _ in range(total_cols)
            )
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
            lines.append(
                "\u2502" + " " * padding + cl.ljust(max_w) + " " * padding + "\u2502"
            )
    lines.append(bot)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CoverageBar (moved from coverage.py)
# ---------------------------------------------------------------------------


class CoverageBar:
    """Displayable coverage bar for TimeSeriesList objects."""

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
            bar = "".join("\u2588" if b else "\u2591" for b in binned)
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

        lt = THEME["light"]
        for row_idx, (name, mask) in enumerate(self._masks):
            y = row_idx * row_h
            # label
            parts.append(
                f'<text x="{label_w - 6}" y="{y + 15}" '
                f'text-anchor="end" fill="{lt["coverage_label"]}">{escape(name)}</text>'
            )
            # bar segments
            binned = self._bin_coverage(mask, n_bins)
            seg_w = bar_w / len(binned) if binned else bar_w
            for i, b in enumerate(binned):
                color = lt["coverage_present"] if b else lt["coverage_absent"]
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
                f'text-anchor="start" fill="{lt["coverage_date"]}">'
                f'{escape(_fmt_short_date(self._begin))}</text>'
            )
        if self._end:
            parts.append(
                f'<text x="{label_w + bar_w}" y="{date_y}" '
                f'text-anchor="end" fill="{lt["coverage_date"]}">'
                f'{escape(_fmt_short_date(self._end))}</text>'
            )

        parts.append("</svg>")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# HierarchyTree (moved from hierarchy.py)
# ---------------------------------------------------------------------------


class HierarchyTree:
    """Displayable tree visualization for a HierarchicalTimeSeries."""

    __slots__ = ("_root",)

    def __init__(self, root: HierarchyNode) -> None:
        self._root = root

    def __repr__(self) -> str:
        lines: list[str] = []
        self._build_tree(self._root, "", True, lines)
        return "\n".join(lines)

    @staticmethod
    def _build_tree(
        node: HierarchyNode,
        prefix: str,
        is_last: bool,
        lines: list[str],
    ) -> None:
        connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
        label = node.key
        if node.is_leaf and node.timeseries is not None:
            label += f" [{len(node.timeseries)} pts]"
        elif not node.is_leaf:
            label += f" ({node.level})"
        lines.append(f"{prefix}{connector}{label}")
        new_prefix = prefix + ("    " if is_last else "\u2502   ")
        for i, child in enumerate(node.children):
            HierarchyTree._build_tree(
                child, new_prefix, i == len(node.children) - 1, lines
            )

    def _repr_html_(self) -> str:
        css = """\
<style>
.tsh-tree { font-family: monospace; font-size: 13px; }
.tsh-tree details { margin-left: 16px; }
.tsh-tree summary { cursor: pointer; padding: 1px 0; }
.tsh-tree .tsh-leaf { margin-left: 16px; padding: 1px 0; }
</style>"""
        return css + '\n<div class="tsh-tree">\n' + self._html_node(self._root) + "</div>"

    @staticmethod
    def _html_node(node: HierarchyNode) -> str:
        if node.is_leaf:
            label = escape(node.key)
            if node.timeseries is not None:
                label += f" [{len(node.timeseries)} pts]"
            return f'<div class="tsh-leaf">{label}</div>\n'
        label = f"{escape(node.key)} ({escape(node.level)})"
        children_html = "".join(
            HierarchyTree._html_node(c) for c in node.children
        )
        return (
            f"<details open>"
            f"<summary>{label}</summary>\n"
            f"{children_html}"
            f"</details>\n"
        )


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


# ---------------------------------------------------------------------------
# Mixin: _TimeSeriesBaseReprMixin
# ---------------------------------------------------------------------------


class _TimeSeriesBaseReprMixin:
    """Repr mixin for _TimeSeriesBase (shared by List and Table)."""
    __slots__ = ()

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

    @staticmethod
    def _fmt_value(v: float | None) -> str:
        return _fmt_value(v)

    @staticmethod
    def _fmt_location(loc: GeoLocation | GeoArea | None) -> str:
        return _fmt_location(loc)


# ---------------------------------------------------------------------------
# Mixin: _TimeSeriesListReprMixin
# ---------------------------------------------------------------------------


class _TimeSeriesListReprMixin:
    """Repr mixin for TimeSeriesList."""
    __slots__ = ()

    def _repr_meta_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = [
            ("Name", self.name or "unnamed"),
            ("Length", str(len(self._timestamps))),
            ("Frequency", str(self.frequency)),
            ("Timezone", _fmt_tz_with_offset(self.timezone, self._timestamps)),
        ]
        if self.unit:
            pairs.append(("Unit", self.unit))
        if self.data_type:
            pairs.append(("Data type", str(self.data_type)))
        if self.location:
            pairs.append(("Location", _fmt_location(self.location)))
        if self.description:
            pairs.append(("Description", self.description))
        if self.timeseries_type and self.timeseries_type != "FLAT":
            pairs.append(("Timeseries type", str(self.timeseries_type)))
        if self.labels:
            pairs.append(("Labels", str(self.labels)))
        return pairs

    def _repr_data_rows(self, indices: list[int]) -> list[list[str]]:
        rows: list[list[str]] = []
        for i in indices:
            rows.append([_fmt_timestamp(self._timestamps[i]), _fmt_value(self._values[i])])
        return rows

    def _repr_html_(self) -> str:
        n = len(self._timestamps)
        meta_rows = self._repr_meta_pairs()

        def _html_row(i: int) -> str:
            ts_cells = _fmt_timestamp_cells(self._timestamps[i])
            val_cells = (
                f"<td>{escape(_fmt_value(self._values[i]))}</td>"
            )
            return f"<tr>{ts_cells}{val_cells}</tr>"

        return _build_repr_html(
            class_name=type(self).__name__,
            meta_rows=meta_rows,
            index_names=self.index_names,
            column_names=self.column_names,
            n_rows=n,
            html_row_fn=_html_row,
        )

    def _coverage_masks(self) -> list[tuple[str, list[bool]]]:
        return [(self.name or "value", [v is not None for v in self._values])]

    def coverage_bar(self) -> CoverageBar:
        """Return a displayable coverage bar."""
        return CoverageBar(self._coverage_masks(), self.begin, self.end)


# ---------------------------------------------------------------------------
# Mixin: _TimeSeriesTableReprMixin
# ---------------------------------------------------------------------------


class _TimeSeriesTableReprMixin:
    """Repr mixin for TimeSeriesTable."""
    __slots__ = ()

    def _repr_meta_pairs(self) -> list[tuple[str, str]]:
        cn = self.column_names
        pairs: list[tuple[str, str]] = [
            ("Name", "unnamed"),
            ("Columns", ", ".join(cn)),
            ("Length", f"{len(self._timestamps)} \u00d7 {self.n_columns}"),
            ("Frequency", str(self.frequency)),
            ("Timezone", _fmt_tz_with_offset(self.timezone, self._timestamps)),
        ]

        # Unit — show if any is set
        unit_vals = [self._get_attr(self.units, i) for i in range(self.n_columns)]
        if any(u is not None for u in unit_vals):
            pairs.append(("Unit", ", ".join(str(u) if u else "-" for u in unit_vals)))

        # Data type — show if any is set
        dt_vals = [self._get_attr(self.data_types, i) for i in range(self.n_columns)]
        if any(d is not None for d in dt_vals):
            pairs.append(("Data type", ", ".join(str(d) if d else "-" for d in dt_vals)))

        # Location — show if any is set
        loc_vals = [self._get_attr(self.locations, i) for i in range(self.n_columns)]
        if any(loc is not None for loc in loc_vals):
            pairs.append(("Location", ", ".join(_fmt_location(loc) or "-" for loc in loc_vals)))

        # Timeseries type — show if any is not FLAT
        tst_vals = [self._get_attr(self.timeseries_types, i) for i in range(self.n_columns)]
        if any(t != "FLAT" for t in tst_vals):
            pairs.append(("Timeseries type", ", ".join(str(t) for t in tst_vals)))

        # Labels — show if any is non-empty
        lbl_vals = [self._get_attr(self.labels, i) for i in range(self.n_columns)]
        if any(lbl for lbl in lbl_vals):
            pairs.append(("Labels", ", ".join(str(lbl) for lbl in lbl_vals)))

        return pairs

    def _repr_data_rows(self, indices: list[int]) -> list[list[str]]:
        rows: list[list[str]] = []
        for i in indices:
            rows.append(
                [_fmt_timestamp(self._timestamps[i])] + [_fmt_value(float(v)) for v in self._values[i]]
            )
        return rows

    def _repr_html_(self) -> str:
        n = len(self._timestamps)
        meta_rows = self._repr_meta_pairs()

        def _html_row(i: int) -> str:
            ts_cells = _fmt_timestamp_cells(self._timestamps[i])
            val_cells = "".join(
                f"<td>{escape(_fmt_value(float(v)))}</td>"
                for v in self._values[i]
            )
            return f"<tr>{ts_cells}{val_cells}</tr>"

        return _build_repr_html(
            class_name=type(self).__name__,
            meta_rows=meta_rows,
            index_names=self.index_names,
            column_names=self.column_names,
            n_rows=n,
            html_row_fn=_html_row,
        )

    def _coverage_masks(self) -> list[tuple[str, list[bool]]]:
        masks: list[tuple[str, list[bool]]] = []
        for col, name in enumerate(self.column_names):
            col_data = self._values[:, col]
            masks.append((name, [not np.isnan(v) for v in col_data]))
        return masks

    def coverage_bar(self) -> CoverageBar:
        """Return a displayable coverage bar."""
        return CoverageBar(self._coverage_masks(), self.begin, self.end)


# ---------------------------------------------------------------------------
# Mixin: _TimeSeriesArrayReprMixin
# ---------------------------------------------------------------------------


class _TimeSeriesArrayReprMixin:
    """Repr mixin for TimeSeriesArray."""
    __slots__ = ()

    def _repr_meta_pairs(self) -> list[tuple[str, str]]:
        dim_parts = [f"{d.name}: {len(d.labels)}" for d in self.dimensions]
        pairs: list[tuple[str, str]] = [
            ("Name", self.name or "unnamed"),
            ("Dimensions", ", ".join(dim_parts)),
            ("Shape", str(self.shape)),
            ("Frequency", str(self.frequency)),
            ("Timezone", _fmt_tz_with_offset(self.timezone, self.primary_time_dim.labels)),
        ]
        if self.unit:
            pairs.append(("Unit", self.unit))
        if self.data_type:
            pairs.append(("Data type", str(self.data_type)))
        total = self._values.size
        if total > 0:
            n_masked = int(self._values.mask.sum()) if self._values.mask.any() else 0
            if n_masked > 0:
                pct = n_masked / total * 100
                pairs.append(("Masked", f"{n_masked}/{total} ({pct:.1f}%)"))
        return pairs

    def __repr__(self) -> str:
        return _render_box(type(self).__name__, _format_meta_lines(self._repr_meta_pairs()))

    def _repr_html_(self) -> str:
        n_dims = self.ndim
        meta_rows = self._repr_meta_pairs()

        if n_dims >= 2:
            # Classify dimensions: datetime → rows, others → columns
            row_dims: list = []
            col_dims: list = []
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
                        f"{escape(_fmt_value(v))}</td>"
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
            html = [_get_repr_css(), '<div class="ts-repr">']
            html.append(
                f'<div class="ts-header">'
                f"{escape(type(self).__name__)}</div>"
            )
            html.append('<div class="ts-meta"><table>')
            for label, value in meta_rows:
                html.append(
                    f"<tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>"
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
                val_cell = f"<td>{escape(_fmt_value(v))}</td>"
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


# ---------------------------------------------------------------------------
# Mixin: _HierarchicalTimeSeriesReprMixin
# ---------------------------------------------------------------------------


class _HierarchicalTimeSeriesReprMixin:
    """Repr mixin for HierarchicalTimeSeries."""
    __slots__ = ()

    _MAX_LEAF_ROWS = 7

    def _repr_meta_pairs(self) -> list[tuple[str, str]]:
        _tz_timestamps = [self._begin] if self._begin is not None else []
        pairs: list[tuple[str, str]] = [
            ("Name", self._name or "unnamed"),
            ("Levels", ", ".join(self._levels)),
            ("Nodes", f"{self.n_nodes} ({self.n_leaves} leaves)"),
            ("Frequency", str(self._frequency)),
            ("Timezone", _fmt_tz_with_offset(self._timezone, _tz_timestamps)),
        ]
        if self._unit:
            pairs.append(("Unit", self._unit))
        pairs.append(("Aggregation", str(self._aggregation)))
        return pairs

    def _leaf_summary_rows(self) -> list[dict[str, str]]:
        """Build summary rows for leaf nodes."""
        leaf_nodes = self.leaves()
        rows: list[dict[str, str]] = []
        for node in leaf_nodes:
            ts = node.timeseries
            length = str(len(ts)) if ts is not None else "0"
            begin = _fmt_short_date(ts.begin) if ts and ts.begin else "-"
            end = _fmt_short_date(ts.end) if ts and ts.end else "-"
            rows.append({
                "name": node.key,
                "level": node.level,
                "length": length,
                "begin": begin,
                "end": end,
            })
        return rows

    def _leaf_display_rows(self) -> list[dict[str, str]]:
        """Return leaf rows with head/tail truncation applied."""
        rows = self._leaf_summary_rows()
        if len(rows) <= self._MAX_LEAF_ROWS:
            return rows
        headers = ["name", "level", "length", "begin", "end"]
        return rows[:3] + [{h: "..." for h in headers}] + rows[-3:]

    def __repr__(self) -> str:
        class_name = type(self).__name__
        meta_lines = _format_meta_lines(self._repr_meta_pairs())

        # Leaf table
        rows = self._leaf_display_rows()
        headers = ["name", "level", "length", "begin", "end"]
        col_widths = {h: len(h) for h in headers}
        for row in rows:
            for h in headers:
                col_widths[h] = max(col_widths[h], len(row[h]))

        def _fmt_row(vals: dict[str, str]) -> str:
            return "  ".join(f"{vals[h]:<{col_widths[h]}}" for h in headers)

        header_line = _fmt_row({h: h for h in headers})

        # Combine all content lines
        content_lines: list[str | None] = list(meta_lines)
        content_lines.append(None)  # separator
        content_lines.append(header_line)
        content_lines.append(None)  # separator
        for row in rows:
            content_lines.append(_fmt_row(row))

        return _render_box(class_name, content_lines)

    def _repr_html_(self) -> str:
        meta_rows = self._repr_meta_pairs()

        html = [_get_repr_css(), '<div class="ts-repr">']
        html.append(f'<div class="ts-header">{escape(type(self).__name__)}</div>')
        html.append('<div class="ts-meta"><table>')
        for label, value in meta_rows:
            html.append(f"<tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>")
        html.append("</table></div>")

        # Leaf table
        headers = ["name", "level", "length", "begin", "end"]
        display_rows = self._leaf_display_rows()

        html.append('<div class="ts-data"><table style="text-align: left;">')
        html.append(
            "<tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr>"
        )
        for row in display_rows:
            html.append(
                "<tr>"
                + "".join(f"<td>{escape(row[h])}</td>" for h in headers)
                + "</tr>"
            )
        html.append("</table></div></div>")
        return "\n".join(html)

    def tree(self) -> HierarchyTree:
        """Return a displayable tree visualization."""
        return HierarchyTree(self._root)


# ---------------------------------------------------------------------------
# Mixin: _TimeSeriesCollectionReprMixin
# ---------------------------------------------------------------------------


class _TimeSeriesCollectionReprMixin:
    """Repr mixin for TimeSeriesCollection."""
    __slots__ = ()

    def _item_summary(self, key: str, item) -> dict:
        """Summarize a single item for repr tables."""
        kind = type(item).__name__
        freq = str(item.frequency) if hasattr(item, "frequency") else "-"
        tz = item.timezone if hasattr(item, "timezone") else "-"
        n = len(item)
        begin = item.begin
        end = item.end
        begin_s = _fmt_short_date(begin) if begin else "-"
        end_s = _fmt_short_date(end) if end else "-"
        return {
            "name": key,
            "type": kind,
            "freq": freq,
            "tz": tz,
            "length": str(n),
            "begin": begin_s,
            "end": end_s,
        }

    def __repr__(self) -> str:
        class_name = type(self).__name__
        if not self._series:
            return f"{type(self).__name__}(empty)"

        rows = [
            self._item_summary(k, v) for k, v in self._series.items()
        ]
        headers = ["name", "type", "freq", "tz", "length", "begin", "end"]
        col_widths = {h: len(h) for h in headers}
        for row in rows:
            for h in headers:
                col_widths[h] = max(col_widths[h], len(row[h]))

        def _fmt_row(vals: dict) -> str:
            return "  ".join(f"{vals[h]:<{col_widths[h]}}" for h in headers)

        header_line = _fmt_row({h: h for h in headers})
        content_lines: list[str | None] = [header_line]
        content_lines.append(None)  # separator
        for row in rows:
            content_lines.append(_fmt_row(row))

        return _render_box(class_name, content_lines)

    def _repr_html_(self) -> str:
        if not self._series:
            return "<div><b>TimeSeriesCollection</b> (empty)</div>"

        rows = [
            self._item_summary(k, v) for k, v in self._series.items()
        ]
        headers = ["name", "type", "freq", "tz", "length", "begin", "end"]

        html = [_get_repr_css(), '<div class="ts-repr">']
        html.append(f'<div class="ts-header">{escape(type(self).__name__)}</div>')
        html.append('<div class="ts-data"><table style="text-align: left;">')
        html.append(
            "<tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr>"
        )
        nowrap = {"begin", "end"}
        for row in rows:
            cells = []
            for h in headers:
                style = ' style="white-space:nowrap"' if h in nowrap else ""
                cells.append(f"<td{style}>{escape(row[h])}</td>")
            html.append("<tr>" + "".join(cells) + "</tr>")
        html.append("</table></div></div>")
        return "\n".join(html)

    def coverage_bar(self) -> CoverageBar:
        """Return a multi-row CoverageBar spanning the global time range."""
        masks: list[tuple[str, list[bool]]] = []
        all_begins: list[datetime] = []
        all_ends: list[datetime] = []

        for key, item in self._series.items():
            begin = item.begin
            end = item.end
            if begin is not None and end is not None:
                # Unwrap tuples for multi-index
                b = begin[0] if isinstance(begin, tuple) else begin
                e = end[0] if isinstance(end, tuple) else end
                all_begins.append(b)
                all_ends.append(e)

        if not all_begins:
            return CoverageBar([], None, None)

        global_begin = min(all_begins)
        global_end = max(all_ends)
        global_span = (global_end - global_begin).total_seconds()

        n_bins = CoverageBar._TERM_BINS

        for key, item in self._series.items():
            begin = item.begin
            end = item.end

            # Import here to check type without circular import
            from .table import TimeSeriesTable
            from .timeseries import TimeSeriesList

            if isinstance(item, TimeSeriesList):
                label = key
                if begin is None or end is None or global_span == 0:
                    masks.append((label, [False] * n_bins))
                    continue
                b = begin[0] if isinstance(begin, tuple) else begin
                e = end[0] if isinstance(end, tuple) else end
                item_masks = item._coverage_masks()
                _, raw_mask = item_masks[0]
                # Map raw mask onto global bins
                bin_mask = self._rebin_to_global(
                    raw_mask, b, e, global_begin, global_span, n_bins
                )
                masks.append((label, bin_mask))

            elif isinstance(item, TimeSeriesTable):
                if begin is None or end is None or global_span == 0:
                    for col_name in item.column_names:
                        label = f"{key}/{col_name}"
                        masks.append((label, [False] * n_bins))
                    continue
                b = begin[0] if isinstance(begin, tuple) else begin
                e = end[0] if isinstance(end, tuple) else end
                for col_name, raw_mask in item._coverage_masks():
                    label = f"{key}/{col_name}"
                    bin_mask = self._rebin_to_global(
                        raw_mask, b, e, global_begin, global_span, n_bins
                    )
                    masks.append((label, bin_mask))

        return CoverageBar(masks, global_begin, global_end)

    @staticmethod
    def _rebin_to_global(
        raw_mask: list[bool],
        item_begin: datetime,
        item_end: datetime,
        global_begin: datetime,
        global_span: float,
        n_bins: int,
    ) -> list[bool]:
        """Map an item's coverage mask onto global bins."""
        if not raw_mask or global_span == 0:
            return [False] * n_bins

        result = [False] * n_bins
        item_span = (item_end - item_begin).total_seconds()
        n_points = len(raw_mask)

        for i, present in enumerate(raw_mask):
            if not present:
                continue
            # Position of this point in the item's time range
            if n_points == 1:
                t_offset = (item_begin - global_begin).total_seconds()
            else:
                t_offset = (
                    (item_begin - global_begin).total_seconds()
                    + item_span * i / (n_points - 1)
                )
            bin_idx = int(t_offset / global_span * n_bins)
            bin_idx = min(bin_idx, n_bins - 1)
            result[bin_idx] = True

        return result


# ---------------------------------------------------------------------------
# Mixin: _TimeSeriesArrowReprMixin
# ---------------------------------------------------------------------------

#: Maps DataShape.value → ordered index column names for that shape.
_SHAPE_INDEX_COLS: dict[str, tuple[str, ...]] = {
    "SIMPLE":    ("valid_time",),
    "VERSIONED": ("knowledge_time", "valid_time"),
    "CORRECTED": ("valid_time", "change_time"),
    "AUDIT":     ("knowledge_time", "change_time", "valid_time"),
}


class _TimeSeriesArrowReprMixin:
    """Repr mixin for the PyArrow-backed TimeSeries class."""

    __slots__ = ()

    def _repr_meta_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = [
            ("Name",  self.name or "unnamed"),
            ("Shape", self.shape.value),
            ("Rows",  str(self.num_rows)),
        ]
        if self.frequency:
            pairs.append(("Frequency", self.frequency))
        pairs.append(("Timezone", self.timezone))
        if self.unit and self.unit != "dimensionless":
            pairs.append(("Unit", self.unit))
        if self.data_type:
            pairs.append(("Data type", str(self.data_type)))
        if self.location:
            pairs.append(("Location", _fmt_location(self.location)))
        if self.description:
            pairs.append(("Description", self.description))
        if self.timeseries_type and self.timeseries_type != "FLAT":
            pairs.append(("Timeseries type", str(self.timeseries_type)))
        if self.labels:
            pairs.append(("Labels", str(self.labels)))
        return pairs

    def _repr_data_rows(self, indices: list[int]) -> list[list[str]]:
        """Return one [timestamp_str, value_str] row per index, reading from Arrow."""
        idx_cols = _SHAPE_INDEX_COLS[self.shape.value]
        rows: list[list[str]] = []
        for i in indices:
            ts_parts = [
                _fmt_short_date(self._table.column(col)[i].as_py())
                for col in idx_cols
            ]
            val = self._table.column("value")[i].as_py()
            rows.append([", ".join(ts_parts), _fmt_value(val)])
        return rows

    def __repr__(self) -> str:
        meta_lines = _format_meta_lines(self._repr_meta_pairs())
        n = self.num_rows
        col_names = [self.name or "value"]

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
            # Include a header row for column-width computation
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
        idx_cols = _SHAPE_INDEX_COLS[self.shape.value]
        col_names = (self.name or "value",)

        def _html_row(i: int) -> str:
            idx_cells = "".join(
                f"<td>{escape(_fmt_short_date(self._table.column(col)[i].as_py()))}</td>"
                for col in idx_cols
            )
            val = self._table.column("value")[i].as_py()
            return f"<tr>{idx_cells}<td>{escape(_fmt_value(val))}</td></tr>"

        return _build_repr_html(
            class_name="TimeSeries",
            meta_rows=self._repr_meta_pairs(),
            index_names=idx_cols,
            column_names=col_names,
            n_rows=self.num_rows,
            html_row_fn=_html_row,
        )
