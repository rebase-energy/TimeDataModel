from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Iterator

from ._base import _DataFrameMixin, _fmt_short_date, _import_polars, _render_box
from ._theme import THEME
from .coverage import CoverageBar
from .location import GeoArea, GeoLocation
from .table import TimeSeriesTable
from .timeseries import TimeSeriesList


class TimeSeriesCollection(_DataFrameMixin):
    """Container for TimeSeriesList and/or TimeSeriesTable objects that don't share an index.

    Items are stored internally as an ordered ``dict[str, TimeSeriesList | TimeSeriesTable]``.
    """

    __slots__ = ("_series", "_name", "_description")

    def __init__(
        self,
        series: (
            list[TimeSeriesList | TimeSeriesTable]
            | dict[str, TimeSeriesList | TimeSeriesTable]
            | None
        ) = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self._name = name
        self._description = description

        if series is None:
            self._series: dict[str, TimeSeriesList | TimeSeriesTable] = {}
        elif isinstance(series, dict):
            self._series = dict(series)
        else:
            self._series = {}
            used: dict[str, int] = {}
            for idx, item in enumerate(series):
                key: str | None = None
                if isinstance(item, TimeSeriesList) and item.name:
                    key = item.name
                elif isinstance(item, TimeSeriesTable):
                    names = item.column_names
                    if names:
                        key = ",".join(names)

                if key is None:
                    key = f"series_{idx}"

                if key in used:
                    used[key] += 1
                    key = f"{key}_{used[key]}"
                else:
                    used[key] = 0
                self._series[key] = item

    # ---- properties -------------------------------------------------------

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def names(self) -> list[str]:
        return list(self._series.keys())

    @property
    def series_count(self) -> int:
        return len(self._series)

    # ---- mapping / sequence protocol --------------------------------------

    def __len__(self) -> int:
        return len(self._series)

    def __bool__(self) -> bool:
        return len(self._series) > 0

    def __contains__(self, key: str) -> bool:
        return key in self._series

    def __iter__(self) -> Iterator[str]:
        return iter(self._series)

    def __getitem__(self, key: str | int) -> TimeSeriesList | TimeSeriesTable:
        if isinstance(key, int):
            keys = list(self._series.keys())
            return self._series[keys[key]]
        return self._series[key]

    def keys(self):
        return self._series.keys()

    def values(self):
        return self._series.values()

    def items(self):
        return self._series.items()

    # ---- mutation (returns new collection) --------------------------------

    def add(
        self,
        item: TimeSeriesList | TimeSeriesTable,
        name: str | None = None,
    ) -> TimeSeriesCollection:
        if name is None:
            if isinstance(item, TimeSeriesList) and item.name:
                name = item.name
            elif isinstance(item, TimeSeriesTable):
                names = item.column_names
                name = ",".join(names) if names else None
            if name is None:
                name = f"series_{len(self._series)}"
        new_series = dict(self._series)
        new_series[name] = item
        return TimeSeriesCollection(
            new_series, name=self._name, description=self._description
        )

    def remove(self, name: str) -> TimeSeriesCollection:
        new_series = {k: v for k, v in self._series.items() if k != name}
        return TimeSeriesCollection(
            new_series, name=self._name, description=self._description
        )

    # ---- spatial filtering -------------------------------------------------

    @staticmethod
    def _item_distance(
        item: TimeSeriesList | TimeSeriesTable, target: GeoLocation
    ) -> float | None:
        """Return the minimum distance from *item* to *target*, or None if no location."""
        if isinstance(item, TimeSeriesList):
            loc = item.location
            if isinstance(loc, GeoLocation):
                return loc.distance_to(target)
            if isinstance(loc, GeoArea):
                return loc.centroid.distance_to(target)
            return None
        # TimeSeriesTable — take min across columns
        dists: list[float] = []
        for i in range(item.n_columns):
            loc = item._get_attr(item.locations, i)
            if isinstance(loc, GeoLocation):
                dists.append(loc.distance_to(target))
            elif isinstance(loc, GeoArea):
                dists.append(loc.centroid.distance_to(target))
        return min(dists) if dists else None

    @staticmethod
    def _item_in_radius(
        item: TimeSeriesList | TimeSeriesTable,
        center: GeoLocation,
        radius_km: float,
    ) -> bool:
        """True if any location on *item* is within *radius_km* of *center*."""
        if isinstance(item, TimeSeriesList):
            loc = item.location
            if isinstance(loc, GeoLocation):
                return loc.distance_to(center) <= radius_km
            if isinstance(loc, GeoArea):
                return loc.centroid.distance_to(center) <= radius_km
            return False
        for i in range(item.n_columns):
            loc = item._get_attr(item.locations, i)
            if isinstance(loc, GeoLocation) and loc.distance_to(center) <= radius_km:
                return True
            if isinstance(loc, GeoArea) and loc.centroid.distance_to(center) <= radius_km:
                return True
        return False

    @staticmethod
    def _item_in_area(
        item: TimeSeriesList | TimeSeriesTable, area: GeoArea
    ) -> bool:
        """True if any location on *item* is inside *area*."""
        if isinstance(item, TimeSeriesList):
            loc = item.location
            if isinstance(loc, GeoLocation):
                return loc.is_within(area)
            if isinstance(loc, GeoArea):
                return area.contains_area(loc)
            return False
        for i in range(item.n_columns):
            loc = item._get_attr(item.locations, i)
            if isinstance(loc, GeoLocation) and loc.is_within(area):
                return True
            if isinstance(loc, GeoArea) and area.contains_area(loc):
                return True
        return False

    def filter_by_location(
        self, center: GeoLocation, radius_km: float
    ) -> TimeSeriesCollection:
        """Keep series within *radius_km* of *center*."""
        new_series = {
            k: v
            for k, v in self._series.items()
            if self._item_in_radius(v, center, radius_km)
        }
        return TimeSeriesCollection(
            new_series, name=self._name, description=self._description
        )

    def filter_by_area(self, area: GeoArea) -> TimeSeriesCollection:
        """Keep series inside *area*."""
        new_series = {
            k: v
            for k, v in self._series.items()
            if self._item_in_area(v, area)
        }
        return TimeSeriesCollection(
            new_series, name=self._name, description=self._description
        )

    def nearest(
        self, target: GeoLocation, n: int = 1
    ) -> TimeSeriesCollection:
        """Keep the *n* nearest series to *target*."""
        scored: list[tuple[float, str]] = []
        for key, item in self._series.items():
            d = self._item_distance(item, target)
            if d is not None:
                scored.append((d, key))
        scored.sort(key=lambda x: x[0])
        keep_keys = {key for _, key in scored[:n]}
        new_series = {k: v for k, v in self._series.items() if k in keep_keys}
        return TimeSeriesCollection(
            new_series, name=self._name, description=self._description
        )

    # ---- conversion --------------------------------------------------------

    def to_pandas_dataframe(self) -> "pd.DataFrame":
        """Outer-join all series into a single pandas DataFrame.

        Each series becomes a column named by its key.  The index is the
        union of all timestamps (outer join), with ``NaN`` for missing values.
        """
        import pandas as pd

        if not self._series:
            return pd.DataFrame()

        frames: dict[str, "pd.Series"] = {}
        for key, item in self._series.items():
            df_item = item.to_pandas_dataframe()
            # TimeSeriesList produces a single-column DataFrame; extract the Series
            if df_item.shape[1] == 1:
                frames[key] = df_item.iloc[:, 0]
            else:
                # TimeSeriesTable: each column gets a composite key
                for col in df_item.columns:
                    frames[f"{key}/{col}"] = df_item[col]

        return pd.DataFrame(frames)

    def to_pd_df(self) -> "pd.DataFrame":
        """Alias for ``to_pandas_dataframe()``."""
        return self.to_pandas_dataframe()

    def to_polars_dataframe(self):
        """Outer-join all series into a single polars DataFrame."""
        pl = _import_polars()

        pdf = self.to_pandas_dataframe()
        return pl.from_pandas(pdf.reset_index())

    def to_pl_df(self):
        """Alias for ``to_polars_dataframe()``."""
        return self.to_polars_dataframe()

    def to_numpy(self) -> "dict[str, np.ndarray]":
        """Return each series as a numpy array in a dict keyed by series name."""
        import numpy as np

        result: dict[str, np.ndarray] = {}
        for key, item in self._series.items():
            result[key] = item.to_numpy()
        return result

    @property
    def arr(self) -> "dict[str, np.ndarray]":
        """Shorthand for ``to_numpy()``."""
        return self.to_numpy()

    # ---- display -----------------------------------------------------------

    def _item_summary(self, key: str, item: TimeSeriesList | TimeSeriesTable) -> dict:
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

        title = type(self).__name__

        lt = THEME["light"]
        dk = THEME["dark"]
        css = f"""\
<style>
.tsc-repr {{ font-family: monospace; font-size: 13px; max-width: 720px; }}
.tsc-repr .tsc-header {{
  font-weight: bold; font-size: 14px;
  padding: 6px 10px; border-bottom: 2px solid {lt["header_border"]};
  background: {lt["header_bg"]}; color: {lt["header_text"]};
}}
.tsc-repr table {{ border-collapse: collapse; width: 100%; }}
.tsc-repr th {{
  text-align: left; padding: 3px 10px; border-bottom: 1px solid {lt["col_header_border"]};
  color: {lt["col_header_text"]}; font-weight: 600;
}}
.tsc-repr td {{ padding: 2px 10px; }}
.tsc-repr tr:hover {{ background: {lt["hover_bg"]}; }}
@media (prefers-color-scheme: dark) {{
  .tsc-repr .tsc-header {{ background: {dk["header_bg"]}; color: {dk["header_text"]}; border-color: {dk["header_border"]}; }}
  .tsc-repr th {{ color: {dk["col_header_text"]}; border-color: {dk["col_header_border"]}; }}
  .tsc-repr td {{ color: {dk["data_text"]}; }}
  .tsc-repr tr:hover {{ background: {dk["hover_bg"]}; }}
}}
</style>"""

        html = [css, '<div class="tsc-repr">']
        html.append(f'<div class="tsc-header">{escape(title)}</div>')
        html.append("<table>")
        html.append(
            "<tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr>"
        )
        for row in rows:
            html.append(
                "<tr>"
                + "".join(f"<td>{escape(row[h])}</td>" for h in headers)
                + "</tr>"
            )
        html.append("</table></div>")
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
