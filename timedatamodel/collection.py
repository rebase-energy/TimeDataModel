from __future__ import annotations

from typing import Iterator

from ._base import _DataFrameMixin, _import_polars
from ._repr import _TimeSeriesCollectionReprMixin
from .location import GeoArea, GeoLocation
from .table import TimeSeriesTable
from .timeseries import TimeSeriesList


class TimeSeriesCollection(_TimeSeriesCollectionReprMixin, _DataFrameMixin):
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

