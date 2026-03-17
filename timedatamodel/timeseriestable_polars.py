"""
TimeSeriesTablePolars — a Polars-backed container for multivariate time series.

Stores multiple co-indexed time series as named columns in a single
``polars.DataFrame``.  Each value column represents one signal (wind speed,
temperature, solar power output, …) with its own metadata — unit, data type,
and geographic location.

Key characteristics
-------------------
- **Named columns**: value columns are standard Polars DataFrame columns.
- **Shared timestamps**: all columns share a ``valid_time`` column stored as
  ``pl.Datetime("us", time_zone="UTC")``.
- **Per-column metadata**: independent units, data types, and locations per
  column; a length-1 list broadcasts to all columns.
- **Spatial filtering**: select columns by geographic proximity using
  :class:`~timedatamodel.location.GeoLocation`.
- **Pandas interop**: :meth:`from_pandas` / :meth:`to_pandas` with
  ``valid_time`` as index.

Example usage
-------------
>>> import pandas as pd
>>> from timedatamodel.timeseriestable_polars import TimeSeriesTablePolars
>>> from timedatamodel.enums import Frequency
>>>
>>> df = pd.DataFrame({
...     "valid_time": pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC"),
...     "wind":       [8.0, 9.1, 10.2, 9.5],
...     "temp":       [-1.0, -0.5, 0.1, -0.3],
... })
>>> table = TimeSeriesTablePolars.from_pandas(df, frequency=Frequency.PT1H, units=["m/s", "degC"])
>>> table.column_names
['wind', 'temp']
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import pandas as pd
import polars as pl

from ._repr import _TimeSeriesTablePolarsReprMixin
from .enums import DataType, Frequency, TimeSeriesType
from .location import GeoArea, GeoLocation
from .datashape import DataShape
from .timeseries_polars import TimeSeriesPolars, _ingest_pandas_to_polars


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Columns that are NOT value columns
_NON_VALUE_COLS: frozenset = frozenset({"valid_time", "valid_time_end"})


def _broadcast_meta(lst: Optional[List], n: int, default) -> List:
    """Broadcast a length-1 list to *n* elements, or validate length == *n*."""
    if lst is None:
        return [default() if callable(default) else default for _ in range(n)]
    if len(lst) == 1:
        return [lst[0]] * n
    if len(lst) == n:
        return list(lst)
    raise ValueError(f"metadata list has length {len(lst)}, expected 1 or {n}")


def _value_col_names(df: pl.DataFrame) -> List[str]:
    """Return column names that are not temporal metadata columns."""
    return [c for c in df.columns if c not in _NON_VALUE_COLS]


# ---------------------------------------------------------------------------
# TimeSeriesTablePolars
# ---------------------------------------------------------------------------


class TimeSeriesTablePolars(_TimeSeriesTablePolarsReprMixin):
    """Polars-backed container for multivariate time series data.

    Parameters
    ----------
    df:
        A ``polars.DataFrame`` with a ``valid_time`` column
        (``pl.Datetime("us", "UTC")``) and one or more Float64 value columns.
    frequency:
        Expected data cadence (:class:`~timedatamodel.enums.Frequency`).
    timezone:
        IANA timezone string for display purposes (metadata hint; data is UTC).
    units:
        Physical unit per value column.  Length-1 list broadcasts to all.
    descriptions:
        Human-readable description per column.
    data_types:
        Semantic data type per column (:class:`~timedatamodel.enums.DataType`).
    locations:
        Geographic location per column (:class:`~timedatamodel.location.GeoLocation`).
    timeseries_types:
        Storage/versioning model per column.
    labels:
        Arbitrary key-value labels per column.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        *,
        frequency: Frequency,
        timezone: str = "UTC",
        units: Optional[List[Optional[str]]] = None,
        descriptions: Optional[List[Optional[str]]] = None,
        data_types: Optional[List[Optional[DataType]]] = None,
        locations: Optional[List[Optional[GeoLocation]]] = None,
        timeseries_types: Optional[List[TimeSeriesType]] = None,
        labels: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"df must be a polars.DataFrame, got {type(df)!r}")
        if "valid_time" not in df.columns:
            raise ValueError("df must have a 'valid_time' column")
        vt_dtype = df["valid_time"].dtype
        if (
            not isinstance(vt_dtype, pl.Datetime)
            or vt_dtype.time_zone is None
            or vt_dtype.time_zone != "UTC"
        ):
            raise TypeError(
                "valid_time must be pl.Datetime with timezone 'UTC', "
                f"got {vt_dtype!r}"
            )
        if "valid_time_end" in df.columns:
            vte_dtype = df["valid_time_end"].dtype
            if (
                not isinstance(vte_dtype, pl.Datetime)
                or vte_dtype.time_zone is None
                or vte_dtype.time_zone != "UTC"
            ):
                raise TypeError(
                    "valid_time_end must be pl.Datetime with timezone 'UTC', "
                    f"got {vte_dtype!r}"
                )

        # Reject extra time-like columns that are not currently modeled as
        # dedicated time axes by TimeSeriesTablePolars. If present, they would
        # be incorrectly treated as value columns and break metadata broadcasting.
        _unsupported_time_cols = {"knowledge_time", "change_time"}
        present_unsupported = _unsupported_time_cols.intersection(df.columns)
        if present_unsupported:
            raise ValueError(
                "df contains unsupported time-like columns that would be "
                "misinterpreted as value columns: "
                f"{sorted(present_unsupported)!r}. "
                "These columns are not yet supported by TimeSeriesTablePolars."
            )

        self._df: pl.DataFrame = df
        self.frequency: Frequency = frequency
        self.timezone: str = timezone

        n = len(_value_col_names(df))
        self._units: List[Optional[str]] = _broadcast_meta(units, n, lambda: None)
        self._descriptions: List[Optional[str]] = _broadcast_meta(descriptions, n, lambda: None)
        self._data_types: List[Optional[DataType]] = _broadcast_meta(data_types, n, lambda: None)
        self._locations: List[Optional[GeoLocation]] = _broadcast_meta(locations, n, lambda: None)
        self._timeseries_types: List[TimeSeriesType] = _broadcast_meta(
            timeseries_types, n, lambda: TimeSeriesType.FLAT
        )
        self._labels: List[Dict[str, str]] = _broadcast_meta(labels, n, dict)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def df(self) -> pl.DataFrame:
        """The underlying ``polars.DataFrame`` (read-only by convention)."""
        return self._df

    @property
    def column_names(self) -> List[str]:
        """Names of the value columns (excludes ``valid_time``)."""
        return _value_col_names(self._df)

    @property
    def n_columns(self) -> int:
        """Number of value columns."""
        return len(self.column_names)

    @property
    def num_rows(self) -> int:
        """Number of rows."""
        return self._df.height

    @property
    def has_missing(self) -> bool:
        """True if any value column contains null values."""
        for col in self.column_names:
            if self._df[col].is_null().any():
                return True
        return False

    @property
    def units(self) -> List[Optional[str]]:
        """Per-column unit strings."""
        return list(self._units)

    @property
    def locations(self) -> List[Optional[GeoLocation]]:
        """Per-column geographic locations."""
        return list(self._locations)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_polars(
        cls,
        df: pl.DataFrame,
        *,
        frequency: Frequency,
        timezone: str = "UTC",
        units: Optional[List[Optional[str]]] = None,
        descriptions: Optional[List[Optional[str]]] = None,
        data_types: Optional[List[Optional[DataType]]] = None,
        locations: Optional[List[Optional[GeoLocation]]] = None,
        timeseries_types: Optional[List[TimeSeriesType]] = None,
        labels: Optional[List[Dict[str, str]]] = None,
    ) -> "TimeSeriesTablePolars":
        """Create a :class:`TimeSeriesTablePolars` directly from a ``polars.DataFrame``.

        The DataFrame must have a ``valid_time`` column with dtype
        ``pl.Datetime("us", time_zone="UTC")``.
        """
        return cls(
            df,
            frequency=frequency,
            timezone=timezone,
            units=units,
            descriptions=descriptions,
            data_types=data_types,
            locations=locations,
            timeseries_types=timeseries_types,
            labels=labels,
        )

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        *,
        frequency: Frequency,
        timezone: str = "UTC",
        units: Optional[List[Optional[str]]] = None,
        descriptions: Optional[List[Optional[str]]] = None,
        data_types: Optional[List[Optional[DataType]]] = None,
        locations: Optional[List[Optional[GeoLocation]]] = None,
        timeseries_types: Optional[List[TimeSeriesType]] = None,
        labels: Optional[List[Dict[str, str]]] = None,
    ) -> "TimeSeriesTablePolars":
        """Create a :class:`TimeSeriesTablePolars` from a ``pandas.DataFrame``.

        The DataFrame must have a ``valid_time`` column (or as index) and one
        or more value columns.  Timestamps are normalised to UTC.
        """
        polars_df = _ingest_pandas_to_polars(df)
        return cls(
            polars_df,
            frequency=frequency,
            timezone=timezone,
            units=units,
            descriptions=descriptions,
            data_types=data_types,
            locations=locations,
            timeseries_types=timeseries_types,
            labels=labels,
        )

    @classmethod
    def from_timeseries(
        cls,
        series_list: List[TimeSeriesPolars],
        *,
        frequency: Optional[Frequency] = None,
        timezone: str = "UTC",
        units: Optional[List[Optional[str]]] = None,
        descriptions: Optional[List[Optional[str]]] = None,
        data_types: Optional[List[Optional[DataType]]] = None,
        locations: Optional[List[Optional[GeoLocation]]] = None,
        timeseries_types: Optional[List[TimeSeriesType]] = None,
        labels: Optional[List[Dict[str, str]]] = None,
    ) -> "TimeSeriesTablePolars":
        """Build a table from a list of SIMPLE-shape :class:`~timedatamodel.timeseries_polars.TimeSeriesPolars`.

        Column names come from each series' ``name`` attribute.  Per-column
        metadata (unit, data_type, location) is derived from each series
        unless explicitly overridden.

        All input series must share identical ``valid_time`` values; a
        :class:`ValueError` is raised if any series has different timestamps.
        The series are then joined on ``valid_time``, so the resulting table
        has the same time index as the inputs.
        """
        if not series_list:
            raise ValueError("series_list must not be empty")
        for ts in series_list:
            if ts.shape is not DataShape.SIMPLE:
                raise ValueError(
                    f"from_timeseries only accepts SIMPLE-shape TimeSeriesPolars, "
                    f"got {ts.shape.value} for '{ts.name}'"
                )

        # Validate all series share identical timestamps
        ref_times = series_list[0].df["valid_time"]
        for ts in series_list[1:]:
            if not ts.df["valid_time"].equals(ref_times):
                raise ValueError(
                    f"All series must have identical timestamps. "
                    f"Series '{ts.name}' has different valid_time values than '{series_list[0].name}'."
                )

        # Build merged DataFrame
        merged: Optional[pl.DataFrame] = None
        for ts in series_list:
            col_name = ts.name or "value"
            renamed = ts.df.rename({"value": col_name})
            if merged is None:
                merged = renamed
            else:
                merged = merged.join(renamed, on="valid_time", how="full", coalesce=True)

        merged = merged.sort("valid_time")

        # Derive per-column metadata from individual TimeSeriesPolars if not overridden
        if units is None:
            units = [ts.unit for ts in series_list]
        if descriptions is None:
            descriptions = [ts.description for ts in series_list]
        if data_types is None:
            data_types = [ts.data_type for ts in series_list]
        if locations is None:
            locations = [ts.location for ts in series_list]
        if timeseries_types is None:
            timeseries_types = [ts.timeseries_type for ts in series_list]
        if labels is None:
            labels = [dict(ts.labels) for ts in series_list]
        if frequency is None:
            freqs = [ts.frequency for ts in series_list if ts.frequency is not None]
            frequency = freqs[0] if freqs else None
        if frequency is None:
            raise ValueError(
                "frequency could not be inferred from series; pass it explicitly"
            )

        return cls(
            merged,
            frequency=frequency,
            timezone=timezone,
            units=units,
            descriptions=descriptions,
            data_types=data_types,
            locations=locations,
            timeseries_types=timeseries_types,
            labels=labels,
        )

    # ------------------------------------------------------------------
    # Column selection
    # ------------------------------------------------------------------

    def select_column(self, col: Union[int, str]) -> TimeSeriesPolars:
        """Extract one value column as a :class:`~timedatamodel.timeseries_polars.TimeSeriesPolars`.

        Parameters
        ----------
        col:
            Column name (str) or zero-based index among value columns (int).
        """
        names = self.column_names
        if isinstance(col, str):
            if col not in names:
                raise KeyError(f"Column '{col}' not found. Available: {names}")
            idx = names.index(col)
        else:
            if not (0 <= col < len(names)):
                raise IndexError(f"Column index {col} out of range (n_columns={len(names)})")
            idx = col

        col_name = names[idx]
        sub_df = self._df.select(["valid_time", col_name]).rename({col_name: "value"})

        return TimeSeriesPolars(
            sub_df,
            name=col_name,
            unit=self._units[idx] or "dimensionless",
            description=self._descriptions[idx],
            data_type=self._data_types[idx],
            location=self._locations[idx],
            timeseries_type=self._timeseries_types[idx],
            labels=self._labels[idx],
            frequency=self.frequency,
            timezone=self.timezone,
        )

    def _select_columns(self, indices: List[int]) -> "TimeSeriesTablePolars":
        """Return a new table keeping only the given column indices."""
        names = self.column_names
        keep_cols = ["valid_time"] + [names[i] for i in indices]
        new_df = self._df.select(keep_cols)
        return TimeSeriesTablePolars(
            new_df,
            frequency=self.frequency,
            timezone=self.timezone,
            units=[self._units[i] for i in indices] or None,
            descriptions=[self._descriptions[i] for i in indices] or None,
            data_types=[self._data_types[i] for i in indices] or None,
            locations=[self._locations[i] for i in indices] or None,
            timeseries_types=[self._timeseries_types[i] for i in indices] or None,
            labels=[self._labels[i] for i in indices] or None,
        )

    # ------------------------------------------------------------------
    # Spatial filtering
    # ------------------------------------------------------------------

    def filter_columns_by_location(
        self, center: GeoLocation, radius_km: float
    ) -> "TimeSeriesTablePolars":
        """Keep only columns whose location is within *radius_km* of *center*."""
        keep = [
            i for i, loc in enumerate(self._locations)
            if isinstance(loc, GeoLocation) and loc.distance_to(center) <= radius_km
        ]
        return self._select_columns(keep)

    def filter_columns_by_area(self, area: GeoArea) -> "TimeSeriesTablePolars":
        """Keep only columns whose location falls inside *area*."""
        keep = [
            i for i, loc in enumerate(self._locations)
            if isinstance(loc, GeoLocation) and loc.is_within(area)
        ]
        return self._select_columns(keep)

    def nearest_columns(
        self, target: GeoLocation, n: int = 1
    ) -> "TimeSeriesTablePolars":
        """Keep the *n* nearest columns to *target* by Haversine distance."""
        dists = [
            (loc.distance_to(target), i)
            for i, loc in enumerate(self._locations)
            if isinstance(loc, GeoLocation)
        ]
        dists.sort(key=lambda x: x[0])
        keep = [idx for _, idx in dists[:n]]
        return self._select_columns(keep)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def head(self, n: int = 5) -> "TimeSeriesTablePolars":
        """Return the first *n* rows as a new :class:`TimeSeriesTablePolars`."""
        return self._clone_df(self._df.head(n))

    def tail(self, n: int = 5) -> "TimeSeriesTablePolars":
        """Return the last *n* rows as a new :class:`TimeSeriesTablePolars`."""
        return self._clone_df(self._df.tail(n))

    def _clone_df(self, new_df: pl.DataFrame) -> "TimeSeriesTablePolars":
        return TimeSeriesTablePolars(
            new_df,
            frequency=self.frequency,
            timezone=self.timezone,
            units=list(self._units),
            descriptions=list(self._descriptions),
            data_types=list(self._data_types),
            locations=list(self._locations),
            timeseries_types=list(self._timeseries_types),
            labels=list(self._labels),
        )

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_pandas(self) -> pd.DataFrame:
        """Convert to a ``pandas.DataFrame`` with ``valid_time`` as index."""
        return self._df.to_pandas().set_index("valid_time")

    def validate_for_insert(self):
        """Return ``(pl.DataFrame, DataShape.SIMPLE)`` for database write paths."""
        return self._df, DataShape.SIMPLE

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def metadata_dict(self) -> dict:
        """Return all metadata as a plain dict — useful for serialisation."""
        names = self.column_names
        return {
            "frequency": self.frequency,
            "timezone": self.timezone,
            "num_rows": self.num_rows,
            "columns": {
                name: {
                    "unit": self._units[i],
                    "description": self._descriptions[i],
                    "data_type": self._data_types[i].value if self._data_types[i] else None,
                    "location": {
                        "latitude": self._locations[i].latitude,
                        "longitude": self._locations[i].longitude,
                    } if self._locations[i] else None,
                    "timeseries_type": self._timeseries_types[i].value,
                    "labels": self._labels[i],
                }
                for i, name in enumerate(names)
            },
        }

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._df.height
