"""
TimeSeries — a Pandas-backed container for time series data.

This module mirrors ``timeseries_arrow.py`` but uses a ``pandas.DataFrame``
as the internal storage instead of a ``pyarrow.Table``.

The DataFrame is stored *flat* (no MultiIndex) with all timestamp columns as
regular columns with ``datetime64[us, UTC]`` dtype.  The ``to_pandas()`` method
restores the conventional index/MultiIndex for the given shape.

Data shapes
-----------
See ``timeseries_arrow.py`` for the full description of the four shapes
(SIMPLE, VERSIONED, CORRECTED, AUDIT).  The same column conventions apply
here.

Timestamp representation
------------------------
All timestamp columns are stored internally as ``datetime64[us, UTC]``
(pandas timezone-aware dtype).  The ``timezone`` metadata field is a
display/context hint (IANA zone string).

Example usage
-------------
>>> import pandas as pd
>>> from timedatamodel.timeseries_pandas import TimeSeries
>>>
>>> df = pd.DataFrame({
...     "valid_time": pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC"),
...     "value": [1.0, 2.0, 3.0, 4.0],
... })
>>> ts = TimeSeries.from_pandas(df, name="wind_power", unit="MW")
>>> ts
TimeSeries('wind_power', shape=SIMPLE, rows=4, unit='MW', type=flat)
>>> ts.to_pandas()
                           value
valid_time
2024-01-01 00:00:00+00:00    1.0
...
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .enums import DataType, TimeSeriesType
from .location import GeoLocation
from .timeseries_arrow import DataShape, _REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TIME_COLS: frozenset = frozenset({"valid_time", "valid_time_end", "knowledge_time", "change_time"})
_UTC_DTYPE = pd.DatetimeTZDtype(unit="us", tz="UTC")


# ---------------------------------------------------------------------------
# TimeSeries
# ---------------------------------------------------------------------------

class TimeSeries:
    """Pandas-backed container for time series data with rich metadata.

    Internally stores data as a flat ``pandas.DataFrame`` (no MultiIndex) with
    timestamp columns normalised to ``datetime64[us, UTC]``.

    Parameters
    ----------
    df:
        A flat ``pandas.DataFrame`` (no MultiIndex; timestamps as regular
        columns) conforming to one of the recognised
        :class:`~timedatamodel.timeseries_arrow.DataShape` patterns.
    name:
        Series name (e.g. ``"wind_power"``).
    description:
        Human-readable description.
    unit:
        Canonical physical unit string (e.g. ``"MW"``, ``"dimensionless"``).
    labels:
        Arbitrary key-value labels for series differentiation.
    timezone:
        IANA timezone string for display purposes.  Internal data is always
        UTC; this is a metadata hint only.
    frequency:
        Pandas offset alias describing the expected data cadence.
    data_type:
        Semantic nature of the observations (:class:`~timedatamodel.enums.DataType`).
    location:
        Optional geographic location (:class:`~timedatamodel.location.GeoLocation`).
    timeseries_type:
        Storage/versioning model (:class:`~timedatamodel.enums.TimeSeriesType`).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        unit: str = "dimensionless",
        labels: Optional[Dict[str, str]] = None,
        timezone: str = "UTC",
        frequency: Optional[str] = None,
        data_type: Optional[DataType] = None,
        location: Optional[GeoLocation] = None,
        timeseries_type: TimeSeriesType = TimeSeriesType.FLAT,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas.DataFrame, got {type(df)!r}")
        if isinstance(df.index, pd.MultiIndex) or df.index.name in _TIME_COLS:
            raise ValueError(
                "df must be a flat DataFrame (no temporal MultiIndex). "
                "Use from_pandas() to construct from a DataFrame with a temporal index."
            )

        shape = _infer_shape(df)
        _validate_table(df, shape)

        self._df: pd.DataFrame = df
        self._shape: DataShape = shape

        self.name: Optional[str] = name
        self.description: Optional[str] = description
        self.unit: str = unit
        self.labels: Dict[str, str] = labels or {}
        self.timezone: str = timezone
        self.frequency: Optional[str] = frequency
        self.data_type: Optional[DataType] = data_type
        self.location: Optional[GeoLocation] = location
        self.timeseries_type: TimeSeriesType = timeseries_type

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> DataShape:
        """Which temporal columns are present (inferred from the DataFrame)."""
        return self._shape

    @property
    def num_rows(self) -> int:
        """Number of data rows."""
        return len(self._df)

    @property
    def columns(self) -> List[str]:
        """Column names present in the underlying pandas DataFrame."""
        return list(self._df.columns)

    @property
    def df(self) -> pd.DataFrame:
        """The underlying flat ``pandas.DataFrame`` (read-only by convention)."""
        return self._df

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        unit: str = "dimensionless",
        labels: Optional[Dict[str, str]] = None,
        timezone: str = "UTC",
        frequency: Optional[str] = None,
        data_type: Optional[DataType] = None,
        location: Optional[GeoLocation] = None,
        timeseries_type: TimeSeriesType = TimeSeriesType.FLAT,
    ) -> "TimeSeries":
        """Create a :class:`TimeSeries` from an already-normalised flat
        ``pandas.DataFrame``.

        Use this when data is already a flat DataFrame with UTC-aware timestamp
        columns as regular columns (not as index).  For DataFrames that use a
        temporal index, use :meth:`from_pandas` instead.
        """
        return cls(
            df,
            name=name,
            description=description,
            unit=unit,
            labels=labels,
            timezone=timezone,
            frequency=frequency,
            data_type=data_type,
            location=location,
            timeseries_type=timeseries_type,
        )

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        unit: str = "dimensionless",
        labels: Optional[Dict[str, str]] = None,
        timezone: str = "UTC",
        frequency: Optional[str] = None,
        data_type: Optional[DataType] = None,
        location: Optional[GeoLocation] = None,
        timeseries_type: TimeSeriesType = TimeSeriesType.FLAT,
    ) -> "TimeSeries":
        """Create a :class:`TimeSeries` from a ``pandas.DataFrame``.

        Handles DataFrames with a temporal index or MultiIndex (e.g. as
        returned by the existing timedb read API).

        Only ``SIMPLE`` and ``VERSIONED`` shapes can be constructed via
        ``from_pandas``.  ``AUDIT`` and ``CORRECTED`` shapes (which require a
        ``change_time`` column) are read-only results from the database layer.

        Raises
        ------
        ValueError
            If the DataFrame contains a ``change_time`` column.
        """
        flat_df = _ingest_pandas(df)
        shape = _infer_shape(flat_df)
        if shape in (DataShape.AUDIT, DataShape.CORRECTED):
            raise ValueError(
                f"from_pandas produced shape {shape.value} because 'change_time' is present. "
                f"Only SIMPLE and VERSIONED shapes can be created via from_pandas. "
                f"Use from_dataframe() to wrap an existing read result with change_time."
            )
        return cls(
            flat_df,
            name=name,
            description=description,
            unit=unit,
            labels=labels,
            timezone=timezone,
            frequency=frequency,
            data_type=data_type,
            location=location,
            timeseries_type=timeseries_type,
        )

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def validate_for_insert(self) -> "Tuple[pd.DataFrame, DataShape]":
        """Validate that this TimeSeries can be inserted and return the underlying
        DataFrame with its shape.

        Only :attr:`DataShape.SIMPLE` and :attr:`DataShape.VERSIONED` are
        supported for insert.

        Returns
        -------
        Tuple[pd.DataFrame, DataShape]

        Raises
        ------
        ValueError
            If :attr:`shape` is :attr:`DataShape.AUDIT` or
            :attr:`DataShape.CORRECTED`.
        """
        if self._shape in (DataShape.AUDIT, DataShape.CORRECTED):
            raise ValueError(
                f"TimeSeries with shape {self._shape.value} cannot be inserted. "
                f"Only SIMPLE and VERSIONED shapes are supported for insert."
            )
        return self._df, self._shape

    def to_pandas(self) -> pd.DataFrame:
        """Convert to a ``pandas.DataFrame`` with the conventional index.

        * ``SIMPLE``    — ``valid_time`` as index.
        * ``VERSIONED`` — ``(knowledge_time, valid_time)`` MultiIndex.
        * ``AUDIT``     — ``(knowledge_time, change_time, valid_time)`` MultiIndex.
        * ``CORRECTED`` — ``(valid_time, change_time)`` MultiIndex.
        """
        if self._shape == DataShape.SIMPLE:
            return self._df.set_index("valid_time")
        elif self._shape == DataShape.VERSIONED:
            return self._df.set_index(["knowledge_time", "valid_time"])
        elif self._shape == DataShape.AUDIT:
            return self._df.set_index(["knowledge_time", "change_time", "valid_time"])
        elif self._shape == DataShape.CORRECTED:
            return self._df.set_index(["valid_time", "change_time"])
        return self._df.copy()  # unreachable, safe fallback

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def metadata_dict(self) -> Dict:
        """Return all metadata fields as a plain dict."""
        return {
            "name":            self.name,
            "description":     self.description,
            "unit":            self.unit,
            "labels":          self.labels,
            "timezone":        self.timezone,
            "frequency":       self.frequency,
            "data_type":       self.data_type.value if self.data_type else None,
            "location":        {
                "longitude": self.location.longitude,
                "latitude":  self.location.latitude,
            } if self.location else None,
            "timeseries_type": self.timeseries_type.value,
            "shape":           self._shape.value,
            "num_rows":        self.num_rows,
        }

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        name = f"'{self.name}'" if self.name else "unnamed"
        ts_type = self.timeseries_type.value if self.timeseries_type else "?"
        return (
            f"TimeSeries({name}, shape={self._shape.value}, "
            f"rows={self.num_rows}, unit='{self.unit}', type={ts_type})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ingest_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a pandas DataFrame into a flat UTC-aware DataFrame.

    1. Flatten any temporal index levels into regular columns.
    2. Ensure every timestamp column is UTC-aware (``datetime64[us, UTC]``).
    3. Return a flat DataFrame (no temporal index).
    """
    # ── 1. Flatten index ────────────────────────────────────────────────────
    if isinstance(df.index, pd.MultiIndex):
        levels_to_reset = [n for n in df.index.names if n in _TIME_COLS]
    else:
        levels_to_reset = [df.index.name] if df.index.name in _TIME_COLS else []

    if levels_to_reset:
        df = df.reset_index(level=levels_to_reset)
    else:
        df = df.copy(deep=False)

    # ── 2. Ensure every timestamp column is UTC-aware ───────────────────────
    for col in _TIME_COLS:
        if col not in df.columns:
            continue

        s = df[col]

        if not pd.api.types.is_datetime64_any_dtype(s):
            warnings.warn(
                f"Column '{col}' is not a datetime type; parsing and converting to UTC.",
                UserWarning,
                stacklevel=3,
            )
            df[col] = pd.to_datetime(s, utc=True)
            continue

        tz = s.dt.tz
        if tz is None:
            warnings.warn(
                f"Column '{col}' has no timezone; assuming UTC.",
                UserWarning,
                stacklevel=3,
            )
            df[col] = s.dt.tz_localize("UTC")
        elif str(tz) != "UTC":
            df[col] = s.dt.tz_convert("UTC")
        # else: already UTC — no allocation

    return df


def _infer_shape(df: pd.DataFrame) -> DataShape:
    """Infer :class:`DataShape` from the column names present in *df*."""
    names = set(df.columns)
    has_kt = "knowledge_time" in names
    has_ct = "change_time" in names
    if has_kt and has_ct:
        return DataShape.AUDIT
    if has_ct:
        return DataShape.CORRECTED
    if has_kt:
        return DataShape.VERSIONED
    return DataShape.SIMPLE


def _validate_table(df: pd.DataFrame, shape: DataShape) -> None:
    """Raise ``ValueError`` if required columns are missing or have wrong type."""
    names = set(df.columns)
    required = _REQUIRED_COLUMNS[shape]
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns for shape {shape.value}: {missing}"
        )

    # Check timestamp columns have a datetime dtype with UTC timezone
    for col in _TIME_COLS:
        if col not in names:
            continue
        s = df[col]
        if not pd.api.types.is_datetime64_any_dtype(s):
            raise TypeError(
                f"Column '{col}' must be a datetime dtype, got {s.dtype!r}"
            )
        tz = s.dt.tz
        if tz is None:
            raise TypeError(
                f"Column '{col}' must be timezone-aware (expected UTC), got tz=None"
            )
