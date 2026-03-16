"""
TimeSeriesPolars — a Polars-backed container for time series data.

This module mirrors ``timeseries_arrow.py`` but uses a ``polars.DataFrame``
as the internal storage instead of a ``pyarrow.Table``.

Data shapes
-----------
See ``timeseries_arrow.py`` for the full description of the four shapes
(SIMPLE, VERSIONED, CORRECTED, AUDIT).  The same column conventions apply
here.

Timestamp representation
------------------------
All timestamp columns are stored internally as ``pl.Datetime("us", time_zone="UTC")``.
The ``timezone`` metadata field is a display/context hint (IANA zone string).

Example usage
-------------
>>> import pandas as pd
>>> from timedatamodel.timeseries_polars import TimeSeriesPolars
>>> from timedatamodel.enums import DataType
>>>
>>> df = pd.DataFrame({
...     "valid_time": pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC"),
...     "value": [1.0, 2.0, 3.0, 4.0],
... })
>>> ts = TimeSeriesPolars.from_pandas(df, name="wind_power", unit="MW")
>>> ts
TimeSeriesPolars('wind_power', shape=SIMPLE, rows=4, unit='MW', type=flat)
>>> ts.to_pandas()
                           value
valid_time
2024-01-01 00:00:00+00:00    1.0
...
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd
import polars as pl

from ._base import _get_pint_registry
from ._repr import _TimeSeriesPolarsReprMixin
from .enums import DataType, Frequency, TimeSeriesType
from .location import GeoLocation


# ---------------------------------------------------------------------------
# DataShape — shared with timeseries_arrow (defined here to avoid pyarrow dep)
# ---------------------------------------------------------------------------


class DataShape(str, Enum):
    """Which temporal columns are present in the underlying data store."""

    SIMPLE     = "SIMPLE"     # valid_time + value
    VERSIONED  = "VERSIONED"  # knowledge_time + valid_time + value
    AUDIT      = "AUDIT"      # knowledge_time + change_time + valid_time + value
    CORRECTED  = "CORRECTED"  # valid_time + change_time + value


#: Required columns per shape.
_REQUIRED_COLUMNS: Dict[DataShape, List[str]] = {
    DataShape.SIMPLE:     ["valid_time", "value"],
    DataShape.VERSIONED:  ["knowledge_time", "valid_time", "value"],
    DataShape.AUDIT:      ["knowledge_time", "change_time", "valid_time", "value"],
    DataShape.CORRECTED:  ["valid_time", "change_time", "value"],
}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TIME_COLS: frozenset = frozenset({"valid_time", "valid_time_end", "knowledge_time", "change_time"})
_TS_DTYPE = pl.Datetime("us", time_zone="UTC")


# ---------------------------------------------------------------------------
# TimeSeriesPolars
# ---------------------------------------------------------------------------

class TimeSeriesPolars(_TimeSeriesPolarsReprMixin):
    """Polars-backed container for time series data with rich metadata.

    Parameters
    ----------
    df:
        A ``polars.DataFrame`` whose columns conform to one of the recognised
        :class:`~timedatamodel.timeseries_polars.DataShape` patterns.  All
        timestamp columns must use ``pl.Datetime("us", time_zone="UTC")``.
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
        df: pl.DataFrame,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        unit: str = "dimensionless",
        labels: Optional[Dict[str, str]] = None,
        timezone: str = "UTC",
        frequency: Optional[Frequency] = None,
        data_type: Optional[DataType] = None,
        location: Optional[GeoLocation] = None,
        timeseries_type: TimeSeriesType = TimeSeriesType.FLAT,
    ) -> None:
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"df must be a polars.DataFrame, got {type(df)!r}")

        shape = _infer_shape(df)
        _validate_table(df, shape)

        self._df: pl.DataFrame = df
        self._shape: DataShape = shape

        self.name: Optional[str] = name
        self.description: Optional[str] = description
        self.unit: str = unit
        self.labels: Dict[str, str] = labels or {}
        self.timezone: str = timezone
        self.frequency: Optional[Frequency] = frequency
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
        return self._df.height

    @property
    def columns(self) -> List[str]:
        """Column names present in the underlying Polars DataFrame."""
        return self._df.columns

    @property
    def df(self) -> pl.DataFrame:
        """The underlying ``polars.DataFrame`` (read-only by convention)."""
        return self._df

    @property
    def has_missing(self) -> bool:
        """True if the ``value`` column contains any null values."""
        return self._df["value"].is_null().any()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_polars(
        cls,
        df: pl.DataFrame,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        unit: str = "dimensionless",
        labels: Optional[Dict[str, str]] = None,
        timezone: str = "UTC",
        frequency: Optional[Frequency] = None,
        data_type: Optional[DataType] = None,
        location: Optional[GeoLocation] = None,
        timeseries_type: TimeSeriesType = TimeSeriesType.FLAT,
    ) -> "TimeSeriesPolars":
        """Create a :class:`TimeSeriesPolars` directly from a ``polars.DataFrame``.

        All timestamp columns must already use
        ``pl.Datetime("us", time_zone="UTC")``.
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
        frequency: Optional[Frequency] = None,
        data_type: Optional[DataType] = None,
        location: Optional[GeoLocation] = None,
        timeseries_type: TimeSeriesType = TimeSeriesType.FLAT,
    ) -> "TimeSeriesPolars":
        """Create a :class:`TimeSeriesPolars` from a ``pandas.DataFrame``.

        Only ``SIMPLE`` and ``VERSIONED`` shapes can be constructed via
        ``from_pandas``.  ``AUDIT`` and ``CORRECTED`` shapes (which require a
        ``change_time`` column) are read-only results from the database layer.

        The data shape is inferred from the column names (and MultiIndex levels
        if the DataFrame uses an index).

        Raises
        ------
        ValueError
            If the DataFrame contains a ``change_time`` column.
        """
        polars_df = _ingest_pandas_to_polars(df)
        shape = _infer_shape(polars_df)
        if shape in (DataShape.AUDIT, DataShape.CORRECTED):
            raise ValueError(
                f"from_pandas produced shape {shape.value} because 'change_time' is present. "
                f"Only SIMPLE and VERSIONED shapes can be created via from_pandas. "
                f"Use from_polars() to wrap an existing read result with change_time."
            )
        return cls(
            polars_df,
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

    def validate_for_insert(self) -> "Tuple[pl.DataFrame, DataShape]":
        """Validate that this TimeSeriesPolars can be inserted and return the underlying
        DataFrame with its shape.

        Only :attr:`DataShape.SIMPLE` and :attr:`DataShape.VERSIONED` are
        supported for insert.

        Returns
        -------
        Tuple[pl.DataFrame, DataShape]

        Raises
        ------
        ValueError
            If :attr:`shape` is :attr:`DataShape.AUDIT` or
            :attr:`DataShape.CORRECTED`.
        """
        if self._shape in (DataShape.AUDIT, DataShape.CORRECTED):
            raise ValueError(
                f"TimeSeriesPolars with shape {self._shape.value} cannot be inserted. "
                f"Only SIMPLE and VERSIONED shapes are supported for insert."
            )
        return self._df, self._shape

    def to_pandas(self) -> pd.DataFrame:
        """Convert to a ``pandas.DataFrame``.

        Restores the conventional index:

        * ``SIMPLE``    — ``valid_time`` as index.
        * ``VERSIONED`` — ``(knowledge_time, valid_time)`` MultiIndex.
        * ``AUDIT``     — ``(knowledge_time, change_time, valid_time)`` MultiIndex.
        * ``CORRECTED`` — ``(valid_time, change_time)`` MultiIndex.
        """
        df = self._df.to_pandas()

        # Polars converts Datetime("us", tz="UTC") to pandas datetime64[us, UTC]
        # which is exactly what we want.

        if self._shape == DataShape.SIMPLE:
            return df.set_index("valid_time")
        elif self._shape == DataShape.VERSIONED:
            return df.set_index(["knowledge_time", "valid_time"])
        elif self._shape == DataShape.AUDIT:
            return df.set_index(["knowledge_time", "change_time", "valid_time"])
        elif self._shape == DataShape.CORRECTED:
            return df.set_index(["valid_time", "change_time"])
        return df  # unreachable, safe fallback

    # ------------------------------------------------------------------
    # Data access helpers
    # ------------------------------------------------------------------

    def head(self, n: int = 5) -> "TimeSeriesPolars":
        """Return the first *n* rows as a new :class:`TimeSeriesPolars`."""
        return self._clone(self._df.head(n))

    def tail(self, n: int = 5) -> "TimeSeriesPolars":
        """Return the last *n* rows as a new :class:`TimeSeriesPolars`."""
        return self._clone(self._df.tail(n))

    # ------------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------------

    def convert_unit(self, target_unit: str) -> "TimeSeriesPolars":
        """Return a new :class:`TimeSeriesPolars` with values converted to *target_unit*.

        Uses the pint library for unit conversion.  The ``unit`` metadata field
        is updated to *target_unit*.

        Parameters
        ----------
        target_unit:
            Target unit string understood by pint (e.g. ``"km/h"``, ``"kW"``).

        Raises
        ------
        ImportError
            If pint is not installed.
        pint.DimensionalityError
            If the current unit and *target_unit* are dimensionally incompatible.
        """
        try:
            ureg = _get_pint_registry()
        except ImportError as exc:
            raise ImportError(
                "Unit conversion requires the optional 'pint' dependency. "
                "Install it with: pip install timedatamodel[pint]"
            ) from exc
        factor = float(ureg.Quantity(1.0, self.unit).to(target_unit).magnitude)
        new_df = self._df.with_columns(pl.col("value") * factor)
        return self._clone(new_df, unit=target_unit)

    # ------------------------------------------------------------------
    # Internal clone helper
    # ------------------------------------------------------------------

    def _clone(self, new_df: pl.DataFrame, **overrides) -> "TimeSeriesPolars":
        """Create a new :class:`TimeSeriesPolars` with *new_df* and the same metadata.

        Any keyword in *overrides* replaces the corresponding metadata field.
        """
        return TimeSeriesPolars(
            new_df,
            name=overrides.get("name", self.name),
            description=overrides.get("description", self.description),
            unit=overrides.get("unit", self.unit),
            labels=overrides.get("labels", self.labels),
            timezone=overrides.get("timezone", self.timezone),
            frequency=overrides.get("frequency", self.frequency),
            data_type=overrides.get("data_type", self.data_type),
            location=overrides.get("location", self.location),
            timeseries_type=overrides.get("timeseries_type", self.timeseries_type),
        )

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
        return self._df.height


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ingest_pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """Ingest a pandas DataFrame into a ``pl.DataFrame``.

    1. Flatten any temporal index levels into regular columns.
    2. Normalize timestamp columns to UTC-aware pandas Series.
    3. Convert to Polars and cast timestamp columns to
       ``pl.Datetime("us", time_zone="UTC")``.
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

    # ── 3. Convert to Polars and cast timestamp columns ─────────────────────
    polars_df = pl.from_pandas(df)

    cast_exprs = [
        pl.col(c).cast(_TS_DTYPE)
        for c in _TIME_COLS
        if c in polars_df.columns
    ]
    if cast_exprs:
        polars_df = polars_df.with_columns(cast_exprs)

    return polars_df


def _infer_shape(df: pl.DataFrame) -> DataShape:
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


def _validate_table(df: pl.DataFrame, shape: DataShape) -> None:
    """Raise ``ValueError`` if required columns are missing or have wrong type."""
    names = set(df.columns)
    required = _REQUIRED_COLUMNS[shape]
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns for shape {shape.value}: {missing}"
        )

    # Check timestamp columns have the right dtype
    for col in _TIME_COLS:
        if col not in names:
            continue
        dtype = df[col].dtype
        if not isinstance(dtype, pl.Datetime):
            raise TypeError(
                f"Column '{col}' must be a Polars Datetime type, got {dtype!r}"
            )
        if dtype.time_zone is None:
            raise TypeError(
                f"Column '{col}' must be timezone-aware with time_zone='UTC', "
                f"got time_zone=None"
            )
        if dtype.time_zone != "UTC":
            raise TypeError(
                f"Column '{col}' must have time_zone='UTC', "
                f"got time_zone={dtype.time_zone!r}"
            )
