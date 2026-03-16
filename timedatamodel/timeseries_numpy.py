"""
TimeSeries — a NumPy-backed container for time series data.

This module mirrors ``timeseries_arrow.py`` but uses a
``dict[str, numpy.ndarray]`` as the internal storage instead of a
``pyarrow.Table``.

Data shapes
-----------
See ``timeseries_arrow.py`` for the full description of the four shapes
(SIMPLE, VERSIONED, CORRECTED, AUDIT).  The same column conventions apply
here.

Timestamp representation
------------------------
All timestamp arrays are stored as ``numpy.datetime64[us]`` (microsecond
precision).  NumPy does not natively support timezone information; UTC is
always assumed.  The ``timezone`` metadata field is a display/context hint
(IANA zone string).

When converting back to pandas via ``to_pandas()``, timestamp arrays are
converted to ``datetime64[us, UTC]`` so that downstream code sees
timezone-aware timestamps.

Example usage
-------------
>>> import pandas as pd
>>> from timedatamodel.timeseries_numpy import TimeSeries
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

import numpy as np
import pandas as pd

from .enums import DataType, TimeSeriesType
from .location import GeoLocation
from .timeseries_arrow import DataShape, _REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TIME_COLS: frozenset = frozenset({"valid_time", "valid_time_end", "knowledge_time", "change_time"})
_TS_DTYPE = np.dtype("datetime64[us]")
_STRING_COLS: frozenset = frozenset({"changed_by", "annotation"})


# ---------------------------------------------------------------------------
# TimeSeries
# ---------------------------------------------------------------------------

class TimeSeries:
    """NumPy-backed container for time series data with rich metadata.

    Internally stores data as a ``dict[str, numpy.ndarray]`` where:

    * Timestamp columns (``valid_time``, etc.) are stored as
      ``numpy.datetime64[us]`` arrays.  UTC is always assumed — NumPy has no
      native timezone support.
    * The ``value`` column is stored as ``numpy.float64`` (``NaN`` for nulls).
    * String columns (``changed_by``, ``annotation``) are stored as
      ``object``-dtype arrays.

    Parameters
    ----------
    arrays:
        Mapping of column name → ``numpy.ndarray``.  All arrays must have
        the same length.  Must conform to one of the recognised
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
        arrays: Dict[str, np.ndarray],
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
        if not isinstance(arrays, dict):
            raise TypeError(f"arrays must be a dict[str, np.ndarray], got {type(arrays)!r}")

        # Validate all values are ndarrays and lengths are consistent
        lengths = {k: len(v) for k, v in arrays.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"All arrays must have the same length. Got: {lengths}"
            )

        shape = _infer_shape(arrays)
        _validate_arrays(arrays, shape)

        self._arrays: Dict[str, np.ndarray] = arrays
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
        """Which temporal columns are present (inferred from the arrays dict)."""
        return self._shape

    @property
    def num_rows(self) -> int:
        """Number of data rows."""
        if not self._arrays:
            return 0
        return len(next(iter(self._arrays.values())))

    @property
    def columns(self) -> List[str]:
        """Column names present in the underlying arrays dict."""
        return list(self._arrays.keys())

    @property
    def arrays(self) -> Dict[str, np.ndarray]:
        """The underlying ``dict[str, np.ndarray]`` (read-only by convention)."""
        return self._arrays

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_numpy(
        cls,
        arrays: Dict[str, np.ndarray],
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
        """Create a :class:`TimeSeries` directly from a
        ``dict[str, numpy.ndarray]``.

        Timestamp arrays must already be ``numpy.datetime64[us]`` (UTC
        assumed).  Value array must be ``numpy.float64``.
        """
        return cls(
            arrays,
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

        Only ``SIMPLE`` and ``VERSIONED`` shapes can be constructed via
        ``from_pandas``.  ``AUDIT`` and ``CORRECTED`` shapes (which require a
        ``change_time`` column) are read-only results from the database layer.

        Raises
        ------
        ValueError
            If the DataFrame contains a ``change_time`` column.
        """
        arrays = _ingest_pandas_to_numpy(df)
        shape = _infer_shape(arrays)
        if shape in (DataShape.AUDIT, DataShape.CORRECTED):
            raise ValueError(
                f"from_pandas produced shape {shape.value} because 'change_time' is present. "
                f"Only SIMPLE and VERSIONED shapes can be created via from_pandas. "
                f"Use from_numpy() to wrap an existing read result with change_time."
            )
        return cls(
            arrays,
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

    def validate_for_insert(self) -> "Tuple[Dict[str, np.ndarray], DataShape]":
        """Validate that this TimeSeries can be inserted and return the underlying
        arrays dict with its shape.

        Only :attr:`DataShape.SIMPLE` and :attr:`DataShape.VERSIONED` are
        supported for insert.

        Returns
        -------
        Tuple[Dict[str, np.ndarray], DataShape]

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
        return self._arrays, self._shape

    def to_pandas(self) -> pd.DataFrame:
        """Convert to a ``pandas.DataFrame`` with the conventional index.

        Timestamp columns are converted from ``numpy.datetime64[us]`` to
        timezone-aware ``datetime64[us, UTC]`` pandas Series.

        * ``SIMPLE``    — ``valid_time`` as index.
        * ``VERSIONED`` — ``(knowledge_time, valid_time)`` MultiIndex.
        * ``AUDIT``     — ``(knowledge_time, change_time, valid_time)`` MultiIndex.
        * ``CORRECTED`` — ``(valid_time, change_time)`` MultiIndex.
        """
        series_dict: Dict[str, pd.Series] = {}
        for col, arr in self._arrays.items():
            if col in _TIME_COLS:
                # Convert numpy datetime64[us] → pandas UTC-aware Series
                series_dict[col] = pd.to_datetime(arr, utc=True)
            else:
                series_dict[col] = pd.Series(arr)

        df = pd.DataFrame(series_dict)

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
        return self.num_rows

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


def _ingest_pandas_to_numpy(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Ingest a pandas DataFrame into a ``dict[str, np.ndarray]``.

    1. Flatten any temporal index levels into regular columns.
    2. Normalize timestamp columns to UTC-aware pandas Series.
    3. Convert each column to a numpy array:
       - Timestamp cols → ``datetime64[us]``
       - ``value`` col  → ``float64`` (NaN for nulls)
       - String cols    → ``object`` dtype
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

    # ── 3. Convert columns to numpy arrays ──────────────────────────────────
    arrays: Dict[str, np.ndarray] = {}
    for col in df.columns:
        s = df[col]
        if col in _TIME_COLS:
            # Strip timezone info: numpy datetime64 has no tz, UTC is assumed.
            arrays[col] = s.to_numpy(dtype=_TS_DTYPE)
        elif col == "value":
            arrays[col] = s.to_numpy(dtype=np.float64)
        elif col in _STRING_COLS:
            arrays[col] = s.to_numpy(dtype=object)
        else:
            # Unknown extra columns: convert as-is
            arrays[col] = s.to_numpy()

    return arrays


def _infer_shape(arrays: Dict[str, np.ndarray]) -> DataShape:
    """Infer :class:`DataShape` from the keys present in *arrays*."""
    names = set(arrays.keys())
    has_kt = "knowledge_time" in names
    has_ct = "change_time" in names
    if has_kt and has_ct:
        return DataShape.AUDIT
    if has_ct:
        return DataShape.CORRECTED
    if has_kt:
        return DataShape.VERSIONED
    return DataShape.SIMPLE


def _validate_arrays(arrays: Dict[str, np.ndarray], shape: DataShape) -> None:
    """Raise ``ValueError`` if required columns are missing or have wrong dtype."""
    names = set(arrays.keys())
    required = _REQUIRED_COLUMNS[shape]
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(
            f"arrays dict is missing required columns for shape {shape.value}: {missing}"
        )

    # Check timestamp columns have datetime64 dtype
    for col in _TIME_COLS:
        if col not in names:
            continue
        arr = arrays[col]
        if not np.issubdtype(arr.dtype, np.datetime64):
            raise TypeError(
                f"Column '{col}' must be a numpy datetime64 dtype, got {arr.dtype!r}"
            )
