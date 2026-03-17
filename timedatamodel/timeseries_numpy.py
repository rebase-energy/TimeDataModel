"""
TimeSeriesNumpy — a NumPy-backed container for time series data.

Mirrors the public API of :class:`~timedatamodel.timeseries_polars.TimeSeriesPolars`
but stores data internally as a dict of ``numpy.ndarray`` objects rather than a
``polars.DataFrame``.

Internal storage
----------------
``_data`` is a ``Dict[str, np.ndarray]`` with one array per column:

* Timestamp columns (``valid_time``, ``knowledge_time``, ``change_time``) use
  ``np.dtype("datetime64[us]")``.  UTC is assumed; timezone is not encoded in the
  dtype — the ``timezone`` metadata field is a display hint only.
* The ``value`` column uses ``np.float64``.  Missing values are represented as
  ``np.nan``.

Supported shapes follow the same :class:`DataShape` conventions as the Polars
backend:

* **SIMPLE**:     ``{"valid_time": …, "value": …}``
* **VERSIONED**:  ``{"knowledge_time": …, "valid_time": …, "value": …}``
* **CORRECTED**:  ``{"valid_time": …, "change_time": …, "value": …}``
* **AUDIT**:      ``{"knowledge_time": …, "change_time": …, "valid_time": …, "value": …}``

Example usage
-------------
>>> import numpy as np
>>> import pandas as pd
>>> from timedatamodel.timeseries_numpy import TimeSeriesNumpy
>>>
>>> df = pd.DataFrame({
...     "valid_time": pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC"),
...     "value": [1.0, 2.0, 3.0, 4.0],
... })
>>> ts = TimeSeriesNumpy.from_pandas(df, name="wind_power", unit="MW")
>>> ts.num_rows
4
>>> ts.has_missing
False
>>> ts.to_pandas().index.name
'valid_time'
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._base import _get_pint_registry
from ._datashape import DataShape, _REQUIRED_COLUMNS, _TIME_COLS
from ._repr import _TimeSeriesNumpyReprMixin
from .enums import DataType, Frequency, TimeSeriesType
from .location import GeoLocation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NP_DT_DTYPE = np.dtype("datetime64[us]")


# ---------------------------------------------------------------------------
# TimeSeriesNumpy
# ---------------------------------------------------------------------------


class TimeSeriesNumpy(_TimeSeriesNumpyReprMixin):
    """NumPy-backed container for time series data with rich metadata.

    Parameters
    ----------
    data:
        A ``Dict[str, np.ndarray]`` whose keys conform to one of the recognised
        :class:`DataShape` patterns.  Timestamp arrays must use
        ``np.dtype("datetime64[us]")``; the value array must be ``float64``.
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
        Expected data cadence (:class:`~timedatamodel.enums.Frequency`).
    data_type:
        Semantic nature of the observations (:class:`~timedatamodel.enums.DataType`).
    location:
        Optional geographic location (:class:`~timedatamodel.location.GeoLocation`).
    timeseries_type:
        Storage/versioning model (:class:`~timedatamodel.enums.TimeSeriesType`).
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
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
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dict of np.ndarray, got {type(data)!r}")

        shape = _infer_shape_numpy(data)
        _validate_numpy(data, shape)

        self._data: Dict[str, np.ndarray] = data
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
        """Which temporal columns are present (inferred from the data dict)."""
        return self._shape

    @property
    def num_rows(self) -> int:
        """Number of data rows."""
        return len(self._data["value"])

    @property
    def columns(self) -> List[str]:
        """Column names present in the underlying data dict."""
        return list(self._data.keys())

    @property
    def data(self) -> Dict[str, np.ndarray]:
        """The underlying dict of arrays (read-only by convention)."""
        return self._data

    @property
    def has_missing(self) -> bool:
        """True if the ``value`` array contains any ``np.nan``."""
        return bool(np.any(np.isnan(self._data["value"])))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_numpy(
        cls,
        timestamps: np.ndarray,
        values: np.ndarray,
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
    ) -> "TimeSeriesNumpy":
        """Convenience constructor for a **SIMPLE**-shape series from two arrays.

        Parameters
        ----------
        timestamps:
            1-D array of timestamps.  Any dtype convertible to
            ``datetime64[us]`` is accepted.
        values:
            1-D ``float64`` array of observations.  Use ``np.nan`` for
            missing values.
        """
        ts_arr = _ensure_utc_numpy(timestamps)
        val_arr = np.asarray(values, dtype=np.float64)
        return cls(
            {"valid_time": ts_arr, "value": val_arr},
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
    ) -> "TimeSeriesNumpy":
        """Create a :class:`TimeSeriesNumpy` from a ``pandas.DataFrame``.

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
        numpy_data = _ingest_pandas_to_numpy(df)
        shape = _infer_shape_numpy(numpy_data)
        if shape in (DataShape.AUDIT, DataShape.CORRECTED):
            raise ValueError(
                f"from_pandas produced shape {shape.value} because 'change_time' is present. "
                f"Only SIMPLE and VERSIONED shapes can be created via from_pandas. "
                f"Use from_numpy() with the full data dict for CORRECTED/AUDIT."
            )
        return cls(
            numpy_data,
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
        """Validate that this TimeSeriesNumpy can be inserted and return the data dict.

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
                f"TimeSeriesNumpy with shape {self._shape.value} cannot be inserted. "
                f"Only SIMPLE and VERSIONED shapes are supported for insert."
            )
        return self._data, self._shape

    def to_pandas(self) -> pd.DataFrame:
        """Convert to a ``pandas.DataFrame``.

        Timestamp arrays are converted to timezone-aware ``datetime64[us, UTC]``.

        Restores the conventional index:

        * ``SIMPLE``    — ``valid_time`` as index.
        * ``VERSIONED`` — ``(knowledge_time, valid_time)`` MultiIndex.
        * ``AUDIT``     — ``(knowledge_time, change_time, valid_time)`` MultiIndex.
        * ``CORRECTED`` — ``(valid_time, change_time)`` MultiIndex.
        """
        series_dict = {}
        for col, arr in self._data.items():
            if col in _TIME_COLS:
                series_dict[col] = pd.to_datetime(arr, utc=True)
            else:
                series_dict[col] = arr
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
    # Data access helpers
    # ------------------------------------------------------------------

    def head(self, n: int = 5) -> "TimeSeriesNumpy":
        """Return the first *n* rows as a new :class:`TimeSeriesNumpy`."""
        return self._clone({k: v[:n] for k, v in self._data.items()})

    def tail(self, n: int = 5) -> "TimeSeriesNumpy":
        """Return the last *n* rows as a new :class:`TimeSeriesNumpy`."""
        return self._clone({k: v[-n:] for k, v in self._data.items()})

    # ------------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------------

    def convert_unit(self, target_unit: str) -> "TimeSeriesNumpy":
        """Return a new :class:`TimeSeriesNumpy` with values converted to *target_unit*.

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
        ureg = _get_pint_registry()
        factor = float(ureg.Quantity(1.0, self.unit).to(target_unit).magnitude)
        new_data = {**self._data, "value": self._data["value"] * factor}
        return self._clone(new_data, unit=target_unit)

    # ------------------------------------------------------------------
    # Internal clone helper
    # ------------------------------------------------------------------

    def _clone(self, new_data: Dict[str, np.ndarray], **overrides) -> "TimeSeriesNumpy":
        """Create a new :class:`TimeSeriesNumpy` with *new_data* and the same metadata.

        Any keyword in *overrides* replaces the corresponding metadata field.
        """
        return TimeSeriesNumpy(
            new_data,
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
        return len(self._data["value"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_utc_numpy(timestamps) -> np.ndarray:
    """Normalise *timestamps* to UTC and return a ``datetime64[us]`` array.

    Mirrors the UTC enforcement in :func:`_ingest_pandas_to_numpy`:

    * Timezone-aware, non-UTC → converted to UTC.
    * Timezone-naive Python datetimes → assumed UTC with a warning.
    * Already UTC or already ``datetime64`` without tz → passed through.
    """
    s = pd.to_datetime(timestamps)
    tz = getattr(s, "tz", None)
    if tz is None:
        # Check whether the source objects carry tzinfo (Python datetimes)
        src = list(timestamps) if not isinstance(timestamps, (list, np.ndarray)) else timestamps
        first = src[0] if len(src) else None
        if hasattr(first, "tzinfo") and first.tzinfo is not None:
            # tz-aware datetimes but pd.to_datetime lost it somehow — re-parse
            s = pd.to_datetime(timestamps, utc=True)
        else:
            warnings.warn(
                "Timestamps have no timezone; assuming UTC.",
                UserWarning,
                stacklevel=3,
            )
            s = s.tz_localize("UTC")
    elif str(tz) != "UTC":
        warnings.warn(
            f"Timestamps are not UTC (got {tz!r}); converting to UTC.",
            UserWarning,
            stacklevel=3,
        )
        s = s.tz_convert("UTC")
    return s.values.astype(_NP_DT_DTYPE)


def _ingest_pandas_to_numpy(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Ingest a pandas DataFrame into a dict of NumPy arrays.

    1. Flatten any temporal index levels into regular columns.
    2. Normalize timestamp columns to UTC-aware ``datetime64[us]``.
    3. Convert to NumPy arrays.
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

    # ── 3. Convert to NumPy arrays ───────────────────────────────────────────
    result: Dict[str, np.ndarray] = {}
    for col in df.columns:
        if col in _TIME_COLS:
            result[col] = df[col].values.astype(_NP_DT_DTYPE)
        else:
            result[col] = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
    return result


def _infer_shape_numpy(data: Dict[str, np.ndarray]) -> DataShape:
    """Infer :class:`DataShape` from the keys present in *data*."""
    keys = set(data.keys())
    has_kt = "knowledge_time" in keys
    has_ct = "change_time" in keys
    if has_kt and has_ct:
        return DataShape.AUDIT
    if has_ct:
        return DataShape.CORRECTED
    if has_kt:
        return DataShape.VERSIONED
    return DataShape.SIMPLE


def _validate_numpy(data: Dict[str, np.ndarray], shape: DataShape) -> None:
    """Raise ``ValueError`` if required columns are missing or have wrong dtype."""
    keys = set(data.keys())
    required = _REQUIRED_COLUMNS[shape]
    missing = [c for c in required if c not in keys]
    if missing:
        raise ValueError(
            f"data dict is missing required keys for shape {shape.value}: {missing}"
        )

    for col in _TIME_COLS:
        if col not in keys:
            continue
        arr = data[col]
        if not np.issubdtype(arr.dtype, np.datetime64):
            raise TypeError(
                f"Array '{col}' must have a datetime64 dtype, got {arr.dtype!r}"
            )
