"""
TimeSeries — a PyArrow-backed container for time series data.

This module provides a standalone ``TimeSeries`` class that holds both the
time series data (as a ``pyarrow.Table``) and the associated metadata.  It is
*not* yet integrated with the rest of timedb; integration will be done later.

Data shapes
-----------
The class supports four shapes that mirror the read patterns already used in
timedb:

SIMPLE
    One observation per ``valid_time``.  Covers flat (immutable) series and
    the "latest value" projection of overlapping series.

    Columns: ``valid_time``, [``valid_time_end``], ``value``

VERSIONED
    One row per ``(knowledge_time, valid_time)`` pair.  Covers the full
    forecast-revision history of an overlapping series.

    Columns: ``knowledge_time``, ``valid_time``, [``valid_time_end``], ``value``

CORRECTED
    One row per ``(valid_time, change_time)`` pair. Covers historical 
    corrections to flat series (e.g., updated meter readings or sensor 
    re-calibrations) where forecast generation time is not applicable.

    Columns: ``valid_time``, ``change_time``, [``valid_time_end``], ``value``, 
             [``changed_by``], [``annotation``]

AUDIT
    Full bi-temporal log — every correction to every forecast run.

    Columns: ``knowledge_time``, ``change_time``, ``valid_time``,
             [``valid_time_end``], ``value``, [``changed_by``],
             [``annotation``]

Timestamp representation
------------------------
All timestamp columns are stored internally as ``pa.timestamp('us', tz='UTC')``
regardless of the input timezone.  The ``timezone`` metadata field is a
display/context hint (IANA zone string) for the user's local domain.

Example usage
-------------
>>> import pandas as pd
>>> from timedatamodel.timeseries_arrow import TimeSeries, DataShape
>>> from timedatamodel.enums import DataType
>>> from timedatamodel.location import GeoLocation
>>>
>>> df = pd.DataFrame({
...     "valid_time": pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC"),
...     "value": [1.0, 2.0, 3.0, 4.0],
... })
>>> ts = TimeSeries.from_pandas(
...     df,
...     name="wind_power",
...     unit="MW",
...     frequency="1h",
...     data_type=DataType.OBSERVATION,
...     location=GeoLocation(latitude=59.33, longitude=18.07),
... )
>>> print(ts)
TimeSeries('wind_power', shape=SIMPLE, rows=4, unit='MW', type=flat)
>>> ts.to_pandas()
                           value
valid_time
2024-01-01 00:00:00+00:00    1.0
...
"""

from __future__ import annotations

import time
import warnings
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa

try:
    from timedb import profiling as _profiling
except ImportError:
    _profiling = None

from ._repr import _TimeSeriesArrowReprMixin
from .enums import DataType, TimeSeriesType
from .location import GeoLocation


# ---------------------------------------------------------------------------
# Helper types
# ---------------------------------------------------------------------------

class DataShape(str, Enum):
    """Which temporal columns are present in the underlying Arrow table."""

    SIMPLE     = "SIMPLE"     # valid_time + value
    VERSIONED  = "VERSIONED"  # knowledge_time + valid_time + value
    AUDIT      = "AUDIT"      # knowledge_time + change_time + valid_time + value + audit cols
    CORRECTED  = "CORRECTED"  # valid_time + change_time + value + audit cols (no knowledge_time)


# ---------------------------------------------------------------------------
# Arrow schemas
# ---------------------------------------------------------------------------

_TS_TYPE = pa.timestamp("us", tz="UTC")

#: Required columns per shape (optional columns are declared separately).
_REQUIRED_COLUMNS: Dict[DataShape, List[str]] = {
    DataShape.SIMPLE:     ["valid_time", "value"],
    DataShape.VERSIONED:  ["knowledge_time", "valid_time", "value"],
    DataShape.AUDIT:      ["knowledge_time", "change_time", "valid_time", "value"],
    DataShape.CORRECTED:  ["valid_time", "change_time", "value"],
}

#: Canonical Arrow field definitions (all shapes share these).
_FIELD_DEFS: Dict[str, pa.Field] = {
    "valid_time":     pa.field("valid_time",     _TS_TYPE,      nullable=False),
    "valid_time_end": pa.field("valid_time_end", _TS_TYPE,      nullable=True),
    "knowledge_time": pa.field("knowledge_time", _TS_TYPE,      nullable=False),
    "change_time":    pa.field("change_time",    _TS_TYPE,      nullable=False),
    "value":          pa.field("value",          pa.float64(),  nullable=True),
    "changed_by":     pa.field("changed_by",     pa.string(),   nullable=True),
    "annotation":     pa.field("annotation",     pa.string(),   nullable=True),
}


def _schema_for(shape: DataShape) -> pa.Schema:
    """Return the canonical Arrow schema for a given shape.

    Optional columns (``valid_time_end``, ``changed_by``, ``annotation``) are
    *not* included here; they may be present or absent in actual tables.
    """
    required = _REQUIRED_COLUMNS[shape]
    return pa.schema([_FIELD_DEFS[c] for c in required])


# ---------------------------------------------------------------------------
# TimeSeries
# ---------------------------------------------------------------------------

class TimeSeries(_TimeSeriesArrowReprMixin):
    """PyArrow-backed container for time series data with rich metadata.

    Parameters
    ----------
    table:
        A ``pyarrow.Table`` whose columns conform to one of the recognised
        :class:`DataShape` patterns.  All timestamp columns must use
        ``pa.timestamp('us', tz='UTC')``.
    name:
        Series name (e.g. ``"wind_power"``).
    description:
        Human-readable description.
    unit:
        Canonical physical unit string (e.g. ``"MW"``, ``"dimensionless"``).
    labels:
        Arbitrary key-value labels for series differentiation
        (e.g. ``{"site": "Gotland", "turbine": "T01"}``).
    timezone:
        IANA timezone string for display purposes (e.g. ``"Europe/Stockholm"``).
        Internal data is always UTC; this is a metadata hint only.
    frequency:
        Pandas offset alias describing the expected data cadence
        (e.g. ``"1h"``, ``"15min"``, ``"D"``).  ``None`` means irregular.
    data_type:
        Semantic nature of the observations (:class:`DataType`).
    location:
        Optional geographic location (:class:`GeoLocation`).
    timeseries_type:
        Storage/versioning model (:class:`TimeSeriesType`).
    """

    def __init__(
        self,
        table: pa.Table,
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
        if not isinstance(table, pa.Table):
            raise TypeError(f"table must be a pyarrow.Table, got {type(table)!r}")

        _t0 = time.perf_counter() if (_profiling and _profiling._enabled) else 0.0

        shape = _infer_shape(table)
        _validate_table(table, shape)

        self._table: pa.Table = table
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

        if _profiling and _profiling._enabled:
            _profiling._record(_profiling.PHASE_READ_BUILD_TIMESERIES, time.perf_counter() - _t0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> DataShape:
        """Which temporal columns are present (inferred from the table)."""
        return self._shape

    @property
    def num_rows(self) -> int:
        """Number of data rows."""
        return self._table.num_rows

    @property
    def columns(self) -> List[str]:
        """Column names present in the underlying Arrow table."""
        return self._table.schema.names

    @property
    def table(self) -> pa.Table:
        """The underlying ``pyarrow.Table`` (read-only by convention)."""
        return self._table

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

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
        ``from_pandas``.  This method is intended for building objects that
        will be inserted into the database.  ``AUDIT`` and ``CORRECTED``
        shapes (which require a ``change_time`` column) are read-only results
        produced by the database layer and cannot be constructed here.

        The data shape is **inferred automatically** from the column names
        (and MultiIndex levels if the DataFrame uses an index):

        * ``VERSIONED`` — if ``knowledge_time`` is present.
        * ``SIMPLE``    — otherwise.

        Timestamp columns may be in the index, in regular columns, or a mix.
        All timestamp columns are converted to UTC internally.

        Parameters
        ----------
        df:
            Input DataFrame.  Must contain at least ``valid_time`` and
            ``value``.  These may appear as regular columns *or* as index
            levels.  Must **not** contain a ``change_time`` column — use
            :meth:`from_arrow` to wrap an existing read result instead.

        Raises
        ------
        ValueError
            If the DataFrame contains a ``change_time`` column, which would
            produce an ``AUDIT`` or ``CORRECTED`` shape.
        """
        table = _ingest_pandas_to_arrow(df)
        shape = _infer_shape(table)
        if shape in (DataShape.AUDIT, DataShape.CORRECTED):
            raise ValueError(
                f"from_pandas produced shape {shape.value} because 'change_time' is present. "
                f"Only SIMPLE and VERSIONED shapes can be created via from_pandas. "
                f"AUDIT and CORRECTED shapes are read-only results from the database "
                f"layer — use from_arrow() to wrap an existing read result."
            )
        return cls(
            table,
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
    def from_arrow(
        cls,
        table: pa.Table,
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
        """Create a :class:`TimeSeries` directly from a ``pyarrow.Table``.

        This is the preferred path when data is already in Arrow format
        (e.g. read directly from the database via ``adbc``/``cursor.fetch_arrow_table()``).
        All timestamp columns must already use ``pa.timestamp('us', tz='UTC')``.
        """
        return cls(
            table,
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

    def validate_for_insert(self) -> "Tuple[pa.Table, DataShape]":
        """Validate that this TimeSeries can be inserted and return the underlying table with its shape.

        Only :attr:`DataShape.SIMPLE` and :attr:`DataShape.VERSIONED` are
        supported for insert.  ``AUDIT`` and ``CORRECTED`` shapes cannot be
        directly inserted.

        * ``SIMPLE``    — ``[valid_time, (valid_time_end,) value]``
        * ``VERSIONED`` — ``[knowledge_time, valid_time, (valid_time_end,) value]``

        Returns:
            A ``(pa.Table, DataShape)`` tuple.

        Raises:
            ValueError: If :attr:`shape` is :attr:`DataShape.AUDIT` or
                :attr:`DataShape.CORRECTED`.
        """
        if self._shape in (DataShape.AUDIT, DataShape.CORRECTED):
            raise ValueError(
                f"TimeSeries with shape {self._shape.value} cannot be inserted. "
                f"Only SIMPLE and VERSIONED shapes are supported for insert."
            )
        return self._table, self._shape

    def to_pandas(self) -> pd.DataFrame:
        """Convert to a ``pandas.DataFrame``.

        Restores the conventional index used in the existing timedb read API:

        * ``SIMPLE``    — ``valid_time`` as index.
        * ``VERSIONED`` — ``(knowledge_time, valid_time)`` MultiIndex.
        * ``AUDIT``     — ``(knowledge_time, change_time, valid_time)``
          MultiIndex.
        * ``CORRECTED`` — ``(valid_time, change_time)`` MultiIndex.
        """
        _t0 = time.perf_counter() if (_profiling and _profiling._enabled) else 0.0

        df = self._table.to_pandas(timestamp_as_object=False)

        if self._shape == DataShape.SIMPLE:
            result = df.set_index("valid_time")
        elif self._shape == DataShape.VERSIONED:
            result = df.set_index(["knowledge_time", "valid_time"])
        elif self._shape == DataShape.AUDIT:
            result = df.set_index(["knowledge_time", "change_time", "valid_time"])
        elif self._shape == DataShape.CORRECTED:
            result = df.set_index(["valid_time", "change_time"])
        else:
            result = df  # unreachable, but safe fallback

        if _profiling and _profiling._enabled:
            _profiling._record(_profiling.PHASE_READ_TO_PANDAS, time.perf_counter() - _t0)

        return result

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def metadata_dict(self) -> Dict:
        """Return all metadata fields as a plain dict.

        Useful for serialisation, display, or passing to other systems.
        """
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
        return self._table.num_rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TIME_COLS: frozenset = frozenset({"valid_time", "valid_time_end", "knowledge_time", "change_time"})


def _ingest_pandas_to_arrow(df: pd.DataFrame) -> pa.Table:
    """Ingest a pandas DataFrame into a ``pa.Table`` in a single optimised pass.

    Compared to the previous three-helper approach this function:

    * Avoids O(rows) full DataFrame copies.  A ``deep=False`` copy (O(columns))
      is only taken when no ``reset_index`` was needed, to avoid mutating the
      caller's object.  Column reassignments only allocate new memory for the
      specific timestamp columns that actually need coercion.
    * Short-circuits timestamp columns that are already UTC (no redundant
      ``tz_convert`` allocation).
    * Infers all non-timestamp field types from pandas in one call via
      ``pa.Schema.from_pandas``, then overrides only the timestamp fields.
      Passing the complete target schema to ``pa.Table.from_pandas`` lets
      Arrow's multithreaded C++ engine coerce every column in a single sweep.
    """
    # ── 1. Flatten index — no redundant copy ────────────────────────────────
    if isinstance(df.index, pd.MultiIndex):
        levels_to_reset = [n for n in df.index.names if n in _TIME_COLS]
    else:
        levels_to_reset = [df.index.name] if df.index.name in _TIME_COLS else []

    if levels_to_reset:
        # reset_index() already returns a new DataFrame — safe to mutate below.
        df = df.reset_index(level=levels_to_reset)
    else:
        # Shallow copy: O(number-of-columns), not O(rows).
        # Shares underlying column buffers; only reassigned columns get new RAM.
        df = df.copy(deep=False)

    # ── 2. Ensure every timestamp column is UTC-aware ───────────────────────
    for col in _TIME_COLS:
        if col not in df.columns:
            continue
            
        s = df[col]
        
        # Fast Path for Strings / Objects
        if not pd.api.types.is_datetime64_any_dtype(s):
            warnings.warn(
                f"Column '{col}' is not a datetime type; parsing and converting to UTC.",
                UserWarning,
                stacklevel=3,
            )
            # Parses strings and converts to UTC in a single C-level pass.
            # Directly updates the dataframe, fixing the assignment bug!
            df[col] = pd.to_datetime(s, utc=True)
            continue

        # Fast Path for Existing Datetimes
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
        # else: already UTC — no assignment, no allocation

    # ── 3. Single-pass C++ conversion via schema injection ──────────────────
    # Infer the full schema from pandas in one shot so all non-timestamp fields
    # get correct Arrow types automatically, then override timestamp fields with
    # the canonical microsecond-UTC type.  Arrow pre-allocates C++ buffers for
    # the exact target layout and converts all columns in one pass.
    inferred: pa.Schema = pa.Schema.from_pandas(df, preserve_index=False)
    fields = [
        pa.field(f.name, _TS_TYPE, nullable=f.nullable)
        if f.name in _TIME_COLS
        else f
        for f in inferred
    ]
    return pa.Table.from_pandas(df, schema=pa.schema(fields), preserve_index=False)


def _infer_shape(table: pa.Table) -> DataShape:
    """Infer :class:`DataShape` from the column names present in *table*."""
    names = set(table.schema.names)
    has_kt = "knowledge_time" in names
    has_ct = "change_time" in names
    if has_kt and has_ct:
        return DataShape.AUDIT
    if has_ct:
        return DataShape.CORRECTED
    if has_kt:
        return DataShape.VERSIONED
    return DataShape.SIMPLE


def _validate_table(table: pa.Table, shape: DataShape) -> None:
    """Raise ``ValueError`` if required columns are missing or have wrong type."""
    names = set(table.schema.names)
    required = _REQUIRED_COLUMNS[shape]
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(
            f"Table is missing required columns for shape {shape.value}: {missing}"
        )

    # Check timestamp columns have the right type
    ts_cols = ["valid_time", "valid_time_end", "knowledge_time", "change_time"]
    for col in ts_cols:
        if col not in names:
            continue
        idx = table.schema.get_field_index(col)
        field = table.schema.field(idx)
        if not pa.types.is_timestamp(field.type):
            raise TypeError(
                f"Column '{col}' must be a timestamp type, got {field.type!r}"
            )
        if field.type.tz is None:
            raise TypeError(
                f"Column '{col}' must be timezone-aware (expected tz='UTC'), "
                f"got tz=None"
            )
