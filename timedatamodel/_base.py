from __future__ import annotations

import bisect
import functools
from datetime import datetime, timedelta

import numpy as np

from .enums import Frequency

_default_dataframe_backend: str = "pandas"


def set_default_df(backend: str) -> None:
    """Set the default DataFrame backend for the ``.df`` property.

    Parameters
    ----------
    backend : str
        ``"pandas"`` or ``"polars"``.
    """
    global _default_dataframe_backend
    if backend not in ("pandas", "polars"):
        raise ValueError(
            f"backend must be 'pandas' or 'polars', got {backend!r}"
        )
    _default_dataframe_backend = backend


def get_default_df() -> str:
    """Return the current default DataFrame backend (``"pandas"`` or ``"polars"``)."""
    return _default_dataframe_backend


_PANDAS_FREQ_MAP: dict[str, Frequency] = {
    # Modern pandas (>=2.0) aliases
    "YE": Frequency.P1Y, "YE-DEC": Frequency.P1Y, "A": Frequency.P1Y,
    "QE": Frequency.P3M, "QE-DEC": Frequency.P3M, "Q": Frequency.P3M,
    "ME": Frequency.P1M, "M": Frequency.P1M,
    "W": Frequency.P1W, "W-SUN": Frequency.P1W,
    "D": Frequency.P1D,
    "h": Frequency.PT1H, "H": Frequency.PT1H,
    "30min": Frequency.PT30M, "30T": Frequency.PT30M,
    "15min": Frequency.PT15M, "15T": Frequency.PT15M,
    "10min": Frequency.PT10M, "10T": Frequency.PT10M,
    "5min": Frequency.PT5M, "5T": Frequency.PT5M,
    "min": Frequency.PT1M, "T": Frequency.PT1M,
    "s": Frequency.PT1S, "S": Frequency.PT1S,
}


def _import_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError(
            "pandas is required for this operation. "
            "Install it with: pip install timedatamodel[pandas]"
        ) from None


def _import_polars():
    try:
        import polars as pl
        return pl
    except ImportError:
        raise ImportError(
            "polars is required for this operation. "
            "Install it with: pip install timedatamodel[polars]"
        ) from None


def _extract_timestamps_from_pandas_index(df):
    """Extract timestamps and index names from a pandas DataFrame index."""
    pd = _import_pandas()
    if isinstance(df.index, pd.MultiIndex):
        timestamps = [
            tuple(
                lvl.to_pydatetime()
                if hasattr(lvl, "to_pydatetime")
                else lvl
                for lvl in row
            )
            for row in df.index
        ]
        index_names = list(df.index.names)
    elif isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index.to_pydatetime().tolist()
        index_names = None
    else:
        raise ValueError(
            "DataFrame must have a DatetimeIndex or MultiIndex"
        )
    return timestamps, index_names


@functools.lru_cache(maxsize=1)
def _get_pint_registry():
    import pint
    return pint.UnitRegistry()


def _convert_unit_values(values: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    """Convert *values* from *from_unit* to *to_unit* using pint.

    Short-circuits when units are identical.  Wraps pint dimensionality
    errors into a plain ``ValueError`` and ``ImportError`` with install hints.
    """
    if from_unit == to_unit:
        return values
    try:
        import pint
    except ImportError:
        raise ImportError(
            "pint is required for unit conversion. "
            "Install it with: pip install timedatamodel[pint]"
        ) from None
    ureg = _get_pint_registry()
    try:
        factor = ureg.Quantity(1, from_unit).to(to_unit).magnitude
    except pint.errors.DimensionalityError:
        raise ValueError(
            f"cannot convert '{from_unit}' to '{to_unit}': incompatible dimensions"
        ) from None
    return values * factor


# ---------------------------------------------------------------------------
# _TimeSeriesBase — shared logic for univariate and multivariate
# ---------------------------------------------------------------------------


from ._repr import _TimeSeriesBaseReprMixin


class _TimeSeriesBase(_TimeSeriesBaseReprMixin):
    __slots__ = ()

    @property
    def timestamps(self) -> list[datetime] | list[tuple[datetime, ...]]:
        return self._timestamps

    @property
    def is_multi_index(self) -> bool:
        """True if timestamps are tuples of datetimes."""
        return len(self._timestamps) > 0 and isinstance(self._timestamps[0], tuple)

    @property
    def index_names(self) -> tuple[str, ...]:
        """Labels for the index dimensions."""
        if self._index_names is not None:
            return tuple(self._index_names)
        if self.is_multi_index and self._timestamps:
            return tuple(f"index_{i}" for i in range(len(self._timestamps[0])))
        return ("timestamp",)

    def __len__(self) -> int:
        return len(self._timestamps)

    def __bool__(self) -> bool:
        return len(self._timestamps) > 0

    def __contains__(self, dt: datetime) -> bool:
        """Return True if *dt* appears in the timestamps."""
        i = bisect.bisect_left(self._timestamps, dt)
        return i < len(self._timestamps) and self._timestamps[i] == dt

    @property
    def begin(self) -> datetime | tuple[datetime, ...] | None:
        return self._timestamps[0] if self._timestamps else None

    @property
    def end(self) -> datetime | tuple[datetime, ...] | None:
        return self._timestamps[-1] if self._timestamps else None

    @property
    def duration(self) -> timedelta | None:
        """Time span from begin to end; None if empty."""
        if not self._timestamps:
            return None
        first = self._timestamps[0]
        last = self._timestamps[-1]
        if isinstance(first, tuple):
            return last[0] - first[0]
        return last - first

    def validate(self) -> list[str]:
        """Return a list of validation warnings (timestamp ordering and frequency)."""
        warnings: list[str] = []
        td = self.frequency.to_timedelta()
        check_order = True
        check_freq = td is not None
        multi = self.is_multi_index
        for i in range(1, len(self._timestamps)):
            if check_order and self._timestamps[i] <= self._timestamps[i - 1]:
                warnings.append(
                    f"timestamps not strictly increasing at index {i}: "
                    f"{self._timestamps[i-1]} >= {self._timestamps[i]}"
                )
                check_order = False
            if check_freq:
                cur = self._timestamps[i][0] if multi else self._timestamps[i]
                prev = self._timestamps[i - 1][0] if multi else self._timestamps[i - 1]
                if cur - prev != td:
                    warnings.append(
                        f"inconsistent frequency at index {i}: "
                        f"expected {td}, got {cur - prev}"
                    )
                    check_freq = False
            if not check_order and not check_freq:
                break
        return warnings

    # ---- static helpers --------------------------------------------------

    @staticmethod
    def _from_float_array(arr: np.ndarray) -> list[float | None]:
        """Convert float64 ndarray to list[float|None] — NaN → None."""
        values: list[float | None] = arr.tolist()
        for i in np.where(np.isnan(arr))[0]:
            values[i] = None
        return values

    @staticmethod
    def _infer_freq_tz(
        df: "pd.DataFrame", fallback_freq: Frequency, fallback_tz: str,
    ) -> tuple[Frequency, str]:
        pd = _import_pandas()
        new_tz = str(df.index.tz) if df.index.tz is not None else fallback_tz
        freq_str: str | None = None
        if df.index.freq is not None:
            freq_str = df.index.freqstr
        elif len(df.index) >= 3:
            try:
                freq_str = pd.infer_freq(df.index)
            except (ValueError, TypeError):
                pass
        new_freq = (
            _PANDAS_FREQ_MAP.get(freq_str, fallback_freq)
            if freq_str
            else fallback_freq
        )
        return (new_freq, new_tz)


def _xarray_labels_to_list(raw) -> list:
    """Convert xarray coord values (numpy) to Python datetime/float/str."""
    import pandas as pd
    out = []
    for v in raw:
        if isinstance(v, (np.datetime64, pd.Timestamp)):
            out.append(pd.Timestamp(v).to_pydatetime())
        elif isinstance(v, (np.floating, float)):
            out.append(float(v))
        elif isinstance(v, np.integer):
            out.append(float(v))
        else:
            out.append(str(v))
    return out


class _DataFrameMixin:
    """Mixin providing the ``df`` shorthand property."""
    __slots__ = ()

    @property
    def df(self):
        """Shorthand for the default DataFrame backend (pandas or polars)."""
        if _default_dataframe_backend == "polars":
            return self.to_polars_dataframe()
        return self.to_pandas_dataframe()


def _validate_timestamp_sequence(
    timestamps: list[datetime] | list[tuple[datetime, ...]],
) -> None:
    """Validate timestamp container shape and element types."""
    if not timestamps:
        return

    first = timestamps[0]

    if isinstance(first, tuple):
        tuple_len = len(first)
        if tuple_len == 0:
            raise ValueError("multi-index timestamp tuples must not be empty")

        for i, ts in enumerate(timestamps):
            if not isinstance(ts, tuple):
                raise TypeError(
                    "timestamps must be homogeneous: all datetime values "
                    "or all tuples of datetime values"
                )
            if len(ts) != tuple_len:
                raise ValueError(
                    f"inconsistent multi-index tuple length at index {i}: "
                    f"expected {tuple_len}, got {len(ts)}"
                )
            for j, dt in enumerate(ts):
                if not isinstance(dt, datetime):
                    raise TypeError(
                        f"timestamp tuple element at index {i}[{j}] must be datetime, "
                        f"got {type(dt).__name__}"
                    )
        return

    if not isinstance(first, datetime):
        raise TypeError(
            f"timestamp at index 0 must be datetime, got {type(first).__name__}"
        )

    for i, ts in enumerate(timestamps):
        if isinstance(ts, tuple):
            raise TypeError(
                "timestamps must be homogeneous: all datetime values "
                "or all tuples of datetime values"
            )
        if not isinstance(ts, datetime):
            raise TypeError(
                f"timestamp at index {i} must be datetime, got {type(ts).__name__}"
            )
