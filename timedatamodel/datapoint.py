from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

from ._repr import _datapoint_repr, _datapoint_repr_html, _fmt_value


class DataPoint:
    """A single (timestamp, value) observation.

    Supports tuple-style access for backwards compatibility:
    ``ts, val = datapoint`` and ``datapoint[0]`` both work.
    """

    __slots__ = ("timestamp", "value")

    def __init__(self, timestamp: datetime, value: float | None) -> None:
        self.timestamp = timestamp
        self.value = value

    # ---- tuple compatibility ---------------------------------------------

    def __iter__(self) -> Iterator[datetime | float | None]:
        yield self.timestamp
        yield self.value

    def __getitem__(self, index: int) -> datetime | float | None:
        if index == 0:
            return self.timestamp
        if index == 1:
            return self.value
        raise IndexError(f"DataPoint index out of range: {index}")

    def __len__(self) -> int:
        return 2

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DataPoint):
            return self.timestamp == other.timestamp and self.value == other.value
        if isinstance(other, tuple) and len(other) == 2:
            return self.timestamp == other[0] and self.value == other[1]
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.timestamp, self.value))

    # ---- repr ------------------------------------------------------------

    __repr__ = _datapoint_repr
    _repr_html_ = _datapoint_repr_html
    _fmt_value = staticmethod(_fmt_value)
