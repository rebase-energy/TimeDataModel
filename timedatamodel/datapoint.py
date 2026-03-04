from __future__ import annotations

from datetime import datetime
from typing import NamedTuple


class DataPoint(NamedTuple):
    timestamp: datetime
    value: float | None


from ._repr import _datapoint_repr, _datapoint_repr_html, _fmt_value

DataPoint.__repr__ = _datapoint_repr
DataPoint._repr_html_ = _datapoint_repr_html
DataPoint._fmt_value = staticmethod(_fmt_value)
