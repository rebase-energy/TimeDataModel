from __future__ import annotations

from datetime import datetime
from typing import NamedTuple


class DataPoint(NamedTuple):
    timestamp: datetime
    value: float | None
