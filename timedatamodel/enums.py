from __future__ import annotations

from datetime import timedelta
from enum import StrEnum

_CALENDAR_BASED = frozenset({"P1Y", "P3M", "P1M"})

_FREQUENCY_TO_TIMEDELTA: dict[str, timedelta] = {
    "P1W": timedelta(weeks=1),
    "P1D": timedelta(days=1),
    "PT1H": timedelta(hours=1),
    "PT30M": timedelta(minutes=30),
    "PT15M": timedelta(minutes=15),
    "PT10M": timedelta(minutes=10),
    "PT5M": timedelta(minutes=5),
    "PT1M": timedelta(minutes=1),
    "PT1S": timedelta(seconds=1),
}


class Frequency(StrEnum):
    P1Y = "P1Y"
    P3M = "P3M"
    P1M = "P1M"
    P1W = "P1W"
    P1D = "P1D"
    PT1H = "PT1H"
    PT30M = "PT30M"
    PT15M = "PT15M"
    PT10M = "PT10M"
    PT5M = "PT5M"
    PT1M = "PT1M"
    PT1S = "PT1S"
    NONE = "NONE"

    @property
    def is_calendar_based(self) -> bool:
        return self.value in _CALENDAR_BASED

    def to_timedelta(self) -> timedelta | None:
        if self.is_calendar_based or self == Frequency.NONE:
            return None
        return _FREQUENCY_TO_TIMEDELTA[self.value]


class DataType(StrEnum):
    MEASUREMENT = "MEASUREMENT"
    ESTIMATION = "ESTIMATION"
    FORECAST = "FORECAST"
    SCENARIO = "SCENARIO"
    SYNTHETIC = "SYNTHETIC"
    CLIMATE = "CLIMATE"
    ACTUAL = "ACTUAL"


class TimeSeriesType(StrEnum):
    FLAT = "FLAT"
    OVERLAPPING = "OVERLAPPING"
