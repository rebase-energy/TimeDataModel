from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from zoneinfo import ZoneInfo

from .enums import Frequency

_CALENDAR_BASED = {Frequency.P1Y, Frequency.P3M, Frequency.P1M}

_FREQUENCY_TO_TIMEDELTA: dict[Frequency, timedelta] = {
    Frequency.P1W: timedelta(weeks=1),
    Frequency.P1D: timedelta(days=1),
    Frequency.PT1H: timedelta(hours=1),
    Frequency.PT30M: timedelta(minutes=30),
    Frequency.PT15M: timedelta(minutes=15),
    Frequency.PT10M: timedelta(minutes=10),
    Frequency.PT5M: timedelta(minutes=5),
    Frequency.PT1M: timedelta(minutes=1),
    Frequency.PT1S: timedelta(seconds=1),
}


@dataclass(frozen=True, slots=True)
class Resolution:
    frequency: Frequency
    timezone: str = "UTC"

    def __post_init__(self) -> None:
        # Validate timezone
        ZoneInfo(self.timezone)

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def is_calendar_based(self) -> bool:
        return self.frequency in _CALENDAR_BASED

    def to_timedelta(self) -> timedelta | None:
        if self.is_calendar_based or self.frequency == Frequency.NONE:
            return None
        return _FREQUENCY_TO_TIMEDELTA[self.frequency]
