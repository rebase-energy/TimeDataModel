from datetime import timedelta
from zoneinfo import ZoneInfo

import pytest

from timedatamodel import Frequency, Resolution


class TestResolution:
    def test_default_timezone(self):
        r = Resolution(frequency=Frequency.PT1H)
        assert r.timezone == "UTC"

    def test_tz_property(self):
        r = Resolution(frequency=Frequency.PT1H, timezone="Europe/Oslo")
        assert r.tz == ZoneInfo("Europe/Oslo")

    def test_invalid_timezone(self):
        with pytest.raises(Exception):
            Resolution(frequency=Frequency.PT1H, timezone="Invalid/Zone")

    def test_is_calendar_based(self):
        assert Resolution(frequency=Frequency.P1Y).is_calendar_based is True
        assert Resolution(frequency=Frequency.P3M).is_calendar_based is True
        assert Resolution(frequency=Frequency.P1M).is_calendar_based is True
        assert Resolution(frequency=Frequency.P1D).is_calendar_based is False
        assert Resolution(frequency=Frequency.PT1H).is_calendar_based is False

    def test_to_timedelta_fixed(self):
        assert Resolution(frequency=Frequency.PT1H).to_timedelta() == timedelta(hours=1)
        assert Resolution(frequency=Frequency.P1D).to_timedelta() == timedelta(days=1)
        assert Resolution(frequency=Frequency.PT15M).to_timedelta() == timedelta(minutes=15)
        assert Resolution(frequency=Frequency.P1W).to_timedelta() == timedelta(weeks=1)
        assert Resolution(frequency=Frequency.PT1S).to_timedelta() == timedelta(seconds=1)

    def test_to_timedelta_calendar_based(self):
        assert Resolution(frequency=Frequency.P1Y).to_timedelta() is None
        assert Resolution(frequency=Frequency.P1M).to_timedelta() is None

    def test_to_timedelta_none_frequency(self):
        assert Resolution(frequency=Frequency.NONE).to_timedelta() is None

    def test_frozen(self):
        r = Resolution(frequency=Frequency.PT1H)
        with pytest.raises(AttributeError):
            r.frequency = Frequency.P1D  # type: ignore[misc]
