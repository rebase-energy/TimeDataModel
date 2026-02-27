from timedatamodel import DataType, Frequency, TimeSeriesType


class TestFrequency:
    def test_membership(self):
        assert Frequency.PT1H == "PT1H"
        assert Frequency.P1D == "P1D"
        assert Frequency.NONE == "NONE"

    def test_all_values(self):
        expected = {
            "P1Y", "P3M", "P1M", "P1W", "P1D",
            "PT1H", "PT30M", "PT15M", "PT10M", "PT5M", "PT1M", "PT1S",
            "NONE",
        }
        assert {f.value for f in Frequency} == expected

    def test_string_comparison(self):
        assert Frequency.PT15M == "PT15M"
        assert str(Frequency.PT15M) == "PT15M"


class TestDataType:
    def test_membership(self):
        assert DataType.FORECAST == "FORECAST"
        assert DataType.ACTUAL == "ACTUAL"

    def test_all_values(self):
        expected = {
            "MEASUREMENT", "ESTIMATION", "FORECAST", "SCENARIO",
            "SYNTHETIC", "CLIMATE", "ACTUAL",
        }
        assert {dt.value for dt in DataType} == expected


class TestTimeSeriesType:
    def test_membership(self):
        assert TimeSeriesType.FLAT == "FLAT"
        assert TimeSeriesType.OVERLAPPING == "OVERLAPPING"

    def test_string_comparison(self):
        assert str(TimeSeriesType.FLAT) == "FLAT"
