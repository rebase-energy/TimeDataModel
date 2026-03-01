import timedatamodel as tdm


class TestFrequency:
    def test_membership(self):
        assert tdm.Frequency.PT1H == "PT1H"
        assert tdm.Frequency.P1D == "P1D"
        assert tdm.Frequency.NONE == "NONE"

    def test_all_values(self):
        expected = {
            "P1Y", "P3M", "P1M", "P1W", "P1D",
            "PT1H", "PT30M", "PT15M", "PT10M", "PT5M", "PT1M", "PT1S",
            "NONE",
        }
        assert {f.value for f in tdm.Frequency} == expected

    def test_string_comparison(self):
        assert tdm.Frequency.PT15M == "PT15M"
        assert str(tdm.Frequency.PT15M) == "PT15M"


class TestDataType:
    def test_membership(self):
        assert tdm.DataType.FORECAST == "FORECAST"
        assert tdm.DataType.ACTUAL == "ACTUAL"

    def test_all_values(self):
        expected = {
            "MEASUREMENT", "ESTIMATION", "FORECAST", "SCENARIO",
            "SYNTHETIC", "CLIMATE", "ACTUAL",
        }
        assert {dt.value for dt in tdm.DataType} == expected


class TestTimeSeriesType:
    def test_membership(self):
        assert tdm.TimeSeriesType.FLAT == "FLAT"
        assert tdm.TimeSeriesType.OVERLAPPING == "OVERLAPPING"

    def test_string_comparison(self):
        assert str(tdm.TimeSeriesType.FLAT) == "FLAT"
