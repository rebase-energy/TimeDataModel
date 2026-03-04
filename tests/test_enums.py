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
            "ACTUAL", "OBSERVATION", "DERIVED",
            "CALCULATED", "ESTIMATION", "FORECAST", "PREDICTION",
            "SCENARIO", "SIMULATION", "RECONSTRUCTION",
            "REFERENCE", "BASELINE", "BENCHMARK", "IDEAL",
        }
        assert {dt.value for dt in tdm.DataType} == expected

    def test_parent(self):
        assert tdm.DataType.ACTUAL.parent is None
        assert tdm.DataType.CALCULATED.parent is None
        assert tdm.DataType.OBSERVATION.parent == tdm.DataType.ACTUAL
        assert tdm.DataType.FORECAST.parent == tdm.DataType.ESTIMATION
        assert tdm.DataType.BASELINE.parent == tdm.DataType.REFERENCE

    def test_children(self):
        assert tdm.DataType.ACTUAL.children == [
            tdm.DataType.OBSERVATION,
            tdm.DataType.DERIVED,
        ]
        assert tdm.DataType.ESTIMATION.children == [
            tdm.DataType.FORECAST,
            tdm.DataType.PREDICTION,
            tdm.DataType.SCENARIO,
            tdm.DataType.SIMULATION,
            tdm.DataType.RECONSTRUCTION,
        ]
        assert tdm.DataType.FORECAST.children == []

    def test_is_leaf(self):
        assert tdm.DataType.OBSERVATION.is_leaf is True
        assert tdm.DataType.FORECAST.is_leaf is True
        assert tdm.DataType.ACTUAL.is_leaf is False
        assert tdm.DataType.ESTIMATION.is_leaf is False

    def test_root(self):
        assert tdm.DataType.ACTUAL.root == tdm.DataType.ACTUAL
        assert tdm.DataType.OBSERVATION.root == tdm.DataType.ACTUAL
        assert tdm.DataType.FORECAST.root == tdm.DataType.CALCULATED
        assert tdm.DataType.BASELINE.root == tdm.DataType.CALCULATED
        assert tdm.DataType.ESTIMATION.root == tdm.DataType.CALCULATED


class TestTimeSeriesType:
    def test_membership(self):
        assert tdm.TimeSeriesType.FLAT == "FLAT"
        assert tdm.TimeSeriesType.OVERLAPPING == "OVERLAPPING"

    def test_string_comparison(self):
        assert str(tdm.TimeSeriesType.FLAT) == "FLAT"
