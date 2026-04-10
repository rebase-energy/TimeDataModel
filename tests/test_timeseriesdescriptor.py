import pytest
import polars as pl

import timedatamodel as tdm
from timedatamodel import (
    DataType,
    Frequency,
    GeoLocation,
    TimeSeries,
    TimeSeriesDescriptor,
    TimeSeriesType,
)


class TestTimeSeriesDescriptor:
    """Tests for the TimeSeriesDescriptor frozen dataclass."""

    def test_defaults(self):
        desc = TimeSeriesDescriptor()
        assert desc.name is None
        assert desc.unit == "dimensionless"
        assert desc.data_type is None
        assert desc.timeseries_type == TimeSeriesType.FLAT
        assert desc.description is None
        assert desc.labels == {}
        assert desc.frequency is None
        assert desc.location is None
        assert desc.timezone == "UTC"

    def test_all_fields(self):
        loc = GeoLocation(latitude=55.0, longitude=3.0)
        desc = TimeSeriesDescriptor(
            name="power",
            unit="MW",
            data_type=DataType.FORECAST,
            timeseries_type=TimeSeriesType.OVERLAPPING,
            description="Wind power forecast",
            labels={"site": "offshore-1"},
            frequency=Frequency.PT1H,
            location=loc,
            timezone="Europe/Stockholm",
        )
        assert desc.name == "power"
        assert desc.unit == "MW"
        assert desc.data_type == DataType.FORECAST
        assert desc.timeseries_type == TimeSeriesType.OVERLAPPING
        assert desc.description == "Wind power forecast"
        assert desc.labels == {"site": "offshore-1"}
        assert desc.frequency == Frequency.PT1H
        assert desc.location == loc
        assert desc.timezone == "Europe/Stockholm"

    def test_frozen(self):
        desc = TimeSeriesDescriptor(name="power")
        with pytest.raises(AttributeError):
            desc.name = "changed"  # type: ignore[misc]

    def test_equality(self):
        a = TimeSeriesDescriptor(name="power", unit="MW", data_type=DataType.FORECAST)
        b = TimeSeriesDescriptor(name="power", unit="MW", data_type=DataType.FORECAST)
        assert a == b

    def test_inequality(self):
        a = TimeSeriesDescriptor(name="power", data_type=DataType.FORECAST)
        b = TimeSeriesDescriptor(name="power", data_type=DataType.ACTUAL)
        assert a != b

    def test_not_hashable_with_labels(self):
        """Labels dict makes it unhashable — this is expected."""
        desc = TimeSeriesDescriptor(name="power", unit="MW", labels={"a": "b"})
        with pytest.raises(TypeError, match="unhashable"):
            hash(desc)

    def test_exported_from_package(self):
        assert hasattr(tdm, "TimeSeriesDescriptor")
        assert tdm.TimeSeriesDescriptor is TimeSeriesDescriptor


class TestTimeSeriesDescriptorRoundTrip:
    """Tests for to_descriptor() and from_descriptor() on TimeSeries."""

    @pytest.fixture
    def simple_df(self):
        return pl.DataFrame({
            "valid_time": [None],
            "value": [1.0],
        }).cast({"valid_time": pl.Datetime("us", "UTC")})

    @pytest.fixture
    def ts(self, simple_df):
        return TimeSeries(
            simple_df,
            name="power",
            unit="MW",
            data_type=DataType.FORECAST,
            timeseries_type=TimeSeriesType.OVERLAPPING,
            description="Test series",
            labels={"site": "A"},
            frequency=Frequency.PT1H,
            timezone="Europe/Oslo",
        )

    def test_to_descriptor(self, ts):
        desc = ts.to_descriptor()
        assert isinstance(desc, TimeSeriesDescriptor)
        assert desc.name == "power"
        assert desc.unit == "MW"
        assert desc.data_type == DataType.FORECAST
        assert desc.timeseries_type == TimeSeriesType.OVERLAPPING
        assert desc.description == "Test series"
        assert desc.labels == {"site": "A"}
        assert desc.frequency == Frequency.PT1H
        assert desc.timezone == "Europe/Oslo"

    def test_to_descriptor_no_data(self, ts):
        """Descriptor should not carry any DataFrame reference."""
        desc = ts.to_descriptor()
        assert not hasattr(desc, "_df")
        assert not hasattr(desc, "df")

    def test_from_descriptor(self, simple_df):
        desc = TimeSeriesDescriptor(
            name="wind_speed",
            unit="m/s",
            data_type=DataType.ACTUAL,
            timeseries_type=TimeSeriesType.FLAT,
        )
        ts = TimeSeries.from_descriptor(desc, simple_df)
        assert ts.name == "wind_speed"
        assert ts.unit == "m/s"
        assert ts.data_type == DataType.ACTUAL
        assert ts.timeseries_type == TimeSeriesType.FLAT
        assert ts.num_rows == 1

    def test_round_trip(self, ts, simple_df):
        """TimeSeries → descriptor → TimeSeries preserves all metadata."""
        desc = ts.to_descriptor()
        ts2 = TimeSeries.from_descriptor(desc, simple_df)
        assert ts2.name == ts.name
        assert ts2.unit == ts.unit
        assert ts2.data_type == ts.data_type
        assert ts2.timeseries_type == ts.timeseries_type
        assert ts2.description == ts.description
        assert ts2.labels == ts.labels
        assert ts2.frequency == ts.frequency
        assert ts2.timezone == ts.timezone

    def test_descriptor_labels_are_independent(self, ts):
        """Mutating the original labels dict should not affect the descriptor."""
        desc = ts.to_descriptor()
        ts.labels["new_key"] = "new_value"
        assert "new_key" not in desc.labels
