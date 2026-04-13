import pytest
import polars as pl

import timedatamodel as tdm
from timedatamodel import (
    DataType,
    Frequency,
    TimeSeries,
    TimeSeriesDescriptor,
    TimeSeriesType,
)


class TestTimeSeriesDescriptor:
    """Tests for the TimeSeriesDescriptor frozen dataclass."""

    def test_defaults(self):
        desc = TimeSeriesDescriptor(name="power")
        assert desc.name == "power"
        assert desc.unit == "dimensionless"
        assert desc.data_type is None
        assert desc.timeseries_type == TimeSeriesType.FLAT
        assert desc.frequency is None
        assert desc.timezone == "UTC"
        assert desc.description is None

    def test_name_required(self):
        with pytest.raises(TypeError):
            TimeSeriesDescriptor()  # type: ignore[call-arg]

    def test_all_fields(self):
        desc = TimeSeriesDescriptor(
            name="power",
            unit="MW",
            data_type=DataType.FORECAST,
            timeseries_type=TimeSeriesType.OVERLAPPING,
            frequency=Frequency.PT1H,
            timezone="Europe/Stockholm",
            description="Wind power forecast",
        )
        assert desc.name == "power"
        assert desc.unit == "MW"
        assert desc.data_type == DataType.FORECAST
        assert desc.timeseries_type == TimeSeriesType.OVERLAPPING
        assert desc.frequency == Frequency.PT1H
        assert desc.timezone == "Europe/Stockholm"
        assert desc.description == "Wind power forecast"

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

    def test_hashable(self):
        """Without label/location mappings, the slim descriptor is hashable."""
        desc = TimeSeriesDescriptor(name="power", unit="MW")
        assert isinstance(hash(desc), int)

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
        assert ts2.frequency == ts.frequency
        assert ts2.timezone == ts.timezone

    def test_all_descriptor_fields_round_trip(self, simple_df):
        """Guard against future fields being added to TimeSeriesDescriptor
        but forgotten in to_descriptor() / from_descriptor().

        Builds a descriptor with a distinctive value for every field, goes
        desc → TimeSeries → desc, and asserts field-by-field equality.
        """
        import dataclasses

        desc1 = TimeSeriesDescriptor(
            name="power",
            unit="MW",
            data_type=DataType.FORECAST,
            timeseries_type=TimeSeriesType.OVERLAPPING,
            frequency=Frequency.PT1H,
            timezone="Europe/Oslo",
            description="Test series",
        )
        # Sanity check: every non-required field was set to a non-default value.
        defaults = TimeSeriesDescriptor(name="power")
        for f in dataclasses.fields(TimeSeriesDescriptor):
            if f.name == "name":
                continue
            assert getattr(desc1, f.name) != getattr(defaults, f.name), (
                f"Test doesn't set a distinctive value for field {f.name!r} — "
                "update this test to cover the new field."
            )

        ts = TimeSeries.from_descriptor(desc1, simple_df)
        desc2 = ts.to_descriptor()
        for f in dataclasses.fields(TimeSeriesDescriptor):
            assert getattr(desc1, f.name) == getattr(desc2, f.name), (
                f"Field {f.name!r} did not round-trip"
            )
