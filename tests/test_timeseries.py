from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from timedatamodel import (
    DataPoint,
    DataType,
    Frequency,
    Metadata,
    Resolution,
    TimeSeries,
)


@pytest.fixture
def hourly_resolution():
    return Resolution(frequency=Frequency.PT1H)


@pytest.fixture
def sample_ts(hourly_resolution):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [base + timedelta(hours=i) for i in range(5)]
    values = [1.0, 2.0, 3.0, None, 5.0]
    metadata = Metadata(name="power", unit="MW", data_type=DataType.ACTUAL)
    return TimeSeries(
        hourly_resolution, metadata, timestamps=timestamps, values=values
    )


class TestConstruction:
    def test_from_lists(self, hourly_resolution):
        ts = TimeSeries(
            hourly_resolution,
            timestamps=[datetime(2024, 1, 1, tzinfo=timezone.utc)],
            values=[42.0],
        )
        assert len(ts) == 1
        assert ts[0].value == 42.0

    def test_from_data(self, hourly_resolution):
        dp = DataPoint(datetime(2024, 1, 1, tzinfo=timezone.utc), 42.0)
        ts = TimeSeries(hourly_resolution, data=[dp])
        assert len(ts) == 1
        assert ts[0] == dp

    def test_both_raises(self, hourly_resolution):
        with pytest.raises(ValueError, match="cannot specify both"):
            TimeSeries(
                hourly_resolution,
                timestamps=[datetime(2024, 1, 1, tzinfo=timezone.utc)],
                values=[1.0],
                data=[DataPoint(datetime(2024, 1, 1, tzinfo=timezone.utc), 1.0)],
            )

    def test_empty(self, hourly_resolution):
        ts = TimeSeries(hourly_resolution)
        assert len(ts) == 0
        assert not ts
        assert list(ts) == []


class TestSequenceProtocol:
    def test_len(self, sample_ts):
        assert len(sample_ts) == 5

    def test_getitem(self, sample_ts):
        dp = sample_ts[0]
        assert isinstance(dp, DataPoint)
        assert dp.value == 1.0

    def test_getitem_slice(self, sample_ts):
        result = sample_ts[1:3]
        assert len(result) == 2
        assert result[0].value == 2.0
        assert result[1].value == 3.0

    def test_iter(self, sample_ts):
        points = list(sample_ts)
        assert len(points) == 5
        assert all(isinstance(p, DataPoint) for p in points)

    def test_bool_true(self, sample_ts):
        assert bool(sample_ts) is True

    def test_bool_false(self, hourly_resolution):
        assert bool(TimeSeries(hourly_resolution)) is False


class TestNumpy:
    def test_to_numpy(self, sample_ts):
        arr = sample_ts.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert len(arr) == 5
        assert arr[0] == 1.0
        assert np.isnan(arr[3])


class TestPandas:
    def test_to_pandas_dataframe(self, sample_ts):
        df = sample_ts.to_pandas_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.columns.tolist() == ["power"]
        assert len(df) == 5
        assert pd.isna(df.iloc[3, 0])

    def test_default_column_name(self, hourly_resolution):
        ts = TimeSeries(
            hourly_resolution,
            timestamps=[datetime(2024, 1, 1, tzinfo=timezone.utc)],
            values=[1.0],
        )
        df = ts.to_pandas_dataframe()
        assert df.columns.tolist() == ["value"]

    def test_round_trip(self, sample_ts):
        df = sample_ts.to_pandas_dataframe()
        ts2 = TimeSeries.from_pandas(df, sample_ts.resolution)
        assert len(ts2) == len(sample_ts)
        assert ts2[0].value == sample_ts[0].value
        assert ts2[3].value is None


class TestPolars:
    def test_to_polars_dataframe(self, sample_ts):
        import polars as pl

        df = sample_ts.to_polars_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert "timestamp" in df.columns
        assert "power" in df.columns
        assert len(df) == 5

    def test_round_trip(self, sample_ts):
        df = sample_ts.to_polars_dataframe()
        ts2 = TimeSeries.from_polars(df, sample_ts.resolution)
        assert len(ts2) == len(sample_ts)
        assert ts2[0].value == sample_ts[0].value
        assert ts2[3].value is None


class TestValidation:
    def test_valid(self, sample_ts):
        warnings = sample_ts.validate()
        assert warnings == []

    def test_unordered(self, hourly_resolution):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = TimeSeries(
            hourly_resolution,
            timestamps=[base + timedelta(hours=1), base],
            values=[1.0, 2.0],
        )
        warnings = ts.validate()
        assert any("not strictly increasing" in w for w in warnings)

    def test_inconsistent_frequency(self, hourly_resolution):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = TimeSeries(
            hourly_resolution,
            timestamps=[base, base + timedelta(hours=1), base + timedelta(hours=3)],
            values=[1.0, 2.0, 3.0],
        )
        warnings = ts.validate()
        assert any("inconsistent frequency" in w for w in warnings)
