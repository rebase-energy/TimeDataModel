import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timedatamodel import (
    DataPoint,
    DataType,
    Frequency,
    GeoLocation,
    MultiTimeSeries,
    MultivariateTimeSeries,
    TimeSeriesType,
    TimeSeries,
)


@pytest.fixture
def hourly_frequency():
    return Frequency.PT1H


@pytest.fixture
def sample_ts(hourly_frequency):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [base + timedelta(hours=i) for i in range(5)]
    values = [1.0, 2.0, 3.0, None, 5.0]
    return TimeSeries(
        hourly_frequency,
        timestamps=timestamps,
        values=values,
        name="power",
        unit="MW",
        data_type=DataType.ACTUAL,
    )


class TestConstruction:
    def test_from_lists(self, hourly_frequency):
        ts = TimeSeries(
            hourly_frequency,
            timestamps=[datetime(2024, 1, 1, tzinfo=timezone.utc)],
            values=[42.0],
        )
        assert len(ts) == 1
        assert ts[0].value == 42.0

    def test_from_data(self, hourly_frequency):
        dp = DataPoint(datetime(2024, 1, 1, tzinfo=timezone.utc), 42.0)
        ts = TimeSeries(hourly_frequency, data=[dp])
        assert len(ts) == 1
        assert ts[0] == dp

    def test_both_raises(self, hourly_frequency):
        with pytest.raises(ValueError, match="cannot specify both"):
            TimeSeries(
                hourly_frequency,
                timestamps=[datetime(2024, 1, 1, tzinfo=timezone.utc)],
                values=[1.0],
                data=[DataPoint(datetime(2024, 1, 1, tzinfo=timezone.utc), 1.0)],
            )

    def test_empty(self, hourly_frequency):
        ts = TimeSeries(hourly_frequency)
        assert len(ts) == 0
        assert not ts
        assert list(ts) == []

    def test_scalar_metadata(self, hourly_frequency):
        loc = GeoLocation(latitude=59.91, longitude=10.75)
        ts = TimeSeries(
            hourly_frequency,
            timestamps=[datetime(2024, 1, 1, tzinfo=timezone.utc)],
            values=[1.0],
            name="power",
            unit="MW",
            description="Hourly power",
            data_type=DataType.ACTUAL,
            location=loc,
            timeseries_type=TimeSeriesType.FLAT,
            attributes={"source": "test"},
        )
        assert ts.name == "power"
        assert ts.unit == "MW"
        assert ts.description == "Hourly power"
        assert ts.data_type == DataType.ACTUAL
        assert ts.location == loc
        assert ts.timeseries_type == TimeSeriesType.FLAT
        assert ts.attributes["source"] == "test"

    def test_defaults(self, hourly_frequency):
        ts = TimeSeries(hourly_frequency)
        assert ts.name is None
        assert ts.unit is None
        assert ts.description is None
        assert ts.data_type is None
        assert ts.location is None
        assert ts.timeseries_type == TimeSeriesType.FLAT
        assert ts.attributes == {}


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

    def test_bool_false(self, hourly_frequency):
        assert bool(TimeSeries(hourly_frequency)) is False


class TestBeginEnd:
    def test_begin_end_non_empty(self, sample_ts):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert sample_ts.begin == base
        assert sample_ts.end == base + timedelta(hours=4)

    def test_begin_end_empty(self, hourly_frequency):
        ts = TimeSeries(hourly_frequency)
        assert ts.begin is None
        assert ts.end is None


class TestNumpy:
    def test_to_numpy(self, sample_ts):
        arr = sample_ts.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert len(arr) == 5
        assert arr[0] == 1.0
        assert np.isnan(arr[3])


class TestDataFrameAliases:
    def test_to_pd_df(self, sample_ts):
        df = sample_ts.to_pd_df()
        assert isinstance(df, pd.DataFrame)
        pd.testing.assert_frame_equal(df, sample_ts.to_pandas_dataframe())

    def test_to_pl_df(self, sample_ts):
        import polars as pl

        df = sample_ts.to_pl_df()
        assert isinstance(df, pl.DataFrame)
        assert df.equals(sample_ts.to_polars_dataframe())


class TestPandas:
    def test_to_pandas_dataframe(self, sample_ts):
        df = sample_ts.to_pandas_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.columns.tolist() == ["power"]
        assert len(df) == 5
        assert pd.isna(df.iloc[3, 0])

    def test_default_column_name(self, hourly_frequency):
        ts = TimeSeries(
            hourly_frequency,
            timestamps=[datetime(2024, 1, 1, tzinfo=timezone.utc)],
            values=[1.0],
        )
        df = ts.to_pandas_dataframe()
        assert df.columns.tolist() == ["value"]

    def test_round_trip(self, sample_ts):
        df = sample_ts.to_pandas_dataframe()
        ts2 = TimeSeries.from_pandas(df, sample_ts.frequency)
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
        ts2 = TimeSeries.from_polars(df, sample_ts.frequency)
        assert len(ts2) == len(sample_ts)
        assert ts2[0].value == sample_ts[0].value
        assert ts2[3].value is None


class TestTier1:
    def test_duration_non_empty(self, sample_ts):
        assert sample_ts.duration == timedelta(hours=4)

    def test_duration_empty(self, hourly_frequency):
        ts = TimeSeries(hourly_frequency)
        assert ts.duration is None

    def test_duration_single_point(self, hourly_frequency):
        ts = TimeSeries(
            hourly_frequency,
            timestamps=[datetime(2024, 1, 1, tzinfo=timezone.utc)],
            values=[1.0],
        )
        assert ts.duration == timedelta(0)

    def test_head_default(self, sample_ts):
        h = sample_ts.head()
        assert len(h) == 5

    def test_head_n(self, sample_ts):
        h = sample_ts.head(2)
        assert len(h) == 2
        assert h[0].value == 1.0
        assert h[1].value == 2.0

    def test_head_n_gt_len(self, sample_ts):
        h = sample_ts.head(100)
        assert len(h) == len(sample_ts)

    def test_head_zero(self, sample_ts):
        h = sample_ts.head(0)
        assert len(h) == 0

    def test_tail_default(self, sample_ts):
        t = sample_ts.tail()
        assert len(t) == 5

    def test_tail_n(self, sample_ts):
        t = sample_ts.tail(2)
        assert len(t) == 2
        assert t[0].value == sample_ts[-2].value
        assert t[1].value == sample_ts[-1].value

    def test_tail_n_gt_len(self, sample_ts):
        t = sample_ts.tail(100)
        assert len(t) == len(sample_ts)

    def test_tail_zero(self, sample_ts):
        t = sample_ts.tail(0)
        assert len(t) == 0

    def test_copy_independence(self, sample_ts):
        c = sample_ts.copy()
        assert len(c) == len(sample_ts)
        assert c[0].value == sample_ts[0].value
        c._values[0] = 999.0
        assert sample_ts[0].value == 1.0

    def test_has_missing_true(self, sample_ts):
        assert sample_ts.has_missing is True

    def test_has_missing_false(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = TimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1)],
            values=[1.0, 2.0],
        )
        assert ts.has_missing is False

    def test_has_missing_empty(self, hourly_frequency):
        ts = TimeSeries(hourly_frequency)
        assert ts.has_missing is False

    def test_contains_hit(self, sample_ts):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert base in sample_ts

    def test_contains_miss(self, sample_ts):
        assert datetime(2025, 6, 1, tzinfo=timezone.utc) not in sample_ts


class TestTier5Arithmetic:
    def test_add_scalar(self, sample_ts):
        ts2 = sample_ts + 10
        assert ts2[0].value == 11.0
        assert ts2[3].value is None

    def test_radd_scalar(self, sample_ts):
        ts2 = 10 + sample_ts
        assert ts2[0].value == 11.0

    def test_sub_scalar(self, sample_ts):
        ts2 = sample_ts - 1
        assert ts2[0].value == 0.0

    def test_rsub_scalar(self, sample_ts):
        ts2 = 10 - sample_ts
        assert ts2[0].value == 9.0
        assert ts2[3].value is None

    def test_mul_scalar(self, sample_ts):
        ts2 = sample_ts * 2
        assert ts2[0].value == 2.0
        assert ts2[3].value is None

    def test_rmul_scalar(self, sample_ts):
        ts2 = 2 * sample_ts
        assert ts2[0].value == 2.0

    def test_mul_zero(self, sample_ts):
        ts2 = sample_ts * 0
        assert ts2[0].value == 0.0

    def test_truediv_scalar(self, sample_ts):
        ts2 = sample_ts / 2
        assert ts2[0].value == 0.5
        assert ts2[3].value is None

    def test_neg(self, sample_ts):
        ts2 = -sample_ts
        assert ts2[0].value == -1.0
        assert ts2[3].value is None

    def test_abs(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = TimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1)],
            values=[-3.0, 4.0],
        )
        ts2 = abs(ts)
        assert ts2[0].value == 3.0
        assert ts2[1].value == 4.0

    def test_round(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = TimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1)],
            values=[1.567, 2.345],
        )
        ts2 = round(ts, 1)
        assert ts2[0].value == 1.6
        assert ts2[1].value == 2.3

    def test_unsupported_operand_returns_not_implemented(self, sample_ts):
        result = sample_ts.__add__("not_a_number")
        assert result is NotImplemented

    def test_immutability(self, sample_ts):
        ts2 = sample_ts + 1
        assert sample_ts[0].value == 1.0


class TestTier6IO:
    def test_json_round_trip(self, sample_ts):
        s = sample_ts.to_json()
        ts2 = TimeSeries.from_json(s, sample_ts.frequency)
        assert len(ts2) == len(sample_ts)
        for i, (orig, restored) in enumerate(zip(sample_ts, ts2)):
            assert orig.timestamp == restored.timestamp, f"timestamp mismatch at {i}"
            assert orig.value == restored.value, f"value mismatch at {i}"

    def test_json_is_valid(self, sample_ts):
        s = sample_ts.to_json()
        parsed = json.loads(s)
        assert "timestamps" in parsed
        assert "values" in parsed
        assert len(parsed["timestamps"]) == len(sample_ts)

    def test_json_preserves_none(self, sample_ts):
        s = sample_ts.to_json()
        ts2 = TimeSeries.from_json(s, sample_ts.frequency)
        assert ts2[3].value is None

    def test_json_empty(self, hourly_frequency):
        ts = TimeSeries(hourly_frequency)
        s = ts.to_json()
        ts2 = TimeSeries.from_json(s, hourly_frequency)
        assert len(ts2) == 0

    def test_csv_round_trip(self, sample_ts):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            sample_ts.to_csv(path)
            ts2 = TimeSeries.from_csv(path, sample_ts.frequency)
            assert len(ts2) == len(sample_ts)
            for orig, restored in zip(sample_ts, ts2):
                assert orig.timestamp == restored.timestamp
                assert orig.value == restored.value
        finally:
            path.unlink(missing_ok=True)

    def test_csv_preserves_none(self, sample_ts):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            sample_ts.to_csv(path)
            ts2 = TimeSeries.from_csv(path, sample_ts.frequency)
            assert ts2[3].value is None
        finally:
            path.unlink(missing_ok=True)

    def test_csv_column_name(self, sample_ts):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            sample_ts.to_csv(path)
            ts2 = TimeSeries.from_csv(path, sample_ts.frequency)
            assert ts2.name == "power"
        finally:
            path.unlink(missing_ok=True)


class TestFromPandasAutoInfer:
    def test_infer_hourly(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        index = pd.DatetimeIndex(
            [base + timedelta(hours=i) for i in range(5)], freq="h"
        )
        df = pd.DataFrame({"power": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
        ts = TimeSeries.from_pandas(df)
        assert ts.frequency == Frequency.PT1H
        assert ts.name == "power"
        assert len(ts) == 5

    def test_infer_daily(self):
        index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        df = pd.DataFrame({"temp": range(5)}, index=index)
        ts = TimeSeries.from_pandas(df)
        assert ts.frequency == Frequency.P1D

    def test_infer_from_few_points_falls_back(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        index = pd.DatetimeIndex([base, base + timedelta(hours=1)])
        df = pd.DataFrame({"v": [1.0, 2.0]}, index=index)
        ts = TimeSeries.from_pandas(df)
        assert ts.frequency == Frequency.NONE

    def test_infer_timezone(self):
        index = pd.date_range("2024-01-01", periods=5, freq="h", tz="Europe/Berlin")
        df = pd.DataFrame({"v": range(5)}, index=index)
        ts = TimeSeries.from_pandas(df)
        assert ts.timezone == "Europe/Berlin"

    def test_explicit_frequency_still_works(self):
        index = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=index)
        ts = TimeSeries.from_pandas(df, Frequency.P1D, timezone="US/Eastern")
        assert ts.frequency == Frequency.P1D
        assert ts.timezone == "US/Eastern"

    def test_no_datetime_index_raises(self):
        df = pd.DataFrame({"v": [1, 2, 3]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            TimeSeries.from_pandas(df)


class TestUpdateFromPandas:
    def test_returns_new_by_default(self, sample_ts):
        new_index = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({"energy": [10.0, 20.0, 30.0]}, index=new_index)
        result = sample_ts.update_from_pandas(df)
        assert isinstance(result, TimeSeries)
        assert len(result) == 3
        assert result[0].value == 10.0
        assert result.frequency == Frequency.P1D
        assert result.name == "energy"
        assert len(sample_ts) == 5
        assert sample_ts[0].value == 1.0

    def test_inplace_returns_none(self, sample_ts):
        new_index = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({"energy": [10.0, 20.0, 30.0]}, index=new_index)
        result = sample_ts.update_from_pandas(df, inplace=True)
        assert result is None
        assert len(sample_ts) == 3
        assert sample_ts[0].value == 10.0
        assert sample_ts.frequency == Frequency.P1D
        assert sample_ts.name == "energy"

    def test_value_column(self, sample_ts):
        new_index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=new_index)
        result = sample_ts.update_from_pandas(df, value_column="b")
        assert result[0].value == 3.0
        assert result[1].value == 4.0

    def test_no_datetime_index_raises(self, sample_ts):
        df = pd.DataFrame({"v": [1, 2]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            sample_ts.update_from_pandas(df)


class TestApplyPandas:
    def test_ffill(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df.ffill())
        assert ts2[3].value is not None
        assert ts2[3].value == sample_ts[2].value

    def test_clip(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df.clip(lower=2))
        assert ts2[0].value == 2.0

    def test_arithmetic(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df * 10)
        assert ts2[0].value == 10.0

    def test_metadata_preserved(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df.ffill())
        assert ts2.name == sample_ts.name
        assert ts2.unit == sample_ts.unit
        assert ts2.data_type == sample_ts.data_type

    def test_frequency_unchanged_for_noop(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df.ffill())
        assert ts2.frequency == sample_ts.frequency
        assert ts2.timezone == sample_ts.timezone

    def test_frequency_updated_after_resample(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df.resample("2h").mean())
        assert ts2.frequency == Frequency.PT1H

    def test_timezone_updated_after_tz_convert(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df.tz_convert("Europe/Berlin"))
        assert ts2.timezone == "Europe/Berlin"

    def test_none_roundtrip(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df)
        assert ts2[3].value is None

    def test_immutability(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df + 99)
        assert sample_ts[0].value == 1.0

    def test_column_rename_updates_name(self, sample_ts):
        ts2 = sample_ts.apply_pandas(
            lambda df: df.rename(columns={"power": "power_filled"})
        )
        assert ts2.name == "power_filled"
        assert ts2.unit == sample_ts.unit

    def test_no_rename_preserves_name(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df.ffill())
        assert ts2.name == "power"

    def test_unit_preserved_after_arithmetic(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df * 0.001)
        assert ts2.unit == sample_ts.unit


class TestApplyNumpy:
    def test_clip(self, sample_ts):
        ts2 = sample_ts.apply_numpy(lambda arr: np.clip(arr, 2, None))
        assert ts2[0].value == 2.0

    def test_arithmetic(self, sample_ts):
        ts2 = sample_ts.apply_numpy(lambda arr: arr * 10)
        assert ts2[0].value == 10.0

    def test_sqrt(self, sample_ts):
        ts2 = sample_ts.apply_numpy(np.sqrt)
        assert ts2[0].value == pytest.approx(1.0)
        assert ts2[1].value == pytest.approx(np.sqrt(2.0))

    def test_nan_passthrough(self, sample_ts):
        ts2 = sample_ts.apply_numpy(lambda arr: arr)
        assert ts2[3].value is None

    def test_nan_fill(self, sample_ts):
        ts2 = sample_ts.apply_numpy(lambda arr: np.nan_to_num(arr, nan=0.0))
        assert ts2[3].value == 0.0

    def test_timestamps_unchanged(self, sample_ts):
        ts2 = sample_ts.apply_numpy(lambda arr: arr + 1)
        assert ts2.timestamps == sample_ts.timestamps

    def test_frequency_unchanged(self, sample_ts):
        ts2 = sample_ts.apply_numpy(lambda arr: arr + 1)
        assert ts2.frequency == sample_ts.frequency
        assert ts2.timezone == sample_ts.timezone

    def test_metadata_preserved(self, sample_ts):
        ts2 = sample_ts.apply_numpy(lambda arr: arr + 1)
        assert ts2.name == sample_ts.name
        assert ts2.unit == sample_ts.unit
        assert ts2.data_type == sample_ts.data_type

    def test_immutability(self, sample_ts):
        ts2 = sample_ts.apply_numpy(lambda arr: arr + 99)
        assert sample_ts[0].value == 1.0

    def test_wrong_length_raises(self, sample_ts):
        with pytest.raises(ValueError, match="result length"):
            sample_ts.apply_numpy(lambda arr: arr[:-1])

    def test_nancumsum(self, sample_ts):
        ts2 = sample_ts.apply_numpy(np.nancumsum)
        assert ts2[0].value == pytest.approx(1.0)
        assert ts2[1].value == pytest.approx(3.0)
        assert ts2[2].value == pytest.approx(6.0)
        assert ts2[3].value == pytest.approx(6.0)
        assert ts2[4].value == pytest.approx(11.0)


class TestValidation:
    def test_valid(self, sample_ts):
        warnings = sample_ts.validate()
        assert warnings == []

    def test_unordered(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = TimeSeries(
            hourly_frequency,
            timestamps=[base + timedelta(hours=1), base],
            values=[1.0, 2.0],
        )
        warnings = ts.validate()
        assert any("not strictly increasing" in w for w in warnings)

    def test_inconsistent_frequency(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = TimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1), base + timedelta(hours=3)],
            values=[1.0, 2.0, 3.0],
        )
        warnings = ts.validate()
        assert any("inconsistent frequency" in w for w in warnings)


class TestPintUnit:
    def test_pint_unit_valid(self):
        ts = TimeSeries(
            Frequency.PT1H,
            timestamps=[datetime(2024, 1, 1, tzinfo=timezone.utc)],
            values=[1.0],
            unit="MW",
        )
        u = ts.pint_unit
        assert str(u) == "megawatt"

    def test_pint_unit_none(self):
        ts = TimeSeries(Frequency.PT1H)
        with pytest.raises(ValueError, match="unit is not set"):
            ts.pint_unit

    def test_pint_unit_invalid(self):
        ts = TimeSeries(
            Frequency.PT1H,
            timestamps=[datetime(2024, 1, 1, tzinfo=timezone.utc)],
            values=[1.0],
            unit="not_a_real_unit_xyz",
        )
        with pytest.raises(ValueError, match="invalid unit string"):
            ts.pint_unit


# ---- Multi-Index Tests ------------------------------------------------


class TestMultiIndex:
    @pytest.fixture
    def base_times(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return [base + timedelta(hours=i) for i in range(5)]

    @pytest.fixture
    def multi_ts(self, hourly_frequency, base_times):
        knowledge_time = datetime(2024, 1, 2, tzinfo=timezone.utc)
        timestamps = [(vt, knowledge_time) for vt in base_times]
        values = [1.0, 2.0, 3.0, None, 5.0]
        return TimeSeries(
            hourly_frequency,
            timestamps=timestamps,
            values=values,
            index_names=["valid_time", "knowledge_time"],
        )

    def test_construction(self, multi_ts):
        assert len(multi_ts) == 5

    def test_is_multi_index_true(self, multi_ts):
        assert multi_ts.is_multi_index is True

    def test_is_multi_index_false(self, sample_ts):
        assert sample_ts.is_multi_index is False

    def test_index_names(self, multi_ts):
        assert multi_ts.index_names == ("valid_time", "knowledge_time")

    def test_index_names_default(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        kt = datetime(2024, 1, 2, tzinfo=timezone.utc)
        ts = TimeSeries(
            hourly_frequency,
            timestamps=[(base, kt)],
            values=[1.0],
        )
        assert ts.index_names == ("index_0", "index_1")

    def test_begin_end(self, multi_ts, base_times):
        knowledge_time = datetime(2024, 1, 2, tzinfo=timezone.utc)
        assert multi_ts.begin == (base_times[0], knowledge_time)
        assert multi_ts.end == (base_times[-1], knowledge_time)

    def test_duration(self, multi_ts):
        assert multi_ts.duration == timedelta(hours=4)

    def test_contains(self, multi_ts):
        knowledge_time = datetime(2024, 1, 2, tzinfo=timezone.utc)
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert (base, knowledge_time) in multi_ts

    def test_head(self, multi_ts):
        h = multi_ts.head(2)
        assert len(h) == 2
        assert h.is_multi_index

    def test_tail(self, multi_ts):
        t = multi_ts.tail(2)
        assert len(t) == 2
        assert t.is_multi_index

    def test_copy(self, multi_ts):
        c = multi_ts.copy()
        assert len(c) == len(multi_ts)
        assert c.is_multi_index
        c._values[0] = 999.0
        assert multi_ts.values[0] == 1.0

    def test_validate(self, multi_ts):
        warnings = multi_ts.validate()
        assert warnings == []

    def test_to_pandas_multiindex(self, multi_ts):
        df = multi_ts.to_pandas_dataframe()
        assert isinstance(df.index, pd.MultiIndex)
        assert list(df.index.names) == ["valid_time", "knowledge_time"]
        assert len(df) == 5

    def test_from_pandas_multiindex_round_trip(self, multi_ts, hourly_frequency):
        df = multi_ts.to_pandas_dataframe()
        ts2 = TimeSeries.from_pandas(df, hourly_frequency)
        assert ts2.is_multi_index
        assert len(ts2) == len(multi_ts)
        assert ts2.values[0] == multi_ts.values[0]

    def test_json_round_trip(self, multi_ts, hourly_frequency):
        s = multi_ts.to_json()
        ts2 = TimeSeries.from_json(s, hourly_frequency)
        assert ts2.is_multi_index
        assert len(ts2) == len(multi_ts)
        assert ts2.timestamps[0] == multi_ts.timestamps[0]
        assert ts2.values[3] is None

    def test_csv_round_trip(self, multi_ts, hourly_frequency):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            multi_ts.to_csv(path)
            ts2 = TimeSeries.from_csv(path, hourly_frequency)
            assert ts2.is_multi_index
            assert len(ts2) == len(multi_ts)
            assert ts2.values[3] is None
        finally:
            path.unlink(missing_ok=True)

    def test_repr(self, multi_ts):
        r = repr(multi_ts)
        assert "TimeSeries" in r
        assert "(5,)" in r
        assert "\u250c" in r  # box-drawing top-left


# ---- Multivariate Tests -----------------------------------------------


class TestMultivariate:
    @pytest.fixture
    def mv_ts(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = [base + timedelta(hours=i) for i in range(5)]
        values = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [np.nan, 40.0],
            [5.0, 50.0],
        ])
        return MultivariateTimeSeries(
            hourly_frequency,
            timestamps=timestamps,
            values=values,
            names=["power", "temperature"],
        )

    def test_construction(self, mv_ts):
        assert len(mv_ts) == 5

    def test_n_columns(self, mv_ts):
        assert mv_ts.n_columns == 2

    def test_column_names(self, mv_ts):
        assert mv_ts.column_names == ("power", "temperature")

    def test_column_names_default(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        ts = MultivariateTimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1)],
            values=values,
        )
        assert ts.column_names == ("value_0", "value_1")

    def test_has_missing_true(self, mv_ts):
        assert mv_ts.has_missing is True

    def test_has_missing_false(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        ts = MultivariateTimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1)],
            values=values,
        )
        assert ts.has_missing is False

    def test_add_scalar(self, mv_ts):
        ts2 = mv_ts + 10
        arr = ts2.to_numpy()
        assert arr[0, 0] == 11.0
        assert arr[0, 1] == 20.0
        assert np.isnan(arr[3, 0])

    def test_mul_scalar(self, mv_ts):
        ts2 = mv_ts * 2
        arr = ts2.to_numpy()
        assert arr[1, 0] == 4.0
        assert arr[1, 1] == 40.0

    def test_to_numpy_2d(self, mv_ts):
        arr = mv_ts.to_numpy()
        assert arr.ndim == 2
        assert arr.shape == (5, 2)

    def test_to_pandas_multicolumn(self, mv_ts):
        df = mv_ts.to_pandas_dataframe()
        assert isinstance(df.index, pd.DatetimeIndex)
        assert list(df.columns) == ["power", "temperature"]
        assert len(df) == 5

    def test_from_pandas_multicolumn_round_trip(self, mv_ts, hourly_frequency):
        df = mv_ts.to_pandas_dataframe()
        ts2 = MultivariateTimeSeries.from_pandas(df, hourly_frequency)
        assert ts2.n_columns == 2
        np.testing.assert_array_equal(ts2.to_numpy(), mv_ts.to_numpy())

    def test_json_round_trip(self, mv_ts, hourly_frequency):
        s = mv_ts.to_json()
        ts2 = MultivariateTimeSeries.from_json(s, hourly_frequency)
        assert ts2.n_columns == 2
        np.testing.assert_array_equal(ts2.to_numpy(), mv_ts.to_numpy())

    def test_head(self, mv_ts):
        h = mv_ts.head(2)
        assert len(h) == 2
        assert isinstance(h, MultivariateTimeSeries)
        assert h.n_columns == 2

    def test_tail(self, mv_ts):
        t = mv_ts.tail(2)
        assert len(t) == 2
        assert isinstance(t, MultivariateTimeSeries)

    def test_copy(self, mv_ts):
        c = mv_ts.copy()
        assert len(c) == len(mv_ts)
        assert isinstance(c, MultivariateTimeSeries)
        c._values[0, 0] = 999.0
        assert mv_ts.values[0, 0] == 1.0

    def test_getitem_returns_tuple(self, mv_ts):
        item = mv_ts[0]
        assert isinstance(item, tuple)
        assert not isinstance(item, DataPoint)
        assert item[1] == [1.0, 10.0]

    def test_iter_returns_tuples(self, mv_ts):
        items = list(mv_ts)
        assert all(isinstance(item, tuple) for item in items)

    def test_repr(self, mv_ts):
        r = repr(mv_ts)
        assert "MultivariateTimeSeries" in r
        assert "(5, 2)" in r
        assert "\u250c" in r  # box-drawing top-left

    def test_validate(self, mv_ts):
        warnings = mv_ts.validate()
        assert warnings == []

    def test_csv_round_trip(self, mv_ts, hourly_frequency):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            mv_ts.to_csv(path)
            ts2 = MultivariateTimeSeries.from_csv(path, hourly_frequency)
            assert ts2.n_columns == 2
            assert np.isnan(ts2.to_numpy()[3, 0])
            assert ts2.to_numpy()[0, 0] == 1.0
        finally:
            path.unlink(missing_ok=True)

    def test_round(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[1.567, 2.345], [3.891, 4.123]])
        ts = MultivariateTimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1)],
            values=values,
        )
        ts2 = round(ts, 1)
        arr = ts2.to_numpy()
        assert arr[0, 0] == pytest.approx(1.6)
        assert arr[0, 1] == pytest.approx(2.3)


# ---- Broadcast Validation Tests ------------------------------------------


class TestBroadcast:
    def test_broadcast_single_unit(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        ts = MultivariateTimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1)],
            values=values,
            names=["a", "b"],
            units=["MW"],
        )
        assert ts.units == ["MW"]
        assert ts._get_attr(ts.units, 0) == "MW"
        assert ts._get_attr(ts.units, 1) == "MW"

    def test_broadcast_per_column(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        ts = MultivariateTimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1)],
            values=values,
            names=["power", "temperature"],
            units=["MW", "°C"],
        )
        assert ts._get_attr(ts.units, 0) == "MW"
        assert ts._get_attr(ts.units, 1) == "°C"

    def test_broadcast_invalid_length_raises(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with pytest.raises(ValueError, match="must have length 1 or 3"):
            MultivariateTimeSeries(
                hourly_frequency,
                timestamps=[base, base + timedelta(hours=1)],
                values=values,
                names=["a", "b"],
            )

    def test_broadcast_default_names(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        ts = MultivariateTimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1)],
            values=values,
        )
        assert ts.names == [None]
        assert ts.column_names == ("value_0", "value_1")


# ---- MultiTimeSeries Alias Test -----------------------------------------


class TestMultiTimeSeriesAlias:
    def test_alias_is_same_class(self):
        assert MultiTimeSeries is MultivariateTimeSeries

    def test_alias_construction(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[1.0, 2.0]])
        ts = MultiTimeSeries(
            hourly_frequency,
            timestamps=[base],
            values=values,
            names=["a", "b"],
        )
        assert isinstance(ts, MultivariateTimeSeries)
        assert ts.n_columns == 2


# ---- Backward Compatibility Tests ----------------------------------------


class TestBackwardCompatibility:
    def test_single_index_construction(self, sample_ts):
        assert len(sample_ts) == 5
        assert sample_ts.is_multi_index is False

    def test_datapoint_from_getitem(self, sample_ts):
        dp = sample_ts[0]
        assert isinstance(dp, DataPoint)
        assert dp.value == 1.0
        assert dp.timestamp == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_values_returns_list(self, sample_ts):
        assert isinstance(sample_ts.values, list)

    def test_iter_returns_datapoints(self, sample_ts):
        points = list(sample_ts)
        assert all(isinstance(p, DataPoint) for p in points)

    def test_index_names_default(self, sample_ts):
        assert sample_ts.index_names == ("timestamp",)


# ---- Multivariate Conversion Tests ----------------------------------------


class TestMultivariateConversion:
    @pytest.fixture
    def mv_ts(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = [base + timedelta(hours=i) for i in range(5)]
        values = np.array([
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [np.nan, 40.0, 400.0],
            [5.0, 50.0, 500.0],
        ])
        return MultivariateTimeSeries(
            hourly_frequency,
            timestamps=timestamps,
            values=values,
            names=["power", "temperature", "pressure"],
            units=["MW", "°C", "hPa"],
            descriptions=["Power output", "Ambient temp", "Atmospheric pressure"],
        )

    def test_select_column_by_index(self, mv_ts):
        ts = mv_ts.select_column(0)
        assert isinstance(ts, TimeSeries)
        assert ts.name == "power"
        assert ts.unit == "MW"
        assert ts.description == "Power output"
        assert len(ts) == 5
        assert ts[0].value == 1.0
        assert ts[3].value is None  # NaN -> None

    def test_select_column_by_name(self, mv_ts):
        ts = mv_ts.select_column("temperature")
        assert isinstance(ts, TimeSeries)
        assert ts.name == "temperature"
        assert ts.unit == "°C"
        assert len(ts) == 5
        assert ts[0].value == 10.0

    def test_select_column_invalid_name_raises(self, mv_ts):
        with pytest.raises(KeyError, match="nonexistent"):
            mv_ts.select_column("nonexistent")

    def test_select_column_preserves_timestamps(self, mv_ts):
        ts = mv_ts.select_column(1)
        assert ts.timestamps == mv_ts.timestamps

    def test_to_univariate_list(self, mv_ts):
        series_list = mv_ts.to_univariate_list()
        assert len(series_list) == 3
        assert all(isinstance(s, TimeSeries) for s in series_list)
        assert series_list[0].name == "power"
        assert series_list[1].name == "temperature"
        assert series_list[2].name == "pressure"

    def test_to_univariate_list_values(self, mv_ts):
        series_list = mv_ts.to_univariate_list()
        assert series_list[2][0].value == 100.0
        assert series_list[2][4].value == 500.0

    def test_merge_roundtrip(self, mv_ts):
        """select_column -> merge should reconstruct the original."""
        series_list = mv_ts.to_univariate_list()
        merged = TimeSeries.merge(series_list)
        assert isinstance(merged, MultivariateTimeSeries)
        assert merged.n_columns == 3
        assert merged.column_names == ("power", "temperature", "pressure")
        np.testing.assert_array_equal(merged.to_numpy(), mv_ts.to_numpy())

    def test_merge_empty_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            TimeSeries.merge([])

    def test_merge_mismatched_timestamps_raises(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts1 = TimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1)],
            values=[1.0, 2.0],
            name="a",
        )
        ts2 = TimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=2)],
            values=[3.0, 4.0],
            name="b",
        )
        with pytest.raises(ValueError, match="do not match"):
            TimeSeries.merge([ts1, ts2])

    def test_merge_preserves_metadata(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts1 = TimeSeries(
            hourly_frequency,
            timestamps=[base],
            values=[1.0],
            name="x", unit="MW", description="desc_x",
        )
        ts2 = TimeSeries(
            hourly_frequency,
            timestamps=[base],
            values=[2.0],
            name="y", unit="°C", description="desc_y",
        )
        merged = TimeSeries.merge([ts1, ts2])
        assert merged.names == ["x", "y"]
        assert merged.units == ["MW", "°C"]
        assert merged.descriptions == ["desc_x", "desc_y"]

    def test_select_column_broadcast_unit(self, hourly_frequency):
        """When a single broadcast unit is used, select_column should resolve it."""
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        mv = MultivariateTimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1)],
            values=values,
            names=["a", "b"],
            units=["MW"],  # broadcast
        )
        ts0 = mv.select_column(0)
        ts1 = mv.select_column(1)
        assert ts0.unit == "MW"
        assert ts1.unit == "MW"
