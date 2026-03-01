import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import timedatamodel as tdm


@pytest.fixture
def hourly_frequency():
    return tdm.Frequency.PT1H


@pytest.fixture
def sample_ts(hourly_frequency):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [base + timedelta(hours=i) for i in range(5)]
    values = [1.0, 2.0, 3.0, None, 5.0]
    return tdm.TimeSeries(
        hourly_frequency,
        timestamps=timestamps,
        values=values,
        name="power",
        unit="MW",
        data_type=tdm.DataType.ACTUAL,
    )


class TestConstruction:
    def test_from_lists(self, hourly_frequency):
        ts = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[datetime(2024, 1, 1, tzinfo=timezone.utc)],
            values=[42.0],
        )
        assert len(ts) == 1
        assert ts[0].value == 42.0

    def test_from_data(self, hourly_frequency):
        dp = tdm.DataPoint(datetime(2024, 1, 1, tzinfo=timezone.utc), 42.0)
        ts = tdm.TimeSeries(hourly_frequency, data=[dp])
        assert len(ts) == 1
        assert ts[0] == dp

    def test_both_raises(self, hourly_frequency):
        with pytest.raises(ValueError, match="cannot specify both"):
            tdm.TimeSeries(
                hourly_frequency,
                timestamps=[datetime(2024, 1, 1, tzinfo=timezone.utc)],
                values=[1.0],
                data=[tdm.DataPoint(datetime(2024, 1, 1, tzinfo=timezone.utc), 1.0)],
            )

    def test_empty(self, hourly_frequency):
        ts = tdm.TimeSeries(hourly_frequency)
        assert len(ts) == 0
        assert not ts
        assert list(ts) == []

    def test_scalar_metadata(self, hourly_frequency):
        loc = tdm.GeoLocation(latitude=59.91, longitude=10.75)
        ts = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[datetime(2024, 1, 1, tzinfo=timezone.utc)],
            values=[1.0],
            name="power",
            unit="MW",
            description="Hourly power",
            data_type=tdm.DataType.ACTUAL,
            location=loc,
            timeseries_type=tdm.TimeSeriesType.FLAT,
            attributes={"source": "test"},
        )
        assert ts.name == "power"
        assert ts.unit == "MW"
        assert ts.description == "Hourly power"
        assert ts.data_type == tdm.DataType.ACTUAL
        assert ts.location == loc
        assert ts.timeseries_type == tdm.TimeSeriesType.FLAT
        assert ts.attributes["source"] == "test"

    def test_defaults(self, hourly_frequency):
        ts = tdm.TimeSeries(hourly_frequency)
        assert ts.name is None
        assert ts.unit is None
        assert ts.description is None
        assert ts.data_type is None
        assert ts.location is None
        assert ts.timeseries_type == tdm.TimeSeriesType.FLAT
        assert ts.attributes == {}


class TestSequenceProtocol:
    def test_len(self, sample_ts):
        assert len(sample_ts) == 5

    def test_getitem(self, sample_ts):
        dp = sample_ts[0]
        assert isinstance(dp, tdm.DataPoint)
        assert dp.value == 1.0

    def test_getitem_slice(self, sample_ts):
        result = sample_ts[1:3]
        assert len(result) == 2
        assert result[0].value == 2.0
        assert result[1].value == 3.0

    def test_iter(self, sample_ts):
        points = list(sample_ts)
        assert len(points) == 5
        assert all(isinstance(p, tdm.DataPoint) for p in points)

    def test_bool_true(self, sample_ts):
        assert bool(sample_ts) is True

    def test_bool_false(self, hourly_frequency):
        assert bool(tdm.TimeSeries(hourly_frequency)) is False


class TestBeginEnd:
    def test_begin_end_non_empty(self, sample_ts):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert sample_ts.begin == base
        assert sample_ts.end == base + timedelta(hours=4)

    def test_begin_end_empty(self, hourly_frequency):
        ts = tdm.TimeSeries(hourly_frequency)
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
        ts = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[datetime(2024, 1, 1, tzinfo=timezone.utc)],
            values=[1.0],
        )
        df = ts.to_pandas_dataframe()
        assert df.columns.tolist() == ["value"]

    def test_round_trip(self, sample_ts):
        df = sample_ts.to_pandas_dataframe()
        ts2 = tdm.TimeSeries.from_pandas(df, sample_ts.frequency)
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
        ts2 = tdm.TimeSeries.from_polars(df, sample_ts.frequency)
        assert len(ts2) == len(sample_ts)
        assert ts2[0].value == sample_ts[0].value
        assert ts2[3].value is None


class TestTier1:
    def test_duration_non_empty(self, sample_ts):
        assert sample_ts.duration == timedelta(hours=4)

    def test_duration_empty(self, hourly_frequency):
        ts = tdm.TimeSeries(hourly_frequency)
        assert ts.duration is None

    def test_duration_single_point(self, hourly_frequency):
        ts = tdm.TimeSeries(
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
        ts = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1)],
            values=[1.0, 2.0],
        )
        assert ts.has_missing is False

    def test_has_missing_empty(self, hourly_frequency):
        ts = tdm.TimeSeries(hourly_frequency)
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
        ts = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1)],
            values=[-3.0, 4.0],
        )
        ts2 = abs(ts)
        assert ts2[0].value == 3.0
        assert ts2[1].value == 4.0

    def test_round(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = tdm.TimeSeries(
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
        ts2 = tdm.TimeSeries.from_json(s, sample_ts.frequency)
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
        ts2 = tdm.TimeSeries.from_json(s, sample_ts.frequency)
        assert ts2[3].value is None

    def test_json_empty(self, hourly_frequency):
        ts = tdm.TimeSeries(hourly_frequency)
        s = ts.to_json()
        ts2 = tdm.TimeSeries.from_json(s, hourly_frequency)
        assert len(ts2) == 0

    def test_csv_round_trip(self, sample_ts):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            sample_ts.to_csv(path)
            ts2 = tdm.TimeSeries.from_csv(path, sample_ts.frequency)
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
            ts2 = tdm.TimeSeries.from_csv(path, sample_ts.frequency)
            assert ts2[3].value is None
        finally:
            path.unlink(missing_ok=True)

    def test_csv_column_name(self, sample_ts):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            sample_ts.to_csv(path)
            ts2 = tdm.TimeSeries.from_csv(path, sample_ts.frequency)
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
        ts = tdm.TimeSeries.from_pandas(df)
        assert ts.frequency == tdm.Frequency.PT1H
        assert ts.name == "power"
        assert len(ts) == 5

    def test_infer_daily(self):
        index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        df = pd.DataFrame({"temp": range(5)}, index=index)
        ts = tdm.TimeSeries.from_pandas(df)
        assert ts.frequency == tdm.Frequency.P1D

    def test_infer_from_few_points_falls_back(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        index = pd.DatetimeIndex([base, base + timedelta(hours=1)])
        df = pd.DataFrame({"v": [1.0, 2.0]}, index=index)
        ts = tdm.TimeSeries.from_pandas(df)
        assert ts.frequency == tdm.Frequency.NONE

    def test_infer_timezone(self):
        index = pd.date_range("2024-01-01", periods=5, freq="h", tz="Europe/Berlin")
        df = pd.DataFrame({"v": range(5)}, index=index)
        ts = tdm.TimeSeries.from_pandas(df)
        assert ts.timezone == "Europe/Berlin"

    def test_explicit_frequency_still_works(self):
        index = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=index)
        ts = tdm.TimeSeries.from_pandas(df, tdm.Frequency.P1D, timezone="US/Eastern")
        assert ts.frequency == tdm.Frequency.P1D
        assert ts.timezone == "US/Eastern"

    def test_no_datetime_index_raises(self):
        df = pd.DataFrame({"v": [1, 2, 3]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            tdm.TimeSeries.from_pandas(df)


class TestUpdateFromPandas:
    def test_returns_new_by_default(self, sample_ts):
        new_index = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({"energy": [10.0, 20.0, 30.0]}, index=new_index)
        result = sample_ts.update_from_pandas(df)
        assert isinstance(result, tdm.TimeSeries)
        assert len(result) == 3
        assert result[0].value == 10.0
        assert result.frequency == tdm.Frequency.P1D
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
        assert sample_ts.frequency == tdm.Frequency.P1D
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
        assert ts2.frequency == tdm.Frequency.PT1H

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
        ts = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base + timedelta(hours=1), base],
            values=[1.0, 2.0],
        )
        warnings = ts.validate()
        assert any("not strictly increasing" in w for w in warnings)

    def test_inconsistent_frequency(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1), base + timedelta(hours=3)],
            values=[1.0, 2.0, 3.0],
        )
        warnings = ts.validate()
        assert any("inconsistent frequency" in w for w in warnings)


class TestPintUnit:
    def test_pint_unit_valid(self):
        ts = tdm.TimeSeries(
            tdm.Frequency.PT1H,
            timestamps=[datetime(2024, 1, 1, tzinfo=timezone.utc)],
            values=[1.0],
            unit="MW",
        )
        u = ts.pint_unit
        assert str(u) == "megawatt"

    def test_pint_unit_none(self):
        ts = tdm.TimeSeries(tdm.Frequency.PT1H)
        with pytest.raises(ValueError, match="unit is not set"):
            ts.pint_unit

    def test_pint_unit_invalid(self):
        ts = tdm.TimeSeries(
            tdm.Frequency.PT1H,
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
        return tdm.TimeSeries(
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
        ts = tdm.TimeSeries(
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
        ts2 = tdm.TimeSeries.from_pandas(df, hourly_frequency)
        assert ts2.is_multi_index
        assert len(ts2) == len(multi_ts)
        assert ts2.values[0] == multi_ts.values[0]

    def test_json_round_trip(self, multi_ts, hourly_frequency):
        s = multi_ts.to_json()
        ts2 = tdm.TimeSeries.from_json(s, hourly_frequency)
        assert ts2.is_multi_index
        assert len(ts2) == len(multi_ts)
        assert ts2.timestamps[0] == multi_ts.timestamps[0]
        assert ts2.values[3] is None

    def test_csv_round_trip(self, multi_ts, hourly_frequency):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            multi_ts.to_csv(path)
            ts2 = tdm.TimeSeries.from_csv(path, hourly_frequency)
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
        return tdm.MultivariateTimeSeries(
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
        ts = tdm.MultivariateTimeSeries(
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
        ts = tdm.MultivariateTimeSeries(
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
        ts2 = tdm.MultivariateTimeSeries.from_pandas(df, hourly_frequency)
        assert ts2.n_columns == 2
        np.testing.assert_array_equal(ts2.to_numpy(), mv_ts.to_numpy())

    def test_json_round_trip(self, mv_ts, hourly_frequency):
        s = mv_ts.to_json()
        ts2 = tdm.MultivariateTimeSeries.from_json(s, hourly_frequency)
        assert ts2.n_columns == 2
        np.testing.assert_array_equal(ts2.to_numpy(), mv_ts.to_numpy())

    def test_head(self, mv_ts):
        h = mv_ts.head(2)
        assert len(h) == 2
        assert isinstance(h, tdm.MultivariateTimeSeries)
        assert h.n_columns == 2

    def test_tail(self, mv_ts):
        t = mv_ts.tail(2)
        assert len(t) == 2
        assert isinstance(t, tdm.MultivariateTimeSeries)

    def test_copy(self, mv_ts):
        c = mv_ts.copy()
        assert len(c) == len(mv_ts)
        assert isinstance(c, tdm.MultivariateTimeSeries)
        c._values[0, 0] = 999.0
        assert mv_ts.values[0, 0] == 1.0

    def test_getitem_returns_tuple(self, mv_ts):
        item = mv_ts[0]
        assert isinstance(item, tuple)
        assert not isinstance(item, tdm.DataPoint)
        assert item[1] == [1.0, 10.0]

    def test_iter_returns_tuples(self, mv_ts):
        items = list(mv_ts)
        assert all(isinstance(item, tuple) for item in items)

    def test_repr(self, mv_ts):
        r = repr(mv_ts)
        assert "TimeSeriesTable" in r
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
            ts2 = tdm.MultivariateTimeSeries.from_csv(path, hourly_frequency)
            assert ts2.n_columns == 2
            assert np.isnan(ts2.to_numpy()[3, 0])
            assert ts2.to_numpy()[0, 0] == 1.0
        finally:
            path.unlink(missing_ok=True)

    def test_round(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[1.567, 2.345], [3.891, 4.123]])
        ts = tdm.MultivariateTimeSeries(
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
        ts = tdm.MultivariateTimeSeries(
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
        ts = tdm.MultivariateTimeSeries(
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
            tdm.MultivariateTimeSeries(
                hourly_frequency,
                timestamps=[base, base + timedelta(hours=1)],
                values=values,
                names=["a", "b"],
            )

    def test_broadcast_default_names(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        ts = tdm.MultivariateTimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1)],
            values=values,
        )
        assert ts.names == [None]
        assert ts.column_names == ("value_0", "value_1")


# ---- MultiTimeSeries Alias Test -----------------------------------------


class TestMultiTimeSeriesAlias:
    def test_alias_is_same_class(self):
        assert tdm.TimeSeriesTable is tdm.MultivariateTimeSeries is tdm.MultiTimeSeries

    def test_alias_construction(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[1.0, 2.0]])
        ts = tdm.MultiTimeSeries(
            hourly_frequency,
            timestamps=[base],
            values=values,
            names=["a", "b"],
        )
        assert isinstance(ts, tdm.TimeSeriesTable)
        assert isinstance(ts, tdm.MultivariateTimeSeries)
        assert ts.n_columns == 2


# ---- Backward Compatibility Tests ----------------------------------------


class TestBackwardCompatibility:
    def test_single_index_construction(self, sample_ts):
        assert len(sample_ts) == 5
        assert sample_ts.is_multi_index is False

    def test_datapoint_from_getitem(self, sample_ts):
        dp = sample_ts[0]
        assert isinstance(dp, tdm.DataPoint)
        assert dp.value == 1.0
        assert dp.timestamp == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_values_returns_list(self, sample_ts):
        assert isinstance(sample_ts.values, list)

    def test_iter_returns_datapoints(self, sample_ts):
        points = list(sample_ts)
        assert all(isinstance(p, tdm.DataPoint) for p in points)

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
        return tdm.MultivariateTimeSeries(
            hourly_frequency,
            timestamps=timestamps,
            values=values,
            names=["power", "temperature", "pressure"],
            units=["MW", "°C", "hPa"],
            descriptions=["Power output", "Ambient temp", "Atmospheric pressure"],
        )

    def test_select_column_by_index(self, mv_ts):
        ts = mv_ts.select_column(0)
        assert isinstance(ts, tdm.TimeSeries)
        assert ts.name == "power"
        assert ts.unit == "MW"
        assert ts.description == "Power output"
        assert len(ts) == 5
        assert ts[0].value == 1.0
        assert ts[3].value is None  # NaN -> None

    def test_select_column_by_name(self, mv_ts):
        ts = mv_ts.select_column("temperature")
        assert isinstance(ts, tdm.TimeSeries)
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
        assert all(isinstance(s, tdm.TimeSeries) for s in series_list)
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
        merged = tdm.TimeSeries.merge(series_list)
        assert isinstance(merged, tdm.MultivariateTimeSeries)
        assert merged.n_columns == 3
        assert merged.column_names == ("power", "temperature", "pressure")
        np.testing.assert_array_equal(merged.to_numpy(), mv_ts.to_numpy())

    def test_merge_empty_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            tdm.TimeSeries.merge([])

    def test_merge_mismatched_timestamps_raises(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts1 = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1)],
            values=[1.0, 2.0],
            name="a",
        )
        ts2 = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=2)],
            values=[3.0, 4.0],
            name="b",
        )
        with pytest.raises(ValueError, match="do not match"):
            tdm.TimeSeries.merge([ts1, ts2])

    def test_merge_preserves_metadata(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts1 = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base],
            values=[1.0],
            name="x", unit="MW", description="desc_x",
        )
        ts2 = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base],
            values=[2.0],
            name="y", unit="°C", description="desc_y",
        )
        merged = tdm.TimeSeries.merge([ts1, ts2])
        assert merged.names == ["x", "y"]
        assert merged.units == ["MW", "°C"]
        assert merged.descriptions == ["desc_x", "desc_y"]

    def test_select_column_broadcast_unit(self, hourly_frequency):
        """When a single broadcast unit is used, select_column should resolve it."""
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        mv = tdm.MultivariateTimeSeries(
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


# ---- TimeSeriesCollection Tests ------------------------------------------


class TestTimeSeriesCollection:
    @pytest.fixture
    def ts_a(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = [base + timedelta(hours=i) for i in range(5)]
        return tdm.TimeSeries(
            hourly_frequency,
            timestamps=timestamps,
            values=[1.0, 2.0, 3.0, 4.0, 5.0],
            name="power",
            unit="MW",
        )

    @pytest.fixture
    def ts_b(self, hourly_frequency):
        base = datetime(2024, 1, 5, tzinfo=timezone.utc)
        timestamps = [base + timedelta(hours=i) for i in range(3)]
        return tdm.TimeSeries(
            hourly_frequency,
            timestamps=timestamps,
            values=[10.0, 20.0, 30.0],
            name="temperature",
            unit="°C",
        )

    @pytest.fixture
    def table(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = [base + timedelta(hours=i) for i in range(4)]
        values = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ])
        return tdm.TimeSeriesTable(
            hourly_frequency,
            timestamps=timestamps,
            values=values,
            names=["wind", "solar"],
        )

    def test_construction_from_list(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection([ts_a, ts_b])
        assert len(coll) == 2
        assert "power" in coll
        assert "temperature" in coll

    def test_construction_from_list_with_table(self, ts_a, table):
        coll = tdm.TimeSeriesCollection([ts_a, table])
        assert len(coll) == 2
        assert "power" in coll

    def test_construction_from_dict(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection({"a": ts_a, "b": ts_b})
        assert len(coll) == 2
        assert "a" in coll
        assert "b" in coll

    def test_construction_empty(self):
        coll = tdm.TimeSeriesCollection()
        assert len(coll) == 0
        assert not coll

    def test_auto_naming(self, hourly_frequency):
        ts = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[datetime(2024, 1, 1, tzinfo=timezone.utc)],
            values=[1.0],
        )
        coll = tdm.TimeSeriesCollection([ts])
        assert "series_0" in coll

    def test_deduplication(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts1 = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base],
            values=[1.0],
            name="x",
        )
        ts2 = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base],
            values=[2.0],
            name="x",
        )
        coll = tdm.TimeSeriesCollection([ts1, ts2])
        assert len(coll) == 2
        assert "x" in coll
        assert "x_1" in coll

    def test_len(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection([ts_a, ts_b])
        assert len(coll) == 2

    def test_bool_true(self, ts_a):
        coll = tdm.TimeSeriesCollection([ts_a])
        assert bool(coll) is True

    def test_bool_false(self):
        coll = tdm.TimeSeriesCollection()
        assert bool(coll) is False

    def test_contains(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection([ts_a, ts_b])
        assert "power" in coll
        assert "nonexistent" not in coll

    def test_getitem_by_name(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection([ts_a, ts_b])
        result = coll["power"]
        assert result is ts_a

    def test_getitem_by_index(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection([ts_a, ts_b])
        result = coll[0]
        assert result is ts_a
        result2 = coll[1]
        assert result2 is ts_b

    def test_iter(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection([ts_a, ts_b])
        keys = list(coll)
        assert keys == ["power", "temperature"]

    def test_keys(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection([ts_a, ts_b])
        assert list(coll.keys()) == ["power", "temperature"]

    def test_values(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection([ts_a, ts_b])
        vals = list(coll.values())
        assert vals[0] is ts_a
        assert vals[1] is ts_b

    def test_items(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection([ts_a, ts_b])
        items = list(coll.items())
        assert items[0] == ("power", ts_a)
        assert items[1] == ("temperature", ts_b)

    def test_names_property(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection([ts_a, ts_b])
        assert coll.names == ["power", "temperature"]

    def test_series_count(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection([ts_a, ts_b])
        assert coll.series_count == 2

    def test_add(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection([ts_a])
        coll2 = coll.add(ts_b)
        assert len(coll) == 1  # original unchanged
        assert len(coll2) == 2
        assert "temperature" in coll2

    def test_add_with_name(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection([ts_a])
        coll2 = coll.add(ts_b, name="my_temp")
        assert "my_temp" in coll2

    def test_remove(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection([ts_a, ts_b])
        coll2 = coll.remove("power")
        assert len(coll) == 2  # original unchanged
        assert len(coll2) == 1
        assert "power" not in coll2
        assert "temperature" in coll2

    def test_coverage_bar(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection([ts_a, ts_b])
        bar = coll.coverage_bar()
        r = repr(bar)
        assert "power" in r
        assert "temperature" in r

    def test_coverage_bar_with_table(self, ts_a, table):
        coll = tdm.TimeSeriesCollection([ts_a, table])
        bar = coll.coverage_bar()
        r = repr(bar)
        assert "power" in r

    def test_coverage_bar_empty(self):
        coll = tdm.TimeSeriesCollection()
        bar = coll.coverage_bar()
        assert repr(bar) == ""

    def test_repr(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection([ts_a, ts_b])
        r = repr(coll)
        assert "TimeSeriesCollection" in r
        assert "power" in r
        assert "temperature" in r
        assert "\u250c" in r  # box-drawing top-left

    def test_repr_empty(self):
        coll = tdm.TimeSeriesCollection()
        r = repr(coll)
        assert "empty" in r

    def test_repr_html(self, ts_a, ts_b):
        coll = tdm.TimeSeriesCollection([ts_a, ts_b])
        html = coll._repr_html_()
        assert "TimeSeriesCollection" in html
        assert "power" in html
        assert "temperature" in html


class TestToFloatArray:
    def test_none_to_nan(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base + timedelta(hours=i) for i in range(3)],
            values=[1.0, None, 3.0],
        )
        arr = ts._to_float_array()
        assert arr[0] == 1.0
        assert np.isnan(arr[1])
        assert arr[2] == 3.0
        assert arr.dtype == np.float64


class TestJsonFullMetadata:
    def test_timeseries_full_round_trip(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = tdm.TimeSeries(
            hourly_frequency,
            timezone="Europe/Berlin",
            timestamps=[base + timedelta(hours=i) for i in range(3)],
            values=[1.0, None, 3.0],
            name="power",
            unit="MW",
            description="Test series",
            data_type=tdm.DataType.ACTUAL,
            timeseries_type=tdm.TimeSeriesType.OVERLAPPING,
            attributes={"source": "test"},
        )
        s = ts.to_json()
        ts2 = tdm.TimeSeries.from_json(s)  # no extra args needed
        assert ts2.frequency == hourly_frequency
        assert ts2.timezone == "Europe/Berlin"
        assert ts2.name == "power"
        assert ts2.unit == "MW"
        assert ts2.description == "Test series"
        assert ts2.data_type == tdm.DataType.ACTUAL
        assert ts2.timeseries_type == tdm.TimeSeriesType.OVERLAPPING
        assert ts2.attributes == {"source": "test"}
        assert len(ts2) == 3
        assert ts2[0].value == 1.0
        assert ts2[1].value is None
        assert ts2[2].value == 3.0

    def test_timeseries_backward_compat(self, hourly_frequency):
        """Old-format JSON (just timestamps+values) still works with explicit freq."""
        old_json = json.dumps({
            "timestamps": ["2024-01-01T00:00:00+00:00"],
            "values": [42.0],
        })
        ts = tdm.TimeSeries.from_json(old_json, hourly_frequency, name="x")
        assert ts.frequency == hourly_frequency
        assert ts.name == "x"
        assert ts[0].value == 42.0

    def test_timeseries_kwargs_override(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base],
            values=[1.0],
            name="original",
            unit="MW",
        )
        s = ts.to_json()
        # Override name and unit via kwargs
        ts2 = tdm.TimeSeries.from_json(s, name="overridden", unit="kW")
        assert ts2.name == "overridden"
        assert ts2.unit == "kW"
        assert ts2.frequency == hourly_frequency  # from JSON

    def test_timeseries_no_freq_raises(self):
        old_json = json.dumps({
            "timestamps": ["2024-01-01T00:00:00+00:00"],
            "values": [42.0],
        })
        with pytest.raises(ValueError, match="frequency must be provided"):
            tdm.TimeSeries.from_json(old_json)

    def test_table_full_round_trip(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = [base + timedelta(hours=i) for i in range(3)]
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        tbl = tdm.TimeSeriesTable(
            hourly_frequency,
            timezone="Europe/Berlin",
            timestamps=timestamps,
            values=values,
            names=["power", "temperature"],
            units=["MW", "°C"],
            descriptions=["Power output", "Ambient temp"],
            data_types=[tdm.DataType.ACTUAL, tdm.DataType.MEASUREMENT],
            attributes=[{"source": "a"}, {"source": "b"}],
        )
        s = tbl.to_json()
        tbl2 = tdm.TimeSeriesTable.from_json(s)  # no extra args needed
        assert tbl2.frequency == hourly_frequency
        assert tbl2.timezone == "Europe/Berlin"
        assert tbl2.names == ["power", "temperature"]
        assert tbl2.units == ["MW", "°C"]
        assert tbl2.descriptions == ["Power output", "Ambient temp"]
        assert tbl2.data_types == [tdm.DataType.ACTUAL, tdm.DataType.MEASUREMENT]
        assert tbl2.attributes == [{"source": "a"}, {"source": "b"}]
        np.testing.assert_array_equal(tbl2.to_numpy(), values)

    def test_table_backward_compat(self, hourly_frequency):
        old_json = json.dumps({
            "timestamps": ["2024-01-01T00:00:00+00:00"],
            "values": [[1.0, 2.0]],
            "column_names": ["a", "b"],
        })
        tbl = tdm.TimeSeriesTable.from_json(old_json, hourly_frequency)
        assert tbl.frequency == hourly_frequency
        assert tbl.column_names == ("a", "b")

    def test_table_kwargs_override(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        tbl = tdm.TimeSeriesTable(
            hourly_frequency,
            timestamps=[base],
            values=np.array([[1.0, 2.0]]),
            names=["a", "b"],
            units=["MW", "°C"],
        )
        s = tbl.to_json()
        tbl2 = tdm.TimeSeriesTable.from_json(s, units=["kW", "K"])
        assert tbl2.units == ["kW", "K"]
        assert tbl2.names == ["a", "b"]  # from JSON


class TestTimeSeriesEquality:
    def test_equal(self, sample_ts):
        copy = sample_ts.copy()
        assert copy.equals(sample_ts)

    def test_different_values(self, sample_ts):
        other = sample_ts + 1
        assert not other.equals(sample_ts)

    def test_different_metadata(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts1 = tdm.TimeSeries(hourly_frequency, timestamps=[base], values=[1.0], name="a")
        ts2 = tdm.TimeSeries(hourly_frequency, timestamps=[base], values=[1.0], name="b")
        assert not ts1.equals(ts2)

    def test_nan_equality(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts1 = tdm.TimeSeries(hourly_frequency, timestamps=[base], values=[None])
        ts2 = tdm.TimeSeries(hourly_frequency, timestamps=[base], values=[None])
        assert ts1.equals(ts2)

    def test_empty_equal(self, hourly_frequency):
        ts1 = tdm.TimeSeries(hourly_frequency)
        ts2 = tdm.TimeSeries(hourly_frequency)
        assert ts1.equals(ts2)

    def test_not_hashable(self, sample_ts):
        with pytest.raises(TypeError):
            hash(sample_ts)

    def test_different_type_returns_not_implemented(self, sample_ts):
        assert sample_ts.__eq__("not a ts") is NotImplemented

    def test_equals_non_timeseries_returns_not_implemented(self, sample_ts):
        assert sample_ts.equals("not a ts") is NotImplemented


class TestTimeSeriesTableEquality:
    @pytest.fixture
    def sample_table(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = [base + timedelta(hours=i) for i in range(3)]
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        return tdm.TimeSeriesTable(
            hourly_frequency,
            timestamps=timestamps,
            values=values,
            names=["a", "b"],
        )

    def test_equal(self, sample_table):
        copy = sample_table.copy()
        assert copy == sample_table

    def test_equals_method(self, sample_table):
        copy = sample_table.copy()
        assert copy.equals(sample_table)

    def test_different_values(self, sample_table):
        other = sample_table + 1
        assert other != sample_table

    def test_different_values_equals(self, sample_table):
        other = sample_table + 1
        assert not other.equals(sample_table)

    def test_different_metadata(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[1.0]])
        tbl1 = tdm.TimeSeriesTable(
            hourly_frequency, timestamps=[base], values=values, names=["a"]
        )
        tbl2 = tdm.TimeSeriesTable(
            hourly_frequency, timestamps=[base], values=values, names=["b"]
        )
        assert tbl1 != tbl2

    def test_nan_equality(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[np.nan]])
        tbl1 = tdm.TimeSeriesTable(
            hourly_frequency, timestamps=[base], values=values, names=["x"]
        )
        tbl2 = tdm.TimeSeriesTable(
            hourly_frequency, timestamps=[base], values=values, names=["x"]
        )
        assert tbl1 == tbl2

    def test_not_hashable(self, sample_table):
        with pytest.raises(TypeError):
            hash(sample_table)

    def test_different_type_returns_not_implemented(self, sample_table):
        assert sample_table.__eq__("not a table") is NotImplemented

    def test_equals_non_table_returns_not_implemented(self, sample_table):
        assert sample_table.equals("not a table") is NotImplemented


class TestTimeSeriesArithmeticBinary:
    @pytest.fixture
    def ts_a(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = [base + timedelta(hours=i) for i in range(4)]
        return tdm.TimeSeries(
            hourly_frequency,
            timestamps=timestamps,
            values=[1.0, 2.0, 3.0, 4.0],
            name="a",
            unit="MW",
        )

    @pytest.fixture
    def ts_b(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = [base + timedelta(hours=i) for i in range(4)]
        return tdm.TimeSeries(
            hourly_frequency,
            timestamps=timestamps,
            values=[10.0, 20.0, 30.0, 40.0],
            name="b",
            unit="MW",
        )

    def test_add(self, ts_a, ts_b):
        result = ts_a + ts_b
        assert result.values == [11.0, 22.0, 33.0, 44.0]
        assert result.name is None

    def test_sub(self, ts_a, ts_b):
        result = ts_b - ts_a
        assert result.values == [9.0, 18.0, 27.0, 36.0]

    def test_mul(self, ts_a, ts_b):
        result = ts_a * ts_b
        assert result.values == [10.0, 40.0, 90.0, 160.0]

    def test_truediv(self, ts_a, ts_b):
        result = ts_b / ts_a
        assert result.values == [10.0, 10.0, 10.0, 10.0]

    def test_result_inherits_metadata(self, ts_a, ts_b):
        result = ts_a + ts_b
        assert result.frequency == ts_a.frequency
        assert result.timezone == ts_a.timezone
        assert result.unit == ts_a.unit
        assert result.name is None

    def test_timezone_mismatch(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [base + timedelta(hours=i) for i in range(2)]
        a = tdm.TimeSeries(hourly_frequency, timestamps=ts, values=[1.0, 2.0], timezone="UTC")
        b = tdm.TimeSeries(hourly_frequency, timestamps=ts, values=[1.0, 2.0], timezone="CET")
        with pytest.raises(ValueError, match="timezone mismatch"):
            a + b

    def test_frequency_mismatch(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [base + timedelta(hours=i) for i in range(2)]
        a = tdm.TimeSeries(tdm.Frequency.PT1H, timestamps=ts, values=[1.0, 2.0])
        b = tdm.TimeSeries(tdm.Frequency.P1D, timestamps=ts, values=[1.0, 2.0])
        with pytest.raises(ValueError, match="frequency mismatch"):
            a + b

    def test_timestamp_mismatch(self, hourly_frequency):
        base_a = datetime(2024, 1, 1, tzinfo=timezone.utc)
        base_b = datetime(2024, 1, 2, tzinfo=timezone.utc)
        a = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base_a, base_a + timedelta(hours=1)],
            values=[1.0, 2.0],
        )
        b = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base_b, base_b + timedelta(hours=1)],
            values=[1.0, 2.0],
        )
        with pytest.raises(ValueError, match="timestamps do not match"):
            a + b

    def test_unit_auto_conversion(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [base + timedelta(hours=i) for i in range(2)]
        a = tdm.TimeSeries(hourly_frequency, timestamps=ts, values=[1000.0, 2000.0], unit="kW")
        b = tdm.TimeSeries(hourly_frequency, timestamps=ts, values=[1.0, 2.0], unit="MW")
        result = a + b
        assert result.values[0] == pytest.approx(2000.0)
        assert result.values[1] == pytest.approx(4000.0)
        assert result.unit == "kW"

    def test_incompatible_units(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [base + timedelta(hours=i) for i in range(2)]
        a = tdm.TimeSeries(hourly_frequency, timestamps=ts, values=[1.0, 2.0], unit="MW")
        b = tdm.TimeSeries(hourly_frequency, timestamps=ts, values=[1.0, 2.0], unit="m")
        with pytest.raises(ValueError, match="cannot convert.*incompatible"):
            a + b

    def test_one_unit_none(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [base + timedelta(hours=i) for i in range(2)]
        a = tdm.TimeSeries(hourly_frequency, timestamps=ts, values=[1.0, 2.0], unit="MW")
        b = tdm.TimeSeries(hourly_frequency, timestamps=ts, values=[10.0, 20.0])
        with pytest.raises(ValueError, match="unit mismatch"):
            a + b

    def test_both_units_none(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [base + timedelta(hours=i) for i in range(2)]
        a = tdm.TimeSeries(hourly_frequency, timestamps=ts, values=[1.0, 2.0])
        b = tdm.TimeSeries(hourly_frequency, timestamps=ts, values=[10.0, 20.0])
        result = a + b
        assert result.values == [11.0, 22.0]

    def test_nan_propagation(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [base + timedelta(hours=i) for i in range(3)]
        a = tdm.TimeSeries(hourly_frequency, timestamps=ts, values=[1.0, None, 3.0])
        b = tdm.TimeSeries(hourly_frequency, timestamps=ts, values=[10.0, 20.0, None])
        result = a + b
        assert result.values[0] == 11.0
        assert result.values[1] is None
        assert result.values[2] is None

    def test_scalar_still_works(self, ts_a):
        result = ts_a + 5
        assert result.values == [6.0, 7.0, 8.0, 9.0]

    def test_rtruediv_scalar(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [base + timedelta(hours=i) for i in range(3)]
        a = tdm.TimeSeries(hourly_frequency, timestamps=ts, values=[2.0, 5.0, 10.0])
        result = 10 / a
        assert result.values[0] == pytest.approx(5.0)
        assert result.values[1] == pytest.approx(2.0)
        assert result.values[2] == pytest.approx(1.0)

    def test_unsupported_type_returns_not_implemented(self, ts_a):
        assert ts_a.__add__("string") is NotImplemented
        assert ts_a.__mul__("string") is NotImplemented
        assert ts_a.__truediv__("string") is NotImplemented
        assert ts_a.__rtruediv__("string") is NotImplemented


class TestTimeSeriesComparison:
    @pytest.fixture
    def ts_a(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = [base + timedelta(hours=i) for i in range(4)]
        return tdm.TimeSeries(
            hourly_frequency,
            timestamps=timestamps,
            values=[1.0, 2.0, 3.0, 4.0],
            name="a",
            unit="MW",
        )

    @pytest.fixture
    def ts_b(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = [base + timedelta(hours=i) for i in range(4)]
        return tdm.TimeSeries(
            hourly_frequency,
            timestamps=timestamps,
            values=[1.0, 3.0, 2.0, 4.0],
            name="b",
            unit="MW",
        )

    def test_eq_timeseries(self, ts_a, ts_b):
        result = ts_a == ts_b
        assert isinstance(result, tdm.TimeSeries)
        assert result.values == [1.0, 0.0, 0.0, 1.0]
        assert result.name is None
        assert result.unit is None

    def test_ne_timeseries(self, ts_a, ts_b):
        result = ts_a != ts_b
        assert isinstance(result, tdm.TimeSeries)
        assert result.values == [0.0, 1.0, 1.0, 0.0]

    def test_gt_scalar(self, ts_a):
        result = ts_a > 2
        assert isinstance(result, tdm.TimeSeries)
        assert result.values == [0.0, 0.0, 1.0, 1.0]

    def test_gt_timeseries(self, ts_a, ts_b):
        result = ts_a > ts_b
        assert result.values == [0.0, 0.0, 1.0, 0.0]

    def test_ge_timeseries(self, ts_a, ts_b):
        result = ts_a >= ts_b
        assert result.values == [1.0, 0.0, 1.0, 1.0]

    def test_lt_timeseries(self, ts_a, ts_b):
        result = ts_a < ts_b
        assert result.values == [0.0, 1.0, 0.0, 0.0]

    def test_le_timeseries(self, ts_a, ts_b):
        result = ts_a <= ts_b
        assert result.values == [1.0, 1.0, 0.0, 1.0]

    def test_comparison_nan_propagation(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [base + timedelta(hours=i) for i in range(3)]
        a = tdm.TimeSeries(hourly_frequency, timestamps=ts, values=[1.0, None, 3.0])
        b = tdm.TimeSeries(hourly_frequency, timestamps=ts, values=[1.0, 2.0, None])
        result = a == b
        assert result.values[0] == 1.0
        assert result.values[1] is None
        assert result.values[2] is None

    def test_comparison_alignment_check(self, hourly_frequency):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        a = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base],
            values=[1.0],
            timezone="UTC",
        )
        b = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base],
            values=[1.0],
            timezone="CET",
        )
        with pytest.raises(ValueError, match="timezone mismatch"):
            a > b

    def test_comparison_result_metadata(self, ts_a, ts_b):
        result = ts_a > ts_b
        assert result.name is None
        assert result.unit is None
        assert result.frequency == ts_a.frequency
        assert result.timezone == ts_a.timezone

    def test_eq_scalar(self, ts_a):
        result = ts_a == 2.0
        assert isinstance(result, tdm.TimeSeries)
        assert result.values == [0.0, 1.0, 0.0, 0.0]


# =============================================================================
# TimeSeriesCollection conversion methods
# =============================================================================

class TestTimeSeriesCollectionConversions:
    """Tests for TimeSeriesCollection.to_pandas_dataframe/to_polars_dataframe/to_numpy."""

    @pytest.fixture
    def sample_collection(self, hourly_frequency):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        ts_a = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1), base + timedelta(hours=2)],
            values=[1.0, 2.0, 3.0],
            name="alpha",
        )
        ts_b = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1), base + timedelta(hours=2)],
            values=[10.0, 20.0, 30.0],
            name="beta",
        )
        return tdm.TimeSeriesCollection([ts_a, ts_b])

    def test_collection_to_pandas_dataframe(self, sample_collection):
        df = sample_collection.to_pandas_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert "alpha" in df.columns
        assert "beta" in df.columns
        assert df["alpha"].tolist() == [1.0, 2.0, 3.0]
        assert df["beta"].tolist() == [10.0, 20.0, 30.0]

    def test_collection_to_pd_df_alias(self, sample_collection):
        df = sample_collection.to_pd_df()
        assert isinstance(df, pd.DataFrame)

    def test_collection_df_property(self, sample_collection):
        df = sample_collection.df
        assert isinstance(df, pd.DataFrame)

    def test_collection_to_pandas_empty(self):
        coll = tdm.TimeSeriesCollection()
        df = coll.to_pandas_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_collection_to_pandas_outer_join(self, hourly_frequency):
        """Series with different timestamps produce NaN-filled outer join."""
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        ts_a = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base, base + timedelta(hours=1)],
            values=[1.0, 2.0],
            name="a",
        )
        ts_b = tdm.TimeSeries(
            hourly_frequency,
            timestamps=[base + timedelta(hours=1), base + timedelta(hours=2)],
            values=[10.0, 20.0],
            name="b",
        )
        coll = tdm.TimeSeriesCollection([ts_a, ts_b])
        df = coll.to_pandas_dataframe()
        assert len(df) == 3  # Union of timestamps
        assert pd.isna(df.loc[df.index[0], "b"])  # b missing at first timestamp
        assert pd.isna(df.loc[df.index[-1], "a"])  # a missing at last timestamp

    def test_collection_to_numpy(self, sample_collection):
        result = sample_collection.to_numpy()
        assert isinstance(result, dict)
        assert "alpha" in result
        assert "beta" in result
        np.testing.assert_array_equal(result["alpha"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result["beta"], [10.0, 20.0, 30.0])

    def test_collection_arr_property(self, sample_collection):
        result = sample_collection.arr
        assert isinstance(result, dict)
        assert "alpha" in result


# ---- TimeSeries.convert_unit -------------------------------------------


class TestConvertUnit:
    @pytest.fixture
    def ts_mw(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [base + timedelta(hours=i) for i in range(3)]
        return tdm.TimeSeries(tdm.Frequency.PT1H, timestamps=ts, values=[1.0, 2.0, 3.0], unit="MW", name="power")

    def test_mw_to_kw(self, ts_mw):
        result = ts_mw.convert_unit("kW")
        assert result.unit == "kW"
        assert result.values[0] == pytest.approx(1000.0)
        assert result.values[1] == pytest.approx(2000.0)
        assert result.values[2] == pytest.approx(3000.0)

    def test_same_unit_noop(self, ts_mw):
        result = ts_mw.convert_unit("MW")
        assert result.unit == "MW"
        assert result.values == ts_mw.values

    def test_none_unit_raises(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = tdm.TimeSeries(tdm.Frequency.PT1H, timestamps=[base], values=[1.0])
        with pytest.raises(ValueError, match="source unit is None"):
            ts.convert_unit("MW")

    def test_incompatible_raises(self, ts_mw):
        with pytest.raises(ValueError, match="cannot convert.*incompatible"):
            ts_mw.convert_unit("m")

    def test_none_values_preserved(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [base + timedelta(hours=i) for i in range(3)]
        s = tdm.TimeSeries(tdm.Frequency.PT1H, timestamps=ts, values=[1.0, None, 3.0], unit="MW")
        result = s.convert_unit("kW")
        assert result.values[0] == pytest.approx(1000.0)
        assert result.values[1] is None
        assert result.values[2] == pytest.approx(3000.0)

    def test_immutability(self, ts_mw):
        result = ts_mw.convert_unit("kW")
        assert ts_mw.unit == "MW"
        assert ts_mw.values == [1.0, 2.0, 3.0]
        assert result is not ts_mw


# ---- TimeSeriesTable.convert_unit --------------------------------------


class TestTimeSeriesTableConvertUnit:
    @pytest.fixture
    def table_mw(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [base + timedelta(hours=i) for i in range(3)]
        vals = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        return tdm.TimeSeriesTable(
            tdm.Frequency.PT1H,
            timestamps=ts,
            values=vals,
            names=["a", "b"],
            units=["MW", "MW"],
        )

    def test_convert_all_columns(self, table_mw):
        result = table_mw.convert_unit("kW")
        assert result.units == ["kW", "kW"]
        np.testing.assert_allclose(result.values[:, 0], [1000.0, 2000.0, 3000.0])
        np.testing.assert_allclose(result.values[:, 1], [10000.0, 20000.0, 30000.0])

    def test_convert_single_column_by_index(self, table_mw):
        result = table_mw.convert_unit("kW", column=0)
        assert result.units[0] == "kW"
        assert result.units[1] == "MW"
        np.testing.assert_allclose(result.values[:, 0], [1000.0, 2000.0, 3000.0])
        np.testing.assert_allclose(result.values[:, 1], [10.0, 20.0, 30.0])

    def test_convert_single_column_by_name(self, table_mw):
        result = table_mw.convert_unit("kW", column="b")
        assert result.units[0] == "MW"
        assert result.units[1] == "kW"
        np.testing.assert_allclose(result.values[:, 1], [10000.0, 20000.0, 30000.0])

    def test_none_unit_raises(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [base + timedelta(hours=i) for i in range(2)]
        tbl = tdm.TimeSeriesTable(
            tdm.Frequency.PT1H, timestamps=ts, values=np.array([[1.0], [2.0]]),
        )
        with pytest.raises(ValueError, match="source unit is None"):
            tbl.convert_unit("MW")


# ---- TimeSeriesTable binary arithmetic ---------------------------------


class TestTimeSeriesTableArithmeticBinary:
    @pytest.fixture
    def base_ts(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return [base + timedelta(hours=i) for i in range(3)]

    @pytest.fixture
    def table_a(self, base_ts):
        vals = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        return tdm.TimeSeriesTable(
            tdm.Frequency.PT1H, timestamps=base_ts, values=vals,
            names=["x", "y"], units=["MW", "MW"],
        )

    @pytest.fixture
    def table_b(self, base_ts):
        vals = np.array([[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]])
        return tdm.TimeSeriesTable(
            tdm.Frequency.PT1H, timestamps=base_ts, values=vals,
            names=["x", "y"], units=["MW", "MW"],
        )

    def test_table_add(self, table_a, table_b):
        result = table_a + table_b
        np.testing.assert_allclose(result.values[:, 0], [5.0, 7.0, 9.0])
        np.testing.assert_allclose(result.values[:, 1], [50.0, 70.0, 90.0])

    def test_table_sub(self, table_a, table_b):
        result = table_b - table_a
        np.testing.assert_allclose(result.values[:, 0], [3.0, 3.0, 3.0])

    def test_table_mul(self, table_a, table_b):
        result = table_a * table_b
        np.testing.assert_allclose(result.values[:, 0], [4.0, 10.0, 18.0])

    def test_table_div(self, table_a, table_b):
        result = table_b / table_a
        np.testing.assert_allclose(result.values[:, 0], [4.0, 2.5, 2.0])

    def test_unit_auto_conversion(self, base_ts):
        a = tdm.TimeSeriesTable(
            tdm.Frequency.PT1H, timestamps=base_ts,
            values=np.array([[1000.0], [2000.0], [3000.0]]),
            units=["kW"],
        )
        b = tdm.TimeSeriesTable(
            tdm.Frequency.PT1H, timestamps=base_ts,
            values=np.array([[1.0], [2.0], [3.0]]),
            units=["MW"],
        )
        result = a + b
        np.testing.assert_allclose(result.values[:, 0], [2000.0, 4000.0, 6000.0])
        assert result.units == ["kW"]

    def test_table_plus_series_broadcast(self, table_a, base_ts):
        series = tdm.TimeSeries(
            tdm.Frequency.PT1H, timestamps=base_ts,
            values=[100.0, 200.0, 300.0], unit="MW",
        )
        result = table_a + series
        np.testing.assert_allclose(result.values[:, 0], [101.0, 202.0, 303.0])
        np.testing.assert_allclose(result.values[:, 1], [110.0, 220.0, 330.0])

    def test_column_count_mismatch_raises(self, base_ts):
        a = tdm.TimeSeriesTable(
            tdm.Frequency.PT1H, timestamps=base_ts,
            values=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        )
        b = tdm.TimeSeriesTable(
            tdm.Frequency.PT1H, timestamps=base_ts,
            values=np.array([[1.0], [2.0], [3.0]]),
        )
        with pytest.raises(ValueError, match="column count mismatch"):
            a + b

    def test_timezone_mismatch_raises(self, base_ts):
        a = tdm.TimeSeriesTable(
            tdm.Frequency.PT1H, timezone="UTC", timestamps=base_ts,
            values=np.array([[1.0], [2.0], [3.0]]),
        )
        b = tdm.TimeSeriesTable(
            tdm.Frequency.PT1H, timezone="CET", timestamps=base_ts,
            values=np.array([[1.0], [2.0], [3.0]]),
        )
        with pytest.raises(ValueError, match="timezone mismatch"):
            a + b

    def test_one_unit_none_raises(self, base_ts):
        a = tdm.TimeSeriesTable(
            tdm.Frequency.PT1H, timestamps=base_ts,
            values=np.array([[1.0], [2.0], [3.0]]),
            units=["MW"],
        )
        b = tdm.TimeSeriesTable(
            tdm.Frequency.PT1H, timestamps=base_ts,
            values=np.array([[1.0], [2.0], [3.0]]),
        )
        with pytest.raises(ValueError, match="unit mismatch"):
            a + b
