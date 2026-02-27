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


class TestBeginEnd:
    def test_begin_end_non_empty(self, sample_ts):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert sample_ts.begin == base
        assert sample_ts.end == base + timedelta(hours=4)

    def test_begin_end_empty(self, hourly_resolution):
        ts = TimeSeries(hourly_resolution)
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
        import pandas as pd
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


class TestTier1:
    def test_duration_non_empty(self, sample_ts):
        assert sample_ts.duration == timedelta(hours=4)

    def test_duration_empty(self, hourly_resolution):
        ts = TimeSeries(hourly_resolution)
        assert ts.duration is None

    def test_duration_single_point(self, hourly_resolution):
        ts = TimeSeries(
            hourly_resolution,
            timestamps=[datetime(2024, 1, 1, tzinfo=timezone.utc)],
            values=[1.0],
        )
        assert ts.duration == timedelta(0)

    def test_head_default(self, sample_ts):
        h = sample_ts.head()
        assert len(h) == 5  # series has exactly 5 points

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
        assert len(t) == 5  # series has exactly 5 points

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
        assert sample_ts[0].value == 1.0  # original unchanged

    def test_has_missing_true(self, sample_ts):
        assert sample_ts.has_missing is True

    def test_has_missing_false(self, hourly_resolution):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = TimeSeries(
            hourly_resolution,
            timestamps=[base, base + timedelta(hours=1)],
            values=[1.0, 2.0],
        )
        assert ts.has_missing is False

    def test_has_missing_empty(self, hourly_resolution):
        ts = TimeSeries(hourly_resolution)
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
        assert ts2[3].value is None  # None passthrough

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

    def test_abs(self, hourly_resolution):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = TimeSeries(
            hourly_resolution,
            timestamps=[base, base + timedelta(hours=1)],
            values=[-3.0, 4.0],
        )
        ts2 = abs(ts)
        assert ts2[0].value == 3.0
        assert ts2[1].value == 4.0

    def test_round(self, hourly_resolution):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = TimeSeries(
            hourly_resolution,
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
        assert sample_ts[0].value == 1.0  # original unchanged


class TestTier6IO:
    def test_json_round_trip(self, sample_ts):
        s = sample_ts.to_json()
        ts2 = TimeSeries.from_json(s, sample_ts.resolution)
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
        ts2 = TimeSeries.from_json(s, sample_ts.resolution)
        assert ts2[3].value is None

    def test_json_empty(self, hourly_resolution):
        ts = TimeSeries(hourly_resolution)
        s = ts.to_json()
        ts2 = TimeSeries.from_json(s, hourly_resolution)
        assert len(ts2) == 0

    def test_csv_round_trip(self, sample_ts):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            sample_ts.to_csv(path)
            ts2 = TimeSeries.from_csv(path, sample_ts.resolution)
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
            ts2 = TimeSeries.from_csv(path, sample_ts.resolution)
            assert ts2[3].value is None
        finally:
            path.unlink(missing_ok=True)

    def test_csv_column_name(self, sample_ts):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            sample_ts.to_csv(path)
            ts2 = TimeSeries.from_csv(path, sample_ts.resolution)
            assert ts2.metadata.name == "power"
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
        assert ts.resolution.frequency == Frequency.PT1H
        assert ts.metadata.name == "power"
        assert len(ts) == 5

    def test_infer_daily(self):
        index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        df = pd.DataFrame({"temp": range(5)}, index=index)
        ts = TimeSeries.from_pandas(df)
        assert ts.resolution.frequency == Frequency.P1D

    def test_infer_from_few_points_falls_back(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        index = pd.DatetimeIndex([base, base + timedelta(hours=1)])
        df = pd.DataFrame({"v": [1.0, 2.0]}, index=index)
        ts = TimeSeries.from_pandas(df)
        # Only 2 points, no explicit freq → falls back to NONE
        assert ts.resolution.frequency == Frequency.NONE

    def test_infer_timezone(self):
        index = pd.date_range("2024-01-01", periods=5, freq="h", tz="Europe/Berlin")
        df = pd.DataFrame({"v": range(5)}, index=index)
        ts = TimeSeries.from_pandas(df)
        assert ts.resolution.timezone == "Europe/Berlin"

    def test_explicit_resolution_still_works(self):
        index = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=index)
        res = Resolution(Frequency.P1D, "US/Eastern")
        ts = TimeSeries.from_pandas(df, resolution=res)
        assert ts.resolution == res

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
        assert result.resolution.frequency == Frequency.P1D
        assert result.metadata.name == "energy"
        # Original unchanged
        assert len(sample_ts) == 5
        assert sample_ts[0].value == 1.0

    def test_inplace_returns_none(self, sample_ts):
        new_index = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({"energy": [10.0, 20.0, 30.0]}, index=new_index)
        result = sample_ts.update_from_pandas(df, inplace=True)
        assert result is None
        assert len(sample_ts) == 3
        assert sample_ts[0].value == 10.0
        assert sample_ts.resolution.frequency == Frequency.P1D
        assert sample_ts.metadata.name == "energy"

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
        assert ts2[3].value is not None          # was None
        assert ts2[3].value == sample_ts[2].value  # forward-filled from index 2

    def test_clip(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df.clip(lower=2))
        assert ts2[0].value == 2.0   # was 1.0, clipped up

    def test_arithmetic(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df * 10)
        assert ts2[0].value == 10.0

    def test_metadata_preserved(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df.ffill())
        assert ts2.metadata == sample_ts.metadata

    def test_resolution_unchanged_for_noop(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df.ffill())
        assert ts2.resolution == sample_ts.resolution

    def test_resolution_updated_after_resample(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df.resample("2h").mean())
        assert ts2.resolution.frequency == Frequency.PT1H  # 2h not in map, falls back
        # Note: we test that frequency falls back to original PT1H

    def test_timezone_updated_after_tz_convert(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df.tz_convert("Europe/Berlin"))
        assert ts2.resolution.timezone == "Europe/Berlin"

    def test_none_roundtrip(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df)  # identity
        assert ts2[3].value is None

    def test_immutability(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df + 99)
        assert sample_ts[0].value == 1.0   # original unchanged

    def test_column_rename_updates_metadata_name(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df.rename(columns={"power": "power_filled"}))
        assert ts2.metadata.name == "power_filled"
        assert ts2.metadata.unit == sample_ts.metadata.unit  # other fields preserved

    def test_no_rename_preserves_metadata_name(self, sample_ts):
        ts2 = sample_ts.apply_pandas(lambda df: df.ffill())
        assert ts2.metadata.name == "power"

    def test_unit_preserved_after_arithmetic(self, sample_ts):
        # Cannot auto-detect unit change from pandas — unit stays as original
        ts2 = sample_ts.apply_pandas(lambda df: df * 0.001)
        assert ts2.metadata.unit == sample_ts.metadata.unit  # still "MW"


class TestApplyNumpy:
    def test_clip(self, sample_ts):
        ts2 = sample_ts.apply_numpy(lambda arr: np.clip(arr, 2, None))
        assert ts2[0].value == 2.0  # was 1.0, clipped up

    def test_arithmetic(self, sample_ts):
        ts2 = sample_ts.apply_numpy(lambda arr: arr * 10)
        assert ts2[0].value == 10.0

    def test_sqrt(self, sample_ts):
        ts2 = sample_ts.apply_numpy(np.sqrt)
        assert ts2[0].value == pytest.approx(1.0)
        assert ts2[1].value == pytest.approx(np.sqrt(2.0))

    def test_nan_passthrough(self, sample_ts):
        ts2 = sample_ts.apply_numpy(lambda arr: arr)  # identity
        assert ts2[3].value is None  # NaN stays None

    def test_nan_fill(self, sample_ts):
        ts2 = sample_ts.apply_numpy(lambda arr: np.nan_to_num(arr, nan=0.0))
        assert ts2[3].value == 0.0  # None filled with 0

    def test_timestamps_unchanged(self, sample_ts):
        ts2 = sample_ts.apply_numpy(lambda arr: arr + 1)
        assert ts2.timestamps == sample_ts.timestamps

    def test_resolution_unchanged(self, sample_ts):
        ts2 = sample_ts.apply_numpy(lambda arr: arr + 1)
        assert ts2.resolution == sample_ts.resolution

    def test_metadata_preserved(self, sample_ts):
        ts2 = sample_ts.apply_numpy(lambda arr: arr + 1)
        assert ts2.metadata == sample_ts.metadata

    def test_immutability(self, sample_ts):
        ts2 = sample_ts.apply_numpy(lambda arr: arr + 99)
        assert sample_ts[0].value == 1.0  # original unchanged

    def test_wrong_length_raises(self, sample_ts):
        with pytest.raises(ValueError, match="result length"):
            sample_ts.apply_numpy(lambda arr: arr[:-1])  # shorter array

    def test_nancumsum(self, sample_ts):
        ts2 = sample_ts.apply_numpy(np.nancumsum)
        assert ts2[0].value == pytest.approx(1.0)
        assert ts2[1].value == pytest.approx(3.0)   # 1 + 2
        assert ts2[2].value == pytest.approx(6.0)   # 1 + 2 + 3
        assert ts2[3].value == pytest.approx(6.0)   # NaN treated as 0
        assert ts2[4].value == pytest.approx(11.0)  # 6 + 5


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


# ---- Multi-Index Tests ------------------------------------------------


class TestMultiIndex:
    @pytest.fixture
    def base_times(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return [base + timedelta(hours=i) for i in range(5)]

    @pytest.fixture
    def multi_ts(self, hourly_resolution, base_times):
        knowledge_time = datetime(2024, 1, 2, tzinfo=timezone.utc)
        timestamps = [(vt, knowledge_time) for vt in base_times]
        values = [1.0, 2.0, 3.0, None, 5.0]
        return TimeSeries(
            hourly_resolution,
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

    def test_is_multivariate_false(self, multi_ts):
        assert multi_ts.is_multivariate is False

    def test_index_names(self, multi_ts):
        assert multi_ts.index_names == ("valid_time", "knowledge_time")

    def test_index_names_default(self, hourly_resolution):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        kt = datetime(2024, 1, 2, tzinfo=timezone.utc)
        ts = TimeSeries(
            hourly_resolution,
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

    def test_getitem_returns_tuple(self, multi_ts):
        item = multi_ts[0]
        assert isinstance(item, tuple)
        assert not isinstance(item, DataPoint)

    def test_iter_returns_tuples(self, multi_ts):
        items = list(multi_ts)
        assert all(isinstance(item, tuple) for item in items)
        assert not any(isinstance(item, DataPoint) for item in items)

    def test_validate(self, multi_ts):
        warnings = multi_ts.validate()
        assert warnings == []

    def test_to_pandas_multiindex(self, multi_ts):
        df = multi_ts.to_pandas_dataframe()
        assert isinstance(df.index, pd.MultiIndex)
        assert list(df.index.names) == ["valid_time", "knowledge_time"]
        assert len(df) == 5

    def test_from_pandas_multiindex_round_trip(self, multi_ts, hourly_resolution):
        df = multi_ts.to_pandas_dataframe()
        ts2 = TimeSeries.from_pandas(df, hourly_resolution)
        assert ts2.is_multi_index
        assert len(ts2) == len(multi_ts)
        assert ts2.values[0] == multi_ts.values[0]

    def test_json_round_trip(self, multi_ts, hourly_resolution):
        s = multi_ts.to_json()
        ts2 = TimeSeries.from_json(s, hourly_resolution)
        assert ts2.is_multi_index
        assert len(ts2) == len(multi_ts)
        assert ts2.timestamps[0] == multi_ts.timestamps[0]
        assert ts2.values[3] is None

    def test_csv_round_trip(self, multi_ts, hourly_resolution):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            multi_ts.to_csv(path)
            ts2 = TimeSeries.from_csv(path, hourly_resolution)
            assert ts2.is_multi_index
            assert len(ts2) == len(multi_ts)
            assert ts2.values[3] is None
        finally:
            path.unlink(missing_ok=True)

    def test_repr(self, multi_ts):
        r = repr(multi_ts)
        assert "TimeSeries" in r
        assert "5 points" in r


# ---- Multivariate Tests -----------------------------------------------


class TestMultivariate:
    @pytest.fixture
    def mv_ts(self, hourly_resolution):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = [base + timedelta(hours=i) for i in range(5)]
        values = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [np.nan, 40.0],
            [5.0, 50.0],
        ])
        return TimeSeries(
            hourly_resolution,
            timestamps=timestamps,
            values=values,
            column_names=["power", "temperature"],
        )

    def test_construction(self, mv_ts):
        assert len(mv_ts) == 5

    def test_is_multivariate_true(self, mv_ts):
        assert mv_ts.is_multivariate is True

    def test_is_multivariate_false(self, sample_ts):
        assert sample_ts.is_multivariate is False

    def test_is_multi_index_false(self, mv_ts):
        assert mv_ts.is_multi_index is False

    def test_ndim(self, mv_ts):
        assert mv_ts.ndim == 2

    def test_ndim_univariate(self, sample_ts):
        assert sample_ts.ndim == 1

    def test_column_names(self, mv_ts):
        assert mv_ts.column_names == ("power", "temperature")

    def test_column_names_default(self, hourly_resolution):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        ts = TimeSeries(
            hourly_resolution,
            timestamps=[base, base + timedelta(hours=1)],
            values=values,
        )
        assert ts.column_names == ("value_0", "value_1")

    def test_has_missing_true(self, mv_ts):
        assert mv_ts.has_missing is True

    def test_has_missing_false(self, hourly_resolution):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        ts = TimeSeries(
            hourly_resolution,
            timestamps=[base, base + timedelta(hours=1)],
            values=values,
        )
        assert ts.has_missing is False

    def test_add_scalar(self, mv_ts):
        ts2 = mv_ts + 10
        arr = ts2.to_numpy()
        assert arr[0, 0] == 11.0
        assert arr[0, 1] == 20.0
        assert np.isnan(arr[3, 0])  # NaN propagated

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

    def test_from_pandas_multicolumn_round_trip(self, mv_ts, hourly_resolution):
        df = mv_ts.to_pandas_dataframe()
        ts2 = TimeSeries.from_pandas(df, hourly_resolution)
        assert ts2.is_multivariate
        assert ts2.ndim == 2
        np.testing.assert_array_equal(ts2.to_numpy(), mv_ts.to_numpy())

    def test_json_round_trip(self, mv_ts, hourly_resolution):
        s = mv_ts.to_json()
        ts2 = TimeSeries.from_json(s, hourly_resolution)
        assert ts2.is_multivariate
        assert ts2.ndim == 2
        np.testing.assert_array_equal(ts2.to_numpy(), mv_ts.to_numpy())

    def test_head(self, mv_ts):
        h = mv_ts.head(2)
        assert len(h) == 2
        assert h.is_multivariate
        assert h.ndim == 2

    def test_tail(self, mv_ts):
        t = mv_ts.tail(2)
        assert len(t) == 2
        assert t.is_multivariate

    def test_copy(self, mv_ts):
        c = mv_ts.copy()
        assert len(c) == len(mv_ts)
        assert c.is_multivariate
        c._values[0, 0] = 999.0
        assert mv_ts.values[0, 0] == 1.0  # original unchanged

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
        assert "TimeSeries" in r
        assert "5 points" in r

    def test_validate(self, mv_ts):
        warnings = mv_ts.validate()
        assert warnings == []

    def test_csv_round_trip(self, mv_ts, hourly_resolution):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            mv_ts.to_csv(path)
            ts2 = TimeSeries.from_csv(path, hourly_resolution)
            assert ts2.is_multivariate
            assert ts2.ndim == 2
            # NaN values should round-trip
            assert np.isnan(ts2.to_numpy()[3, 0])
            assert ts2.to_numpy()[0, 0] == 1.0
        finally:
            path.unlink(missing_ok=True)

    def test_round(self, hourly_resolution):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = np.array([[1.567, 2.345], [3.891, 4.123]])
        ts = TimeSeries(
            hourly_resolution,
            timestamps=[base, base + timedelta(hours=1)],
            values=values,
        )
        ts2 = round(ts, 1)
        arr = ts2.to_numpy()
        assert arr[0, 0] == pytest.approx(1.6)
        assert arr[0, 1] == pytest.approx(2.3)


# ---- Backward Compatibility Tests --------------------------------------


class TestBackwardCompatibility:
    def test_single_index_construction(self, sample_ts):
        """Existing single-index construction still works identically."""
        assert len(sample_ts) == 5
        assert sample_ts.is_multi_index is False
        assert sample_ts.is_multivariate is False
        assert sample_ts.ndim == 1

    def test_datapoint_from_getitem(self, sample_ts):
        """DataPoint still returned from __getitem__ for single-index univariate."""
        dp = sample_ts[0]
        assert isinstance(dp, DataPoint)
        assert dp.value == 1.0
        assert dp.timestamp == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_values_returns_list(self, sample_ts):
        """values property still returns list for univariate."""
        assert isinstance(sample_ts.values, list)

    def test_iter_returns_datapoints(self, sample_ts):
        """__iter__ returns DataPoint for single-index univariate."""
        points = list(sample_ts)
        assert all(isinstance(p, DataPoint) for p in points)

    def test_index_names_default(self, sample_ts):
        """Default index name is 'timestamp'."""
        assert sample_ts.index_names == ("timestamp",)
