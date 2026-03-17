from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from timedatamodel.timeseries_numpy import TimeSeriesNumpy, _NP_DT_DTYPE
from timedatamodel.timeseries_numpy import DataShape
from timedatamodel.enums import DataType, Frequency, TimeSeriesType
from timedatamodel.location import GeoLocation


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def timestamps():
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return [base + timedelta(hours=i) for i in range(5)]


@pytest.fixture
def np_timestamps(timestamps):
    return np.array(timestamps, dtype=_NP_DT_DTYPE)


@pytest.fixture
def np_values():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)


@pytest.fixture
def simple_ts(np_timestamps, np_values):
    return TimeSeriesNumpy.from_numpy(np_timestamps, np_values, name="wind", unit="MW")


@pytest.fixture
def pandas_df(timestamps):
    return pd.DataFrame({
        "valid_time": timestamps,
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


@pytest.fixture
def missing_df(timestamps):
    return pd.DataFrame({
        "valid_time": timestamps,
        "value": [1.0, None, 3.0, None, 5.0],
    })


# ---------------------------------------------------------------------------
# Construction — from_numpy
# ---------------------------------------------------------------------------


class TestFromNumpy:
    def test_basic(self, np_timestamps, np_values):
        ts = TimeSeriesNumpy.from_numpy(np_timestamps, np_values, name="wind", unit="MW")
        assert ts.name == "wind"
        assert ts.unit == "MW"
        assert ts.num_rows == 5

    def test_shape_simple(self, np_timestamps, np_values):
        ts = TimeSeriesNumpy.from_numpy(np_timestamps, np_values)
        assert ts.shape is DataShape.SIMPLE

    def test_stores_as_datetime64(self, np_timestamps, np_values):
        ts = TimeSeriesNumpy.from_numpy(np_timestamps, np_values)
        assert ts.data["valid_time"].dtype == _NP_DT_DTYPE

    def test_stores_as_float64(self, np_timestamps, np_values):
        ts = TimeSeriesNumpy.from_numpy(np_timestamps, np_values)
        assert ts.data["value"].dtype == np.float64

    def test_accepts_datetime_list(self, timestamps, np_values):
        ts = TimeSeriesNumpy.from_numpy(timestamps, np_values)
        assert ts.num_rows == 5

    def test_metadata_defaults(self, np_timestamps, np_values):
        ts = TimeSeriesNumpy.from_numpy(np_timestamps, np_values)
        assert ts.name is None
        assert ts.unit == "dimensionless"
        assert ts.labels == {}
        assert ts.timezone == "UTC"
        assert ts.frequency is None
        assert ts.data_type is None
        assert ts.location is None
        assert ts.timeseries_type is TimeSeriesType.FLAT

    def test_all_metadata(self, np_timestamps, np_values):
        loc = GeoLocation(latitude=59.9, longitude=10.7)
        ts = TimeSeriesNumpy.from_numpy(
            np_timestamps,
            np_values,
            name="power",
            description="Wind farm output",
            unit="MW",
            labels={"region": "north"},
            timezone="Europe/Oslo",
            frequency=Frequency.PT1H,
            data_type=DataType.FORECAST,
            location=loc,
            timeseries_type=TimeSeriesType.OVERLAPPING,
        )
        assert ts.name == "power"
        assert ts.description == "Wind farm output"
        assert ts.unit == "MW"
        assert ts.labels == {"region": "north"}
        assert ts.timezone == "Europe/Oslo"
        assert ts.frequency is Frequency.PT1H
        assert ts.data_type is DataType.FORECAST
        assert ts.location == loc
        assert ts.timeseries_type is TimeSeriesType.OVERLAPPING


# ---------------------------------------------------------------------------
# Construction — from_pandas
# ---------------------------------------------------------------------------


class TestFromPandas:
    def test_basic(self, pandas_df):
        ts = TimeSeriesNumpy.from_pandas(pandas_df, name="wind", unit="MW")
        assert ts.name == "wind"
        assert ts.num_rows == 5

    def test_shape_simple(self, pandas_df):
        ts = TimeSeriesNumpy.from_pandas(pandas_df)
        assert ts.shape is DataShape.SIMPLE

    def test_datetime_index(self, timestamps):
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=timestamps)
        df.index.name = "valid_time"
        ts = TimeSeriesNumpy.from_pandas(df)
        assert ts.num_rows == 5

    def test_versioned_shape(self, timestamps):
        df = pd.DataFrame({
            "knowledge_time": timestamps,
            "valid_time": timestamps,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        ts = TimeSeriesNumpy.from_pandas(df)
        assert ts.shape is DataShape.VERSIONED

    def test_change_time_raises(self, timestamps):
        df = pd.DataFrame({
            "valid_time": timestamps,
            "change_time": timestamps,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        with pytest.raises(ValueError, match="change_time"):
            TimeSeriesNumpy.from_pandas(df)

    def test_naive_timestamps_warned(self, timestamps):
        df = pd.DataFrame({
            "valid_time": [t.replace(tzinfo=None) for t in timestamps],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        with pytest.warns(UserWarning, match="UTC"):
            ts = TimeSeriesNumpy.from_pandas(df)
        assert ts.num_rows == 5

    def test_missing_values(self, missing_df):
        ts = TimeSeriesNumpy.from_pandas(missing_df)
        assert np.isnan(ts.data["value"][1])
        assert np.isnan(ts.data["value"][3])


# ---------------------------------------------------------------------------
# Construction — direct __init__ validation
# ---------------------------------------------------------------------------


class TestInitValidation:
    def test_non_dict_raises(self, np_timestamps, np_values):
        with pytest.raises(TypeError, match="dict"):
            TimeSeriesNumpy(np_timestamps)

    def test_missing_valid_time_raises(self, np_values):
        with pytest.raises(ValueError, match="valid_time"):
            TimeSeriesNumpy({"value": np_values})

    def test_missing_value_raises(self, np_timestamps):
        with pytest.raises(ValueError, match="value"):
            TimeSeriesNumpy({"valid_time": np_timestamps})

    def test_wrong_timestamp_dtype_raises(self, np_values):
        bad_ts = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        with pytest.raises(TypeError, match="datetime64"):
            TimeSeriesNumpy({"valid_time": bad_ts, "value": np_values})


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_data_returns_dict(self, simple_ts):
        assert isinstance(simple_ts.data, dict)
        assert "valid_time" in simple_ts.data
        assert "value" in simple_ts.data

    def test_columns(self, simple_ts):
        assert "valid_time" in simple_ts.columns
        assert "value" in simple_ts.columns

    def test_num_rows(self, simple_ts):
        assert simple_ts.num_rows == 5

    def test_len(self, simple_ts):
        assert len(simple_ts) == 5

    def test_has_missing_false(self, simple_ts):
        assert simple_ts.has_missing is False

    def test_has_missing_true(self, missing_df):
        ts = TimeSeriesNumpy.from_pandas(missing_df)
        assert ts.has_missing is True


# ---------------------------------------------------------------------------
# Conversion — to_pandas
# ---------------------------------------------------------------------------


class TestToPandas:
    def test_simple_valid_time_index(self, simple_ts):
        df = simple_ts.to_pandas()
        assert df.index.name == "valid_time"
        assert list(df.columns) == ["value"]

    def test_values_preserved(self, simple_ts):
        df = simple_ts.to_pandas()
        assert list(df["value"]) == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_utc_timezone(self, simple_ts):
        df = simple_ts.to_pandas()
        assert str(df.index.tz) == "UTC"

    def test_versioned_multiindex(self, timestamps):
        df = pd.DataFrame({
            "knowledge_time": timestamps,
            "valid_time": timestamps,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        ts = TimeSeriesNumpy.from_pandas(df)
        result = ts.to_pandas()
        assert result.index.names == ["knowledge_time", "valid_time"]

    def test_round_trip(self, simple_ts):
        ts2 = TimeSeriesNumpy.from_pandas(
            simple_ts.to_pandas(), name=simple_ts.name, unit=simple_ts.unit
        )
        assert ts2.num_rows == simple_ts.num_rows
        assert list(ts2.data["value"]) == pytest.approx(list(simple_ts.data["value"]))

    def test_nan_preserved(self, missing_df):
        ts = TimeSeriesNumpy.from_pandas(missing_df)
        df = ts.to_pandas()
        assert pd.isna(df["value"].iloc[1])


# ---------------------------------------------------------------------------
# Data access — head / tail
# ---------------------------------------------------------------------------


class TestHeadTail:
    def test_head_default(self, simple_ts):
        assert simple_ts.head().num_rows == 5

    def test_head_n(self, simple_ts):
        assert simple_ts.head(3).num_rows == 3

    def test_tail_n(self, simple_ts):
        assert simple_ts.tail(2).num_rows == 2

    def test_head_correct_values(self, simple_ts):
        assert list(simple_ts.head(2).data["value"]) == pytest.approx([1.0, 2.0])

    def test_tail_correct_values(self, simple_ts):
        assert list(simple_ts.tail(2).data["value"]) == pytest.approx([4.0, 5.0])

    def test_head_preserves_metadata(self, simple_ts):
        h = simple_ts.head(2)
        assert h.name == simple_ts.name
        assert h.unit == simple_ts.unit
        assert h.frequency == simple_ts.frequency

    def test_tail_preserves_metadata(self, simple_ts):
        t = simple_ts.tail(2)
        assert t.name == simple_ts.name
        assert t.unit == simple_ts.unit

    def test_head_timestamps_sliced(self, simple_ts):
        h = simple_ts.head(3)
        assert len(h.data["valid_time"]) == 3


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


class TestConvertUnit:
    def test_scale_factor(self, simple_ts):
        ts_kw = simple_ts.convert_unit("kW")
        assert list(ts_kw.data["value"]) == pytest.approx([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])

    def test_unit_updated(self, simple_ts):
        assert simple_ts.convert_unit("kW").unit == "kW"

    def test_original_unchanged(self, simple_ts):
        simple_ts.convert_unit("kW")
        assert simple_ts.unit == "MW"
        assert list(simple_ts.data["value"]) == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_other_metadata_preserved(self, simple_ts):
        ts_kw = simple_ts.convert_unit("kW")
        assert ts_kw.name == simple_ts.name
        assert ts_kw.frequency == simple_ts.frequency

    def test_nan_preserved_after_conversion(self, missing_df):
        ts = TimeSeriesNumpy.from_pandas(missing_df, unit="MW")
        ts_kw = ts.convert_unit("kW")
        assert np.isnan(ts_kw.data["value"][1])
        assert ts_kw.data["value"][0] == pytest.approx(1000.0)

    def test_incompatible_unit_raises(self, simple_ts):
        with pytest.raises(Exception):
            simple_ts.convert_unit("degC")


# ---------------------------------------------------------------------------
# validate_for_insert
# ---------------------------------------------------------------------------


class TestValidateForInsert:
    def test_simple_returns_data_and_shape(self, simple_ts):
        data, shape = simple_ts.validate_for_insert()
        assert isinstance(data, dict)
        assert shape is DataShape.SIMPLE

    def test_versioned_allowed(self, timestamps):
        df = pd.DataFrame({
            "knowledge_time": timestamps,
            "valid_time": timestamps,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        ts = TimeSeriesNumpy.from_pandas(df)
        _, shape = ts.validate_for_insert()
        assert shape is DataShape.VERSIONED

    def test_corrected_raises(self, np_timestamps, np_values):
        ts = TimeSeriesNumpy(
            {
                "valid_time": np_timestamps,
                "change_time": np_timestamps,
                "value": np_values,
            }
        )
        with pytest.raises(ValueError, match="cannot be inserted"):
            ts.validate_for_insert()

    def test_audit_raises(self, np_timestamps, np_values):
        ts = TimeSeriesNumpy(
            {
                "knowledge_time": np_timestamps,
                "change_time": np_timestamps,
                "valid_time": np_timestamps,
                "value": np_values,
            }
        )
        with pytest.raises(ValueError, match="cannot be inserted"):
            ts.validate_for_insert()


# ---------------------------------------------------------------------------
# metadata_dict
# ---------------------------------------------------------------------------


class TestMetadataDict:
    def test_keys_present(self, simple_ts):
        d = simple_ts.metadata_dict()
        for key in ("name", "unit", "shape", "num_rows", "timezone", "timeseries_type"):
            assert key in d

    def test_values(self, simple_ts):
        d = simple_ts.metadata_dict()
        assert d["name"] == "wind"
        assert d["unit"] == "MW"
        assert d["shape"] == "SIMPLE"
        assert d["num_rows"] == 5

    def test_location_serialised(self, np_timestamps, np_values):
        loc = GeoLocation(latitude=59.9, longitude=10.7)
        ts = TimeSeriesNumpy.from_numpy(np_timestamps, np_values, location=loc)
        d = ts.metadata_dict()
        assert d["location"]["latitude"] == pytest.approx(59.9)
        assert d["location"]["longitude"] == pytest.approx(10.7)

    def test_no_location(self, simple_ts):
        assert simple_ts.metadata_dict()["location"] is None
