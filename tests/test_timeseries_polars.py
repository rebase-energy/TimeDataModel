from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import polars as pl
import pytest

from timedatamodel.timeseries_polars import (
    DataShape,
    TimeSeriesPolars,
    _infer_shape,
    _validate_table,
)
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
def simple_df(timestamps):
    return pd.DataFrame({
        "valid_time": timestamps,
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


@pytest.fixture
def simple_ts(simple_df):
    return TimeSeriesPolars.from_pandas(simple_df, name="wind", unit="MW")


@pytest.fixture
def missing_df(timestamps):
    return pd.DataFrame({
        "valid_time": timestamps,
        "value": [1.0, None, 3.0, None, 5.0],
    })


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestFromPandas:
    def test_basic(self, simple_df):
        ts = TimeSeriesPolars.from_pandas(simple_df, name="wind", unit="MW")
        assert ts.name == "wind"
        assert ts.unit == "MW"
        assert ts.num_rows == 5

    def test_shape_simple(self, simple_df):
        ts = TimeSeriesPolars.from_pandas(simple_df)
        assert ts.shape is DataShape.SIMPLE

    def test_datetime_index(self, timestamps):
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=timestamps)
        df.index.name = "valid_time"
        ts = TimeSeriesPolars.from_pandas(df, name="solar")
        assert ts.num_rows == 5
        assert ts.shape is DataShape.SIMPLE

    def test_versioned_shape(self, timestamps):
        df = pd.DataFrame({
            "knowledge_time": timestamps,
            "valid_time": timestamps,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        ts = TimeSeriesPolars.from_pandas(df)
        assert ts.shape is DataShape.VERSIONED

    def test_change_time_raises(self, timestamps):
        df = pd.DataFrame({
            "valid_time": timestamps,
            "change_time": timestamps,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        with pytest.raises(ValueError, match="change_time"):
            TimeSeriesPolars.from_pandas(df)

    def test_metadata_defaults(self, simple_df):
        ts = TimeSeriesPolars.from_pandas(simple_df)
        assert ts.name is None
        assert ts.unit == "dimensionless"
        assert ts.labels == {}
        assert ts.timezone == "UTC"
        assert ts.frequency is None
        assert ts.data_type is None
        assert ts.location is None
        assert ts.timeseries_type is TimeSeriesType.FLAT

    def test_all_metadata(self, simple_df):
        loc = GeoLocation(latitude=59.9, longitude=10.7)
        ts = TimeSeriesPolars.from_pandas(
            simple_df,
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


class TestFromPolars:
    def test_basic(self, timestamps):
        df = pl.DataFrame({
            "valid_time": pl.Series(timestamps).cast(pl.Datetime("us", "UTC")),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        ts = TimeSeriesPolars.from_polars(df, name="wind", unit="kW")
        assert ts.name == "wind"
        assert ts.shape is DataShape.SIMPLE
        assert ts.num_rows == 5

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="polars.DataFrame"):
            TimeSeriesPolars({"not": "a dataframe"})

    def test_missing_column_raises(self):
        df = pl.DataFrame({"value": [1.0, 2.0]})
        with pytest.raises(ValueError, match="valid_time"):
            TimeSeriesPolars(df)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_df_returns_polars(self, simple_ts):
        assert isinstance(simple_ts.df, pl.DataFrame)

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
        ts = TimeSeriesPolars.from_pandas(missing_df)
        assert ts.has_missing is True


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


class TestToPandas:
    def test_simple_valid_time_index(self, simple_ts):
        df = simple_ts.to_pandas()
        assert df.index.name == "valid_time"
        assert list(df.columns) == ["value"]

    def test_values_preserved(self, simple_ts):
        df = simple_ts.to_pandas()
        assert list(df["value"]) == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_utc_timezone(self, simple_ts):
        df = simple_ts.to_pandas()
        assert str(df.index.tz) == "UTC"

    def test_versioned_multiindex(self, timestamps):
        df = pd.DataFrame({
            "knowledge_time": timestamps,
            "valid_time": timestamps,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        ts = TimeSeriesPolars.from_pandas(df)
        result = ts.to_pandas()
        assert result.index.names == ["knowledge_time", "valid_time"]

    def test_round_trip(self, simple_ts):
        ts2 = TimeSeriesPolars.from_pandas(
            simple_ts.to_pandas(), name=simple_ts.name, unit=simple_ts.unit
        )
        assert ts2.num_rows == simple_ts.num_rows


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


class TestHeadTail:
    def test_head_default(self, simple_ts):
        assert simple_ts.head().num_rows == 5

    def test_head_n(self, simple_ts):
        assert simple_ts.head(3).num_rows == 3

    def test_tail_n(self, simple_ts):
        assert simple_ts.tail(2).num_rows == 2

    def test_head_preserves_metadata(self, simple_ts):
        h = simple_ts.head(2)
        assert h.name == simple_ts.name
        assert h.unit == simple_ts.unit
        assert h.frequency == simple_ts.frequency

    def test_tail_preserves_metadata(self, simple_ts):
        t = simple_ts.tail(2)
        assert t.name == simple_ts.name
        assert t.unit == simple_ts.unit

    def test_head_correct_rows(self, simple_ts):
        h = simple_ts.head(2)
        assert list(h.df["value"]) == [1.0, 2.0]

    def test_tail_correct_rows(self, simple_ts):
        t = simple_ts.tail(2)
        assert list(t.df["value"]) == [4.0, 5.0]


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


class TestConvertUnit:
    def test_scale_factor(self, simple_ts):
        ts_kw = simple_ts.convert_unit("kW")
        assert list(ts_kw.df["value"]) == pytest.approx([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])

    def test_unit_updated(self, simple_ts):
        ts_kw = simple_ts.convert_unit("kW")
        assert ts_kw.unit == "kW"

    def test_other_metadata_preserved(self, simple_ts):
        ts_kw = simple_ts.convert_unit("kW")
        assert ts_kw.name == simple_ts.name
        assert ts_kw.frequency == simple_ts.frequency

    def test_incompatible_unit_raises(self, simple_ts):
        with pytest.raises(Exception):
            simple_ts.convert_unit("degC")


# ---------------------------------------------------------------------------
# validate_for_insert
# ---------------------------------------------------------------------------


class TestValidateForInsert:
    def test_simple_returns_df_and_shape(self, simple_ts):
        df, shape = simple_ts.validate_for_insert()
        assert isinstance(df, pl.DataFrame)
        assert shape is DataShape.SIMPLE

    def test_versioned_allowed(self, timestamps):
        df = pd.DataFrame({
            "knowledge_time": timestamps,
            "valid_time": timestamps,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        ts = TimeSeriesPolars.from_pandas(df)
        _, shape = ts.validate_for_insert()
        assert shape is DataShape.VERSIONED

    def test_corrected_raises(self, timestamps):
        polars_df = pl.DataFrame({
            "valid_time": pl.Series(timestamps).cast(pl.Datetime("us", "UTC")),
            "change_time": pl.Series(timestamps).cast(pl.Datetime("us", "UTC")),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        ts = TimeSeriesPolars.from_polars(polars_df)
        with pytest.raises(ValueError, match="cannot be inserted"):
            ts.validate_for_insert()

    def test_audit_raises(self, timestamps):
        polars_df = pl.DataFrame({
            "knowledge_time": pl.Series(timestamps).cast(pl.Datetime("us", "UTC")),
            "change_time": pl.Series(timestamps).cast(pl.Datetime("us", "UTC")),
            "valid_time": pl.Series(timestamps).cast(pl.Datetime("us", "UTC")),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        ts = TimeSeriesPolars.from_polars(polars_df)
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

    def test_location_serialised(self, simple_df):
        loc = GeoLocation(latitude=59.9, longitude=10.7)
        ts = TimeSeriesPolars.from_pandas(simple_df, location=loc)
        d = ts.metadata_dict()
        assert d["location"]["latitude"] == pytest.approx(59.9)
        assert d["location"]["longitude"] == pytest.approx(10.7)

    def test_no_location(self, simple_ts):
        assert simple_ts.metadata_dict()["location"] is None


# ---------------------------------------------------------------------------
# Conversion methods
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_ts_with_nulls(missing_df):
    return TimeSeriesPolars.from_pandas(missing_df, name="wind", unit="MW")


class TestConversionMethods:
    def test_to_polars_returns_dataframe(self, simple_ts):
        result = simple_ts.to_polars()
        assert isinstance(result, pl.DataFrame)
        assert list(result.columns) == ["valid_time", "value"]

    def test_to_python_list_structure(self, simple_ts):
        result = simple_ts.to_python_list()
        assert isinstance(result, list)
        assert len(result) == 5
        assert "valid_time" in result[0]
        assert "value" in result[0]
        assert result[0]["value"] == 1.0

    def test_to_python_list_nulls(self, simple_ts_with_nulls):
        result = simple_ts_with_nulls.to_python_list()
        values = [row["value"] for row in result]
        assert None in values

    def test_to_numpy(self, simple_ts):
        import numpy as np
        result = simple_ts.to_numpy()
        assert isinstance(result, np.ndarray)
        assert result.dtype.names is not None  # structured array
        assert "value" in result.dtype.names
        assert "valid_time" in result.dtype.names
        assert len(result) == 5

    def test_to_numpy_missing_dep(self, simple_ts, monkeypatch):
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="numpy"):
            simple_ts.to_numpy()

    def test_to_pyarrow(self, simple_ts):
        import pyarrow as pa
        result = simple_ts.to_pyarrow()
        assert isinstance(result, pa.Table)
        assert "valid_time" in result.column_names
        assert "value" in result.column_names
        assert len(result) == 5

    def test_to_pyarrow_missing_dep(self, simple_ts, monkeypatch):
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "pyarrow":
                raise ImportError
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pyarrow"):
            simple_ts.to_pyarrow()
