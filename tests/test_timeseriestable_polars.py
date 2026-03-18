from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import polars as pl
import pytest

from timedatamodel.timeseries_polars import DataShape, TimeSeriesPolars
from timedatamodel.timeseriestable_polars import TimeSeriesTablePolars
from timedatamodel.enums import DataType, Frequency
from timedatamodel.location import GeoLocation


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def timestamps():
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return [base + timedelta(hours=i) for i in range(4)]


@pytest.fixture
def pandas_df(timestamps):
    return pd.DataFrame({
        "valid_time": timestamps,
        "wind":  [1.0, 2.0, 3.0, 4.0],
        "solar": [4.0, 5.0, 6.0, 7.0],
    })


@pytest.fixture
def simple_table(pandas_df):
    return TimeSeriesTablePolars.from_pandas(
        pandas_df,
        frequency=Frequency.PT1H,
        units=["MW", "MW"],
    )


@pytest.fixture
def ts_a(timestamps):
    return TimeSeriesPolars.from_pandas(
        pd.DataFrame({"valid_time": timestamps, "value": [1.0, 2.0, 3.0, 4.0]}),
        name="wind", unit="MW", frequency=Frequency.PT1H,
    )


@pytest.fixture
def ts_b(timestamps):
    return TimeSeriesPolars.from_pandas(
        pd.DataFrame({"valid_time": timestamps, "value": [5.0, 6.0, 7.0, 8.0]}),
        name="solar", unit="kW", frequency=Frequency.PT1H,
    )


# ---------------------------------------------------------------------------
# Construction — from_pandas
# ---------------------------------------------------------------------------


class TestFromPandas:
    def test_basic(self, simple_table):
        assert simple_table.n_columns == 2
        assert simple_table.num_rows == 4

    def test_column_names(self, simple_table):
        assert simple_table.column_names == ["wind", "solar"]

    def test_units(self, simple_table):
        assert simple_table.units == ["MW", "MW"]

    def test_datetime_index(self, timestamps):
        df = pd.DataFrame(
            {"wind": [1.0, 2.0, 3.0, 4.0], "solar": [4.0, 5.0, 6.0, 7.0]},
            index=timestamps,
        )
        df.index.name = "valid_time"
        table = TimeSeriesTablePolars.from_pandas(df, frequency=Frequency.PT1H)
        assert table.num_rows == 4

    def test_broadcast_single_unit(self, pandas_df):
        table = TimeSeriesTablePolars.from_pandas(
            pandas_df, frequency=Frequency.PT1H, units=["MW"]
        )
        assert table.units == ["MW", "MW"]

    def test_per_column_metadata(self, pandas_df):
        locs = [GeoLocation(59.0, 10.0), GeoLocation(60.0, 11.0)]
        table = TimeSeriesTablePolars.from_pandas(
            pandas_df, frequency=Frequency.PT1H, locations=locs
        )
        assert table.locations[0] == locs[0]
        assert table.locations[1] == locs[1]


# ---------------------------------------------------------------------------
# Construction — from_polars
# ---------------------------------------------------------------------------


class TestFromPolars:
    def test_basic(self, timestamps):
        df = pl.DataFrame({
            "valid_time": pl.Series(timestamps).cast(pl.Datetime("us", "UTC")),
            "wind":  [1.0, 2.0, 3.0, 4.0],
            "solar": [4.0, 5.0, 6.0, 7.0],
        })
        table = TimeSeriesTablePolars.from_polars(df, frequency=Frequency.PT1H)
        assert table.n_columns == 2
        assert table.column_names == ["wind", "solar"]


# ---------------------------------------------------------------------------
# Construction — from_timeseries
# ---------------------------------------------------------------------------


class TestFromTimeseries:
    def test_basic(self, ts_a, ts_b, timestamps):
        table = TimeSeriesTablePolars.from_timeseries([ts_a, ts_b])
        assert table.column_names == ["wind", "solar"]
        assert table.num_rows == 4

    def test_derives_units(self, ts_a, ts_b):
        table = TimeSeriesTablePolars.from_timeseries([ts_a, ts_b])
        assert table.units == ["MW", "kW"]

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            TimeSeriesTablePolars.from_timeseries([])

    def test_non_simple_shape_raises(self, timestamps):
        df = pd.DataFrame({
            "knowledge_time": timestamps,
            "valid_time": timestamps,
            "value": [1.0, 2.0, 3.0, 4.0],
        })
        ts_versioned = TimeSeriesPolars.from_pandas(df, name="v")
        with pytest.raises(ValueError, match="SIMPLE"):
            TimeSeriesTablePolars.from_timeseries([ts_versioned])

    def test_mismatched_timestamps_raises(self, ts_a, timestamps):
        other_times = [timestamps[0] + timedelta(hours=i + 1) for i in range(4)]
        ts_shifted = TimeSeriesPolars.from_pandas(
            pd.DataFrame({"valid_time": other_times, "value": [1.0, 2.0, 3.0, 4.0]}),
            name="shifted",
        )
        with pytest.raises(ValueError, match="identical timestamps"):
            TimeSeriesTablePolars.from_timeseries([ts_a, ts_shifted])

    def test_derives_locations(self, timestamps):
        loc_a = GeoLocation(59.0, 10.0)
        loc_b = GeoLocation(60.0, 11.0)
        ts1 = TimeSeriesPolars.from_pandas(
            pd.DataFrame({"valid_time": timestamps, "value": [1.0, 2.0, 3.0, 4.0]}),
            name="a", location=loc_a, frequency=Frequency.PT1H,
        )
        ts2 = TimeSeriesPolars.from_pandas(
            pd.DataFrame({"valid_time": timestamps, "value": [5.0, 6.0, 7.0, 8.0]}),
            name="b", location=loc_b, frequency=Frequency.PT1H,
        )
        table = TimeSeriesTablePolars.from_timeseries([ts1, ts2])
        assert table.locations == [loc_a, loc_b]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_n_columns(self, simple_table):
        assert simple_table.n_columns == 2

    def test_num_rows(self, simple_table):
        assert simple_table.num_rows == 4

    def test_len(self, simple_table):
        assert len(simple_table) == 4

    def test_has_missing_false(self, simple_table):
        assert simple_table.has_missing is False

    def test_has_missing_true(self, timestamps):
        df = pd.DataFrame({
            "valid_time": timestamps,
            "wind":  [1.0, None, 3.0, 4.0],
            "solar": [4.0, 5.0, 6.0, 7.0],
        })
        table = TimeSeriesTablePolars.from_pandas(df, frequency=Frequency.PT1H)
        assert table.has_missing is True

    def test_df_returns_polars(self, simple_table):
        assert isinstance(simple_table.df, pl.DataFrame)


# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------


class TestSelectColumn:
    def test_by_name(self, simple_table):
        ts = simple_table.select_column("wind")
        assert isinstance(ts, TimeSeriesPolars)
        assert list(ts.df["value"]) == [1.0, 2.0, 3.0, 4.0]

    def test_by_index(self, simple_table):
        ts = simple_table.select_column(1)
        assert list(ts.df["value"]) == [4.0, 5.0, 6.0, 7.0]

    def test_carries_unit(self, simple_table):
        ts = simple_table.select_column("wind")
        assert ts.unit == "MW"

    def test_invalid_name_raises(self, simple_table):
        with pytest.raises((ValueError, KeyError)):
            simple_table.select_column("nonexistent")


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


class TestHeadTail:
    def test_head(self, simple_table):
        assert simple_table.head(2).num_rows == 2

    def test_tail(self, simple_table):
        assert simple_table.tail(1).num_rows == 1

    def test_head_preserves_columns(self, simple_table):
        assert simple_table.head(2).column_names == simple_table.column_names

    def test_tail_preserves_units(self, simple_table):
        assert simple_table.tail(2).units == simple_table.units


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


class TestToPandas:
    def test_valid_time_as_index(self, simple_table):
        df = simple_table.to_pandas()
        assert df.index.name == "valid_time"

    def test_value_columns(self, simple_table):
        df = simple_table.to_pandas()
        assert "wind" in df.columns
        assert "solar" in df.columns

    def test_utc_index(self, simple_table):
        df = simple_table.to_pandas()
        assert str(df.index.tz) == "UTC"


class TestValidateForInsert:
    def test_returns_df_and_simple(self, simple_table):
        df, shape = simple_table.validate_for_insert()
        assert isinstance(df, pl.DataFrame)
        assert shape is DataShape.SIMPLE


# ---------------------------------------------------------------------------
# Spatial filtering
# ---------------------------------------------------------------------------


@pytest.fixture
def geo_table(timestamps):
    locs = [
        GeoLocation(59.91, 10.75),   # Oslo
        GeoLocation(60.39, 5.32),    # Bergen
        GeoLocation(63.43, 10.39),   # Trondheim
    ]
    df = pd.DataFrame({
        "valid_time": timestamps,
        "oslo":       [1.0, 2.0, 3.0, 4.0],
        "bergen":     [5.0, 6.0, 7.0, 8.0],
        "trondheim":  [9.0, 10.0, 11.0, 12.0],
    })
    return TimeSeriesTablePolars.from_pandas(
        df, frequency=Frequency.PT1H, locations=locs
    )


class TestSpatialFiltering:
    def test_filter_by_location_radius(self, geo_table):
        oslo = GeoLocation(59.91, 10.75)
        result = geo_table.filter_columns_by_location(oslo, radius_km=200)
        assert "oslo" in result.column_names

    def test_filter_excludes_distant(self, geo_table):
        oslo = GeoLocation(59.91, 10.75)
        result = geo_table.filter_columns_by_location(oslo, radius_km=50)
        assert "trondheim" not in result.column_names

    def test_nearest_n(self, geo_table):
        oslo = GeoLocation(59.91, 10.75)
        result = geo_table.nearest_columns(oslo, n=1)
        assert result.n_columns == 1
        assert "oslo" in result.column_names

    def test_nearest_two(self, geo_table):
        oslo = GeoLocation(59.91, 10.75)
        result = geo_table.nearest_columns(oslo, n=2)
        assert result.n_columns == 2


# ---------------------------------------------------------------------------
# metadata_dict
# ---------------------------------------------------------------------------


class TestMetadataDict:
    def test_keys_present(self, simple_table):
        d = simple_table.metadata_dict()
        for key in ("columns", "num_rows", "frequency", "timezone"):
            assert key in d

    def test_values(self, simple_table):
        d = simple_table.metadata_dict()
        assert set(d["columns"].keys()) == {"wind", "solar"}
        assert d["num_rows"] == 4
        assert str(d["frequency"]) == "PT1H"


# ---------------------------------------------------------------------------
# Conversion methods
# ---------------------------------------------------------------------------


class TestConversionMethods:
    def test_to_polars_returns_dataframe(self, simple_table):
        result = simple_table.to_polars()
        assert isinstance(result, pl.DataFrame)
        assert "valid_time" in result.columns
        assert "wind" in result.columns
        assert "solar" in result.columns

    def test_to_list_structure(self, simple_table):
        result = simple_table.to_list()
        assert isinstance(result, dict)
        assert "valid_time" in result
        assert "wind" in result
        assert "solar" in result
        assert len(result["wind"]) == 4
        assert result["wind"][0] == 1.0

    def test_to_numpy(self, simple_table):
        import numpy as np
        result = simple_table.to_numpy()
        assert isinstance(result, dict)
        assert "valid_time" in result
        assert "wind" in result
        assert "solar" in result
        assert isinstance(result["wind"], np.ndarray)
        assert len(result["wind"]) == 4

    def test_to_numpy_missing_dep(self, simple_table, monkeypatch):
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="numpy"):
            simple_table.to_numpy()

    def test_to_pyarrow(self, simple_table):
        import pyarrow as pa
        result = simple_table.to_pyarrow()
        assert isinstance(result, pa.Table)
        assert "valid_time" in result.column_names
        assert "wind" in result.column_names
        assert "solar" in result.column_names
        assert len(result) == 4

    def test_to_pyarrow_missing_dep(self, simple_table, monkeypatch):
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "pyarrow":
                raise ImportError
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pyarrow"):
            simple_table.to_pyarrow()
