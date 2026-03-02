from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import timedatamodel as tdm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def timestamps_5h():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [base + timedelta(hours=i) for i in range(5)]


@pytest.fixture
def cube_2d(timestamps_5h):
    """2D array: scenario(3) x valid_time(5)."""
    scenarios = ["low", "mid", "high"]
    data = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [10.0, 20.0, 30.0, 40.0, 50.0],
        [100.0, 200.0, 300.0, 400.0, 500.0],
    ])
    dims = [
        tdm.Dimension("scenario", scenarios),
        tdm.Dimension("valid_time", timestamps_5h),
    ]
    return tdm.TimeSeriesArray(
        tdm.Frequency.PT1H,
        dimensions=dims,
        values=data,
        name="power",
        unit="MW",
    )


@pytest.fixture
def cube_3d(timestamps_5h):
    """3D array: scenario(2) x valid_time(5) x quantile(3)."""
    scenarios = ["low", "high"]
    quantiles = [0.1, 0.5, 0.9]
    data = np.arange(2 * 5 * 3, dtype=np.float64).reshape(2, 5, 3)
    dims = [
        tdm.Dimension("scenario", scenarios),
        tdm.Dimension("valid_time", timestamps_5h),
        tdm.Dimension("quantile", quantiles),
    ]
    return tdm.TimeSeriesArray(
        tdm.Frequency.PT1H,
        dimensions=dims,
        values=data,
        name="wind",
    )


# ---------------------------------------------------------------------------
# TestConstruction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_2d_array(self, timestamps_5h):
        data = np.ones((3, 5))
        dims = [tdm.Dimension("scenario", ["a", "b", "c"]), tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        assert cube.shape == (3, 5)

    def test_3d_array(self, timestamps_5h):
        data = np.ones((2, 5, 4))
        dims = [
            tdm.Dimension("scenario", ["a", "b"]),
            tdm.Dimension("valid_time", timestamps_5h),
            tdm.Dimension("quantile", [0.1, 0.25, 0.5, 0.9]),
        ]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        assert cube.shape == (2, 5, 4)
        assert cube.ndim == 3

    def test_4d_array(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [base + timedelta(hours=i) for i in range(3)]
        data = np.ones((2, 3, 4, 2))
        dims = [
            tdm.Dimension("scenario", ["a", "b"]),
            tdm.Dimension("valid_time", ts),
            tdm.Dimension("quantile", [0.1, 0.25, 0.5, 0.9]),
            tdm.Dimension("model", ["m1", "m2"]),
        ]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        assert cube.ndim == 4

    def test_plain_ndarray_wraps_nan(self, timestamps_5h):
        data = np.array([[1.0, np.nan, 3.0, 4.0, 5.0]])
        dims = [tdm.Dimension("scenario", ["a"]), tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        assert isinstance(cube._values, np.ma.MaskedArray)
        assert cube._values.mask[0, 1]

    def test_masked_array_preserved(self, timestamps_5h):
        data = np.ma.MaskedArray(
            np.ones((2, 5)),
            mask=[[False]*5, [True]+[False]*4],
        )
        dims = [tdm.Dimension("scenario", ["a", "b"]), tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        assert cube._values.mask[1, 0]

    def test_shape_mismatch_raises(self, timestamps_5h):
        data = np.ones((3, 4))  # wrong: 4 != 5
        dims = [tdm.Dimension("scenario", ["a", "b", "c"]), tdm.Dimension("valid_time", timestamps_5h)]
        with pytest.raises(ValueError, match="does not match"):
            tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)


# ---------------------------------------------------------------------------
# TestProperties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_dim_names(self, cube_2d):
        assert cube_2d.dim_names == ("scenario", "valid_time")

    def test_coords(self, cube_2d, timestamps_5h):
        c = cube_2d.coords
        assert c["scenario"] == ["low", "mid", "high"]
        assert c["valid_time"] == timestamps_5h

    def test_primary_time_dim_valid_time(self, cube_2d):
        assert cube_2d.primary_time_dim.name == "valid_time"

    def test_primary_time_dim_datetime_fallback(self, timestamps_5h):
        dims = [tdm.Dimension("time_axis", timestamps_5h), tdm.Dimension("x", ["a", "b"])]
        data = np.ones((5, 2))
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        assert cube.primary_time_dim.name == "time_axis"

    def test_primary_time_dim_first_fallback(self):
        dims = [tdm.Dimension("x", ["a", "b"]), tdm.Dimension("y", ["c", "d"])]
        data = np.ones((2, 2))
        cube = tdm.TimeSeriesArray(tdm.Frequency.NONE, dimensions=dims, values=data)
        assert cube.primary_time_dim.name == "x"

    def test_begin_end(self, cube_2d, timestamps_5h):
        assert cube_2d.begin == timestamps_5h[0]
        assert cube_2d.end == timestamps_5h[-1]

    def test_has_missing_false(self, cube_2d):
        assert not cube_2d.has_missing

    def test_has_missing_true(self, timestamps_5h):
        data = np.array([[1.0, np.nan, 3.0, 4.0, 5.0]])
        dims = [tdm.Dimension("scenario", ["a"]), tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        assert cube.has_missing


# ---------------------------------------------------------------------------
# TestSel
# ---------------------------------------------------------------------------

class TestSel:
    def test_single_value_drops_axis(self, cube_2d, timestamps_5h):
        result = cube_2d.sel(scenario="low")
        assert isinstance(result, tdm.TimeSeries)
        assert len(result) == 5
        assert result.values == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_slice_keeps_axis(self, cube_2d, timestamps_5h):
        t0, t1 = timestamps_5h[0], timestamps_5h[2]
        result = cube_2d.sel(valid_time=slice(t0, t1))
        assert isinstance(result, tdm.TimeSeriesArray)
        assert result.shape == (3, 3)

    def test_multi_kwarg_collapse_to_timeseries(self, cube_3d, timestamps_5h):
        result = cube_3d.sel(scenario="low", quantile=0.5)
        assert isinstance(result, tdm.TimeSeries)
        assert len(result) == 5

    def test_label_not_found_raises(self, cube_2d):
        with pytest.raises(KeyError):
            cube_2d.sel(scenario="nonexistent")

    def test_sel_to_table(self, cube_3d):
        result = cube_3d.sel(scenario="low")
        assert isinstance(result, tdm.TimeSeriesTable)
        assert result.n_columns == 3

    def test_sel_3d_keeps_array(self, cube_3d, timestamps_5h):
        t0, t1 = timestamps_5h[0], timestamps_5h[2]
        result = cube_3d.sel(valid_time=slice(t0, t1))
        assert isinstance(result, tdm.TimeSeriesArray)
        assert result.shape == (2, 3, 3)


# ---------------------------------------------------------------------------
# TestIsel
# ---------------------------------------------------------------------------

class TestIsel:
    def test_single_int_drops_axis(self, cube_2d):
        result = cube_2d.isel(scenario=0)
        assert isinstance(result, tdm.TimeSeries)
        assert result.values == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_slice_keeps_axis(self, cube_2d):
        result = cube_2d.isel(valid_time=slice(0, 3))
        assert isinstance(result, tdm.TimeSeriesArray)
        assert result.shape == (3, 3)

    def test_isel_3d(self, cube_3d):
        result = cube_3d.isel(scenario=0, quantile=1)
        assert isinstance(result, tdm.TimeSeries)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# TestConversion
# ---------------------------------------------------------------------------

class TestConversion:
    def test_to_timeseries(self, cube_2d):
        ts = cube_2d.to_timeseries(scenario="mid")
        assert isinstance(ts, tdm.TimeSeries)
        assert ts.values == [10.0, 20.0, 30.0, 40.0, 50.0]

    def test_to_timeseries_wrong_ndim(self, cube_2d):
        with pytest.raises(ValueError, match="TimeSeries"):
            cube_2d.to_timeseries()

    def test_to_table(self, cube_2d):
        tbl = cube_2d.to_table()
        assert isinstance(tbl, tdm.TimeSeriesTable)

    def test_to_table_wrong_ndim(self, cube_3d):
        with pytest.raises(ValueError, match="TimeSeriesTable"):
            cube_3d.to_table()

    def test_to_numpy(self, cube_2d):
        arr = cube_2d.to_numpy()
        assert isinstance(arr, np.ma.MaskedArray)
        assert arr.shape == (3, 5)

    def test_to_pandas_dataframe(self, cube_2d):
        df = cube_2d.to_pandas_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.MultiIndex)
        assert len(df) == 15  # 3 scenarios * 5 timestamps
        assert df.columns.tolist() == ["power"]

    def test_to_pandas_multiindex_length(self, cube_3d):
        df = cube_3d.to_pandas_dataframe()
        assert len(df) == 2 * 5 * 3


# ---------------------------------------------------------------------------
# TestFromTimeseriesList
# ---------------------------------------------------------------------------

class TestFromTimeseriesList:
    def test_aligned_timestamps(self, timestamps_5h):
        s1 = tdm.TimeSeries(tdm.Frequency.PT1H, timestamps=timestamps_5h, values=[1.0]*5, name="a")
        s2 = tdm.TimeSeries(tdm.Frequency.PT1H, timestamps=timestamps_5h, values=[2.0]*5, name="b")
        dim = tdm.Dimension("scenario", ["s1", "s2"])
        cube = tdm.TimeSeriesArray.from_timeseries_list([s1, s2], dim)
        assert cube.shape == (2, 5)
        assert cube.dim_names == ("scenario", "valid_time")

    def test_ragged_timestamps(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts1 = [base, base + timedelta(hours=1), base + timedelta(hours=2)]
        ts2 = [base + timedelta(hours=1), base + timedelta(hours=2), base + timedelta(hours=3)]
        s1 = tdm.TimeSeries(tdm.Frequency.PT1H, timestamps=ts1, values=[1.0, 2.0, 3.0])
        s2 = tdm.TimeSeries(tdm.Frequency.PT1H, timestamps=ts2, values=[10.0, 20.0, 30.0])
        dim = tdm.Dimension("scenario", ["a", "b"])
        cube = tdm.TimeSeriesArray.from_timeseries_list([s1, s2], dim)
        # Union has 4 timestamps
        assert cube.shape == (2, 4)
        assert cube.has_missing  # gaps at edges

    def test_label_count_mismatch(self, timestamps_5h):
        s1 = tdm.TimeSeries(tdm.Frequency.PT1H, timestamps=timestamps_5h, values=[1.0]*5)
        dim = tdm.Dimension("scenario", ["a", "b"])  # 2 labels but 1 series
        with pytest.raises(ValueError, match="labels"):
            tdm.TimeSeriesArray.from_timeseries_list([s1], dim)

    def test_empty_list_raises(self):
        dim = tdm.Dimension("scenario", [])
        with pytest.raises(ValueError, match="empty"):
            tdm.TimeSeriesArray.from_timeseries_list([], dim)


# ---------------------------------------------------------------------------
# TestMaskedRoundTrip
# ---------------------------------------------------------------------------

class TestMaskedRoundTrip:
    def test_masked_survives_to_pandas(self, timestamps_5h):
        data = np.array([[1.0, np.nan, 3.0, 4.0, 5.0]])
        dims = [tdm.Dimension("scenario", ["a"]), tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        df = cube.to_pandas_dataframe()
        assert df["value"].isna().sum() == 1


# ---------------------------------------------------------------------------
# TestCoverageBar
# ---------------------------------------------------------------------------

class TestCoverageBar:
    def test_returns_coverage_bar(self, cube_2d):
        cb = cube_2d.coverage_bar()
        assert isinstance(cb, tdm.CoverageBar)

    def test_has_missing_reflected(self, timestamps_5h):
        data = np.array([[1.0, np.nan, 3.0, 4.0, 5.0]])
        dims = [tdm.Dimension("scenario", ["a"]), tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        assert cube.has_missing
        cb = cube.coverage_bar()
        assert isinstance(cb, tdm.CoverageBar)

    def test_1d_array_coverage(self, timestamps_5h):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dims = [tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        cb = cube.coverage_bar()
        assert isinstance(cb, tdm.CoverageBar)


# ---------------------------------------------------------------------------
# TestRepr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_class_and_dims(self, cube_2d):
        r = repr(cube_2d)
        assert "TimeSeriesArray" in r
        assert "scenario" in r
        assert "valid_time" in r

    def test_repr_html_contains_class(self, cube_2d):
        html = cube_2d._repr_html_()
        assert "TimeSeriesArray" in html

    def test_repr_html_1d(self, timestamps_5h):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dims = [tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        html = cube._repr_html_()
        assert "TimeSeriesArray" in html


# ---------------------------------------------------------------------------
# TestReprHtmlMultiIndex
# ---------------------------------------------------------------------------

class TestReprHtmlMultiIndex:
    def test_4d_array_multi_index(self):
        """4D array: datetime dims as row headers, non-datetime as column headers."""
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        kt = [base + timedelta(hours=h) for h in range(3)]
        vt = [base + timedelta(hours=h) for h in range(10)]
        farms = ["A", "B", "C"]
        quantiles = [0.1, 0.5, 0.9]
        data = np.arange(3 * 10 * 3 * 3, dtype=np.float64).reshape(3, 10, 3, 3)
        dims = [
            tdm.Dimension("knowledge_time", kt),
            tdm.Dimension("valid_time", vt),
            tdm.Dimension("wind_farm", farms),
            tdm.Dimension("quantile", quantiles),
        ]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        html = cube._repr_html_()

        # Both datetime dims in row headers
        assert "knowledge_time" in html
        assert "valid_time" in html

        # Non-datetime dim labels in column headers
        for farm in farms:
            assert farm in html
        for q in quantiles:
            assert str(q) in html

        # colspan present for wind_farm grouping
        assert "colspan" in html

        # Row truncation present (3*10=30 combos > 7)
        assert "&hellip;" in html

        # Class name
        assert "TimeSeriesArray" in html

        # ts-idx class used for row index cells
        assert 'class="ts-idx"' in html

    def test_all_datetime_dims_edge_case(self):
        """All datetime dims: last moves to columns."""
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        kt = [base, base + timedelta(hours=12), base + timedelta(hours=24)]
        vt = [base + timedelta(hours=i) for i in range(5)]
        data = np.arange(15, dtype=np.float64).reshape(3, 5)
        dims = [
            tdm.Dimension("knowledge_time", kt),
            tdm.Dimension("valid_time", vt),
        ]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        html = cube._repr_html_()

        # knowledge_time should be row dim name in header
        assert "knowledge_time" in html
        # valid_time labels should appear as column headers (formatted dates)
        assert "2024-01-01 01:00" in html  # one of the valid_time labels
        assert "TimeSeriesArray" in html

    def test_no_datetime_dims_edge_case(self):
        """No datetime dims: first col dim moves to rows."""
        data = np.arange(9, dtype=np.float64).reshape(3, 3)
        dims = [
            tdm.Dimension("scenario", ["A", "B", "C"]),
            tdm.Dimension("quantile", [0.1, 0.5, 0.9]),
        ]
        cube = tdm.TimeSeriesArray(tdm.Frequency.NONE, dimensions=dims, values=data)
        html = cube._repr_html_()

        # scenario should be in row header
        assert "scenario" in html
        # quantile labels should appear in column headers
        assert "0.1" in html
        assert "0.9" in html
        # scenario labels appear as row index values
        assert "A" in html
        assert "C" in html
        assert "TimeSeriesArray" in html

    def test_2d_array_natural(self):
        """2D array: 1 row dim (datetime), 1 col dim (non-datetime)."""
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        vt = [base + timedelta(hours=i) for i in range(5)]
        data = np.arange(15, dtype=np.float64).reshape(3, 5)
        dims = [
            tdm.Dimension("scenario", ["low", "mid", "high"]),
            tdm.Dimension("valid_time", vt),
        ]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        html = cube._repr_html_()

        # valid_time is row dim
        assert "valid_time" in html
        # scenario labels in column headers
        assert "low" in html
        assert "high" in html
        assert "TimeSeriesArray" in html

    def test_column_truncation(self):
        """Column truncation when leaf columns > 9."""
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        vt = [base + timedelta(hours=i) for i in range(3)]
        farms = ["F1", "F2", "F3", "F4"]
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        # 4 * 5 = 20 leaf columns > 9 → truncation
        data = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5)
        dims = [
            tdm.Dimension("valid_time", vt),
            tdm.Dimension("wind_farm", farms),
            tdm.Dimension("quantile", quantiles),
        ]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        html = cube._repr_html_()

        # Column truncation ellipsis present
        assert "&hellip;" in html
        # Head farms visible
        assert "F1" in html
        # Tail farms visible
        assert "F4" in html
        # colspan for farm grouping
        assert "colspan" in html


# ---------------------------------------------------------------------------
# TestNDTimeSeriesAlias
# ---------------------------------------------------------------------------

class TestNDTimeSeriesAlias:
    def test_alias_is_same_class(self):
        assert tdm.NDTimeSeries is tdm.TimeSeriesArray


# ---------------------------------------------------------------------------
# TestEquality
# ---------------------------------------------------------------------------

class TestEquality:
    def test_equal_arrays(self, timestamps_5h):
        data = np.ones((2, 5))
        dims = [tdm.Dimension("scenario", ["a", "b"]), tdm.Dimension("valid_time", timestamps_5h)]
        c1 = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        c2 = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data.copy())
        assert c1.equals(c2)

    def test_unequal_arrays(self, timestamps_5h):
        dims = [tdm.Dimension("scenario", ["a", "b"]), tdm.Dimension("valid_time", timestamps_5h)]
        c1 = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=np.ones((2, 5)))
        c2 = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=np.zeros((2, 5)))
        assert not c1.equals(c2)

    def test_hash_is_none(self):
        assert tdm.TimeSeriesArray.__hash__ is None


# ---------------------------------------------------------------------------
# TestXarray
# ---------------------------------------------------------------------------

class TestXarray:
    def test_to_xarray_2d(self, cube_2d):
        da = cube_2d.to_xarray()
        assert isinstance(da, xr.DataArray)
        assert da.dims == ("scenario", "valid_time")
        assert da.shape == (3, 5)
        assert da.name == "power"
        assert da.attrs["frequency"] == str(tdm.Frequency.PT1H)
        assert da.attrs["timezone"] == "UTC"
        assert da.attrs["unit"] == "MW"

    def test_to_xarray_3d(self, cube_3d):
        da = cube_3d.to_xarray()
        assert da.dims == ("scenario", "valid_time", "quantile")
        assert da.shape == (2, 5, 3)

    def test_to_xarray_masked_to_nan(self, timestamps_5h):
        data = np.array([[1.0, np.nan, 3.0, 4.0, 5.0]])
        dims = [tdm.Dimension("scenario", ["a"]), tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        da = cube.to_xarray()
        assert np.isnan(da.values[0, 1])
        assert da.values[0, 0] == 1.0

    def test_to_xarray_metadata_in_attrs(self, timestamps_5h):
        data = np.ones((2, 5))
        dims = [tdm.Dimension("scenario", ["a", "b"]), tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesArray(
            tdm.Frequency.PT1H, dimensions=dims, values=data,
            name="test", unit="kW", description="desc",
            data_type=tdm.DataType.ACTUAL,
            attributes={"key": "val"},
        )
        da = cube.to_xarray()
        assert da.attrs["unit"] == "kW"
        assert da.attrs["description"] == "desc"
        assert da.attrs["data_type"] == str(tdm.DataType.ACTUAL)
        assert "key" in da.attrs["attributes"]

    def test_round_trip_2d(self, cube_2d):
        da = cube_2d.to_xarray()
        restored = tdm.TimeSeriesArray.from_xarray(da)
        assert cube_2d.equals(restored)

    def test_round_trip_3d(self, cube_3d):
        da = cube_3d.to_xarray()
        restored = tdm.TimeSeriesArray.from_xarray(da)
        assert cube_3d.equals(restored)

    def test_round_trip_preserves_metadata(self, timestamps_5h):
        data = np.ones((2, 5))
        dims = [tdm.Dimension("scenario", ["a", "b"]), tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesArray(
            tdm.Frequency.PT1H, dimensions=dims, values=data,
            name="power", unit="MW", description="test desc",
            data_type=tdm.DataType.FORECAST,
            attributes={"source": "model"},
        )
        restored = tdm.TimeSeriesArray.from_xarray(cube.to_xarray())
        assert restored.name == "power"
        assert restored.unit == "MW"
        assert restored.description == "test desc"
        assert restored.data_type == tdm.DataType.FORECAST
        assert restored.attributes == {"source": "model"}

    def test_round_trip_string_and_float_labels(self, timestamps_5h):
        dims = [
            tdm.Dimension("scenario", ["low", "mid", "high"]),
            tdm.Dimension("valid_time", timestamps_5h),
            tdm.Dimension("quantile", [0.1, 0.5, 0.9]),
        ]
        data = np.arange(3 * 5 * 3, dtype=np.float64).reshape(3, 5, 3)
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        restored = tdm.TimeSeriesArray.from_xarray(cube.to_xarray())
        assert cube.equals(restored)

    def test_from_xarray_explicit_kwargs_override(self, cube_2d):
        da = cube_2d.to_xarray()
        restored = tdm.TimeSeriesArray.from_xarray(
            da, frequency=tdm.Frequency.P1D, timezone="CET",
            name="overridden", unit="GW",
        )
        assert restored.frequency == tdm.Frequency.P1D
        assert restored.timezone == "CET"
        assert restored.name == "overridden"
        assert restored.unit == "GW"

    def test_round_trip_masked_values(self, timestamps_5h):
        data = np.array([[1.0, np.nan, 3.0, np.nan, 5.0]])
        dims = [tdm.Dimension("s", ["a"]), tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        restored = tdm.TimeSeriesArray.from_xarray(cube.to_xarray())
        assert cube.equals(restored)


# ---------------------------------------------------------------------------
# TestArrayApplyXarray
# ---------------------------------------------------------------------------

class TestArrayApplyXarray:
    def test_arithmetic_2d(self, cube_2d):
        result = cube_2d.apply_xarray(lambda da: da * 10)
        expected = cube_2d.to_numpy() * 10
        np.testing.assert_array_almost_equal(
            np.ma.filled(result.to_numpy(), np.nan),
            np.ma.filled(expected, np.nan),
        )

    def test_arithmetic_3d(self, cube_3d):
        result = cube_3d.apply_xarray(lambda da: da + 1)
        expected = cube_3d.to_numpy() + 1
        np.testing.assert_array_almost_equal(
            np.ma.filled(result.to_numpy(), np.nan),
            np.ma.filled(expected, np.nan),
        )

    def test_metadata_preserved(self, cube_2d):
        result = cube_2d.apply_xarray(lambda da: da * 1)
        assert result.name == cube_2d.name
        assert result.unit == cube_2d.unit
        assert result.frequency == cube_2d.frequency
        assert result.timezone == cube_2d.timezone


# ---------------------------------------------------------------------------
# TestArrayApplyPandas
# ---------------------------------------------------------------------------

class TestArrayApplyPandas:
    def test_2d_arithmetic(self, cube_2d):
        result = cube_2d.apply_pandas(lambda df: df * 10)
        expected = cube_2d.to_numpy() * 10
        np.testing.assert_array_almost_equal(
            np.ma.filled(result.to_numpy(), np.nan),
            np.ma.filled(expected, np.nan),
        )

    def test_metadata_preserved(self, cube_2d):
        result = cube_2d.apply_pandas(lambda df: df * 1)
        assert result.name == cube_2d.name
        assert result.unit == cube_2d.unit

    def test_4d_array_raises(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [base + timedelta(hours=i) for i in range(3)]
        data = np.ones((2, 3, 4, 2))
        dims = [
            tdm.Dimension("scenario", ["a", "b"]),
            tdm.Dimension("valid_time", ts),
            tdm.Dimension("quantile", [0.1, 0.25, 0.5, 0.9]),
            tdm.Dimension("model", ["m1", "m2"]),
        ]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        with pytest.raises(ValueError, match="at most 2 non-time"):
            cube.apply_pandas(lambda df: df)


# ---------------------------------------------------------------------------
# TestArrayApplyPolars
# ---------------------------------------------------------------------------

class TestArrayApplyPolars:
    def test_2d_arithmetic(self, cube_2d):
        import polars as pl

        result = cube_2d.apply_polars(
            lambda df: df.with_columns([
                pl.col(c) * 10 for c in df.columns if c != "valid_time"
            ])
        )
        expected = cube_2d.to_numpy() * 10
        np.testing.assert_array_almost_equal(
            np.ma.filled(result.to_numpy(), np.nan),
            np.ma.filled(expected, np.nan),
        )

    def test_4d_array_raises(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [base + timedelta(hours=i) for i in range(3)]
        data = np.ones((2, 3, 4, 2))
        dims = [
            tdm.Dimension("scenario", ["a", "b"]),
            tdm.Dimension("valid_time", ts),
            tdm.Dimension("quantile", [0.1, 0.25, 0.5, 0.9]),
            tdm.Dimension("model", ["m1", "m2"]),
        ]
        cube = tdm.TimeSeriesArray(tdm.Frequency.PT1H, dimensions=dims, values=data)
        with pytest.raises(ValueError, match="at most 2 non-time"):
            cube.apply_polars(lambda df: df)
