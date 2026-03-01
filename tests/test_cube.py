from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

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
    """2D cube: scenario(3) x valid_time(5)."""
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
    return tdm.TimeSeriesCube(
        tdm.Frequency.PT1H,
        dimensions=dims,
        values=data,
        name="power",
        unit="MW",
    )


@pytest.fixture
def cube_3d(timestamps_5h):
    """3D cube: scenario(2) x valid_time(5) x quantile(3)."""
    scenarios = ["low", "high"]
    quantiles = [0.1, 0.5, 0.9]
    data = np.arange(2 * 5 * 3, dtype=np.float64).reshape(2, 5, 3)
    dims = [
        tdm.Dimension("scenario", scenarios),
        tdm.Dimension("valid_time", timestamps_5h),
        tdm.Dimension("quantile", quantiles),
    ]
    return tdm.TimeSeriesCube(
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
        cube = tdm.TimeSeriesCube(tdm.Frequency.PT1H, dimensions=dims, values=data)
        assert cube.shape == (3, 5)

    def test_3d_array(self, timestamps_5h):
        data = np.ones((2, 5, 4))
        dims = [
            tdm.Dimension("scenario", ["a", "b"]),
            tdm.Dimension("valid_time", timestamps_5h),
            tdm.Dimension("quantile", [0.1, 0.25, 0.5, 0.9]),
        ]
        cube = tdm.TimeSeriesCube(tdm.Frequency.PT1H, dimensions=dims, values=data)
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
        cube = tdm.TimeSeriesCube(tdm.Frequency.PT1H, dimensions=dims, values=data)
        assert cube.ndim == 4

    def test_plain_ndarray_wraps_nan(self, timestamps_5h):
        data = np.array([[1.0, np.nan, 3.0, 4.0, 5.0]])
        dims = [tdm.Dimension("scenario", ["a"]), tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesCube(tdm.Frequency.PT1H, dimensions=dims, values=data)
        assert isinstance(cube._values, np.ma.MaskedArray)
        assert cube._values.mask[0, 1]

    def test_masked_array_preserved(self, timestamps_5h):
        data = np.ma.MaskedArray(
            np.ones((2, 5)),
            mask=[[False]*5, [True]+[False]*4],
        )
        dims = [tdm.Dimension("scenario", ["a", "b"]), tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesCube(tdm.Frequency.PT1H, dimensions=dims, values=data)
        assert cube._values.mask[1, 0]

    def test_shape_mismatch_raises(self, timestamps_5h):
        data = np.ones((3, 4))  # wrong: 4 != 5
        dims = [tdm.Dimension("scenario", ["a", "b", "c"]), tdm.Dimension("valid_time", timestamps_5h)]
        with pytest.raises(ValueError, match="does not match"):
            tdm.TimeSeriesCube(tdm.Frequency.PT1H, dimensions=dims, values=data)


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
        cube = tdm.TimeSeriesCube(tdm.Frequency.PT1H, dimensions=dims, values=data)
        assert cube.primary_time_dim.name == "time_axis"

    def test_primary_time_dim_first_fallback(self):
        dims = [tdm.Dimension("x", ["a", "b"]), tdm.Dimension("y", ["c", "d"])]
        data = np.ones((2, 2))
        cube = tdm.TimeSeriesCube(tdm.Frequency.NONE, dimensions=dims, values=data)
        assert cube.primary_time_dim.name == "x"

    def test_begin_end(self, cube_2d, timestamps_5h):
        assert cube_2d.begin == timestamps_5h[0]
        assert cube_2d.end == timestamps_5h[-1]

    def test_has_missing_false(self, cube_2d):
        assert not cube_2d.has_missing

    def test_has_missing_true(self, timestamps_5h):
        data = np.array([[1.0, np.nan, 3.0, 4.0, 5.0]])
        dims = [tdm.Dimension("scenario", ["a"]), tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesCube(tdm.Frequency.PT1H, dimensions=dims, values=data)
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
        assert isinstance(result, tdm.TimeSeriesTable)
        assert len(result) == 3

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

    def test_sel_3d_keeps_cube(self, cube_3d, timestamps_5h):
        t0, t1 = timestamps_5h[0], timestamps_5h[2]
        result = cube_3d.sel(valid_time=slice(t0, t1))
        assert isinstance(result, tdm.TimeSeriesCube)
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
        assert isinstance(result, tdm.TimeSeriesTable)
        assert len(result) == 3

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
        cube = tdm.TimeSeriesCube.from_timeseries_list([s1, s2], dim)
        assert cube.shape == (2, 5)
        assert cube.dim_names == ("scenario", "valid_time")

    def test_ragged_timestamps(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts1 = [base, base + timedelta(hours=1), base + timedelta(hours=2)]
        ts2 = [base + timedelta(hours=1), base + timedelta(hours=2), base + timedelta(hours=3)]
        s1 = tdm.TimeSeries(tdm.Frequency.PT1H, timestamps=ts1, values=[1.0, 2.0, 3.0])
        s2 = tdm.TimeSeries(tdm.Frequency.PT1H, timestamps=ts2, values=[10.0, 20.0, 30.0])
        dim = tdm.Dimension("scenario", ["a", "b"])
        cube = tdm.TimeSeriesCube.from_timeseries_list([s1, s2], dim)
        # Union has 4 timestamps
        assert cube.shape == (2, 4)
        assert cube.has_missing  # gaps at edges

    def test_label_count_mismatch(self, timestamps_5h):
        s1 = tdm.TimeSeries(tdm.Frequency.PT1H, timestamps=timestamps_5h, values=[1.0]*5)
        dim = tdm.Dimension("scenario", ["a", "b"])  # 2 labels but 1 series
        with pytest.raises(ValueError, match="labels"):
            tdm.TimeSeriesCube.from_timeseries_list([s1], dim)

    def test_empty_list_raises(self):
        dim = tdm.Dimension("scenario", [])
        with pytest.raises(ValueError, match="empty"):
            tdm.TimeSeriesCube.from_timeseries_list([], dim)


# ---------------------------------------------------------------------------
# TestMaskedRoundTrip
# ---------------------------------------------------------------------------

class TestMaskedRoundTrip:
    def test_masked_survives_to_pandas(self, timestamps_5h):
        data = np.array([[1.0, np.nan, 3.0, 4.0, 5.0]])
        dims = [tdm.Dimension("scenario", ["a"]), tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesCube(tdm.Frequency.PT1H, dimensions=dims, values=data)
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
        cube = tdm.TimeSeriesCube(tdm.Frequency.PT1H, dimensions=dims, values=data)
        assert cube.has_missing
        cb = cube.coverage_bar()
        assert isinstance(cb, tdm.CoverageBar)

    def test_1d_cube_coverage(self, timestamps_5h):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dims = [tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesCube(tdm.Frequency.PT1H, dimensions=dims, values=data)
        cb = cube.coverage_bar()
        assert isinstance(cb, tdm.CoverageBar)


# ---------------------------------------------------------------------------
# TestRepr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_class_and_dims(self, cube_2d):
        r = repr(cube_2d)
        assert "TimeSeriesCube" in r
        assert "scenario" in r
        assert "valid_time" in r

    def test_repr_html_contains_class(self, cube_2d):
        html = cube_2d._repr_html_()
        assert "TimeSeriesCube" in html

    def test_repr_html_1d(self, timestamps_5h):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dims = [tdm.Dimension("valid_time", timestamps_5h)]
        cube = tdm.TimeSeriesCube(tdm.Frequency.PT1H, dimensions=dims, values=data)
        html = cube._repr_html_()
        assert "TimeSeriesCube" in html


# ---------------------------------------------------------------------------
# TestNDTimeSeriesAlias
# ---------------------------------------------------------------------------

class TestNDTimeSeriesAlias:
    def test_alias_is_same_class(self):
        assert tdm.NDTimeSeries is tdm.TimeSeriesCube


# ---------------------------------------------------------------------------
# TestEquality
# ---------------------------------------------------------------------------

class TestEquality:
    def test_equal_cubes(self, timestamps_5h):
        data = np.ones((2, 5))
        dims = [tdm.Dimension("scenario", ["a", "b"]), tdm.Dimension("valid_time", timestamps_5h)]
        c1 = tdm.TimeSeriesCube(tdm.Frequency.PT1H, dimensions=dims, values=data)
        c2 = tdm.TimeSeriesCube(tdm.Frequency.PT1H, dimensions=dims, values=data.copy())
        assert c1.equals(c2)

    def test_unequal_cubes(self, timestamps_5h):
        dims = [tdm.Dimension("scenario", ["a", "b"]), tdm.Dimension("valid_time", timestamps_5h)]
        c1 = tdm.TimeSeriesCube(tdm.Frequency.PT1H, dimensions=dims, values=np.ones((2, 5)))
        c2 = tdm.TimeSeriesCube(tdm.Frequency.PT1H, dimensions=dims, values=np.zeros((2, 5)))
        assert not c1.equals(c2)

    def test_hash_is_none(self):
        assert tdm.TimeSeriesCube.__hash__ is None
