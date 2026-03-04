from datetime import datetime, timedelta, timezone

import pandas as pd
import polars as pl
import pytest

import timedatamodel as tdm
from timedatamodel._base import _default_dataframe_backend


@pytest.fixture(autouse=True)
def _reset_backend():
    """Ensure backend is reset to pandas after each test."""
    tdm.set_default_df("pandas")
    yield
    tdm.set_default_df("pandas")


@pytest.fixture
def sample_ts():
    base = datetime(2024, 1, 1)
    timestamps = [base + timedelta(hours=i) for i in range(5)]
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    return tdm.TimeSeriesList(
        tdm.Frequency.PT1H,
        timestamps=timestamps,
        values=values,
        name="power",
    )


@pytest.fixture
def sample_table():
    base = datetime(2024, 1, 1)
    timestamps = [base + timedelta(hours=i) for i in range(3)]
    return tdm.TimeSeriesTable(
        tdm.Frequency.PT1H,
        timestamps=timestamps,
        values=[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
        names=["a", "b"],
    )


@pytest.fixture
def sample_collection(sample_ts):
    return tdm.TimeSeriesCollection({"ts1": sample_ts})


class TestSetDefaultBackend:
    def test_default_is_pandas(self):
        assert tdm.get_default_df() == "pandas"

    def test_set_polars(self):
        tdm.set_default_df("polars")
        assert tdm.get_default_df() == "polars"

    def test_set_pandas(self):
        tdm.set_default_df("polars")
        tdm.set_default_df("pandas")
        assert tdm.get_default_df() == "pandas"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend must be 'pandas' or 'polars'"):
            tdm.set_default_df("dask")


class TestTimeSeriesDfBackend:
    def test_default_returns_pandas(self, sample_ts):
        assert isinstance(sample_ts.df, pd.DataFrame)

    def test_polars_backend(self, sample_ts):
        tdm.set_default_df("polars")
        assert isinstance(sample_ts.df, pl.DataFrame)

    def test_switch_back_to_pandas(self, sample_ts):
        tdm.set_default_df("polars")
        tdm.set_default_df("pandas")
        assert isinstance(sample_ts.df, pd.DataFrame)


class TestTimeSeriesTableDfBackend:
    def test_default_returns_pandas(self, sample_table):
        assert isinstance(sample_table.df, pd.DataFrame)

    def test_polars_backend(self, sample_table):
        tdm.set_default_df("polars")
        assert isinstance(sample_table.df, pl.DataFrame)


class TestTimeSeriesCollectionDfBackend:
    def test_default_returns_pandas(self, sample_collection):
        assert isinstance(sample_collection.df, pd.DataFrame)

    def test_polars_backend(self, sample_collection):
        tdm.set_default_df("polars")
        assert isinstance(sample_collection.df, pl.DataFrame)
