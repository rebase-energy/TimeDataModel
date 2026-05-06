"""Tests for the metadata-only mode of :class:`TimeSeries` (``df=None``)."""

import polars as pl
import pytest
from timedatamodel import (
    DataShape,
    DataType,
    Frequency,
    TimeSeries,
    TimeSeriesType,
)


class TestTimeSeriesMetadataOnly:
    """A TimeSeries constructed without a DataFrame holds metadata only."""

    def test_default_construction(self):
        ts = TimeSeries(name="power")
        assert ts.name == "power"
        assert ts.unit == "dimensionless"
        assert ts.data_type is None
        assert ts.timeseries_type == TimeSeriesType.FLAT
        assert ts.frequency is None
        assert ts.timezone == "UTC"
        assert ts.description is None

    def test_name_required(self):
        with pytest.raises(TypeError):
            TimeSeries()  # type: ignore[call-arg]

    def test_all_metadata_fields(self):
        ts = TimeSeries(
            name="power",
            unit="MW",
            data_type=DataType.FORECAST,
            timeseries_type=TimeSeriesType.OVERLAPPING,
            frequency=Frequency.PT1H,
            timezone="Europe/Stockholm",
            description="Wind power forecast",
        )
        assert ts.unit == "MW"
        assert ts.data_type == DataType.FORECAST
        assert ts.timeseries_type == TimeSeriesType.OVERLAPPING
        assert ts.frequency == Frequency.PT1H
        assert ts.timezone == "Europe/Stockholm"
        assert ts.description == "Wind power forecast"

    def test_has_df_false(self):
        assert TimeSeries(name="power").has_df is False

    def test_df_attribute_is_none(self):
        assert TimeSeries(name="power").df is None

    def test_shape_is_none(self):
        assert TimeSeries(name="power").shape is None

    def test_counts_return_zero(self):
        ts = TimeSeries(name="power")
        assert len(ts) == 0
        assert ts.num_rows == 0
        assert ts.has_missing is False
        assert ts.columns == []


class TestTimeSeriesMetadataOnlyMethodGuards:
    """Methods that need data raise ValueError on metadata-only TimeSeries."""

    @pytest.fixture
    def ts(self):
        return TimeSeries(name="power", unit="MW", data_type=DataType.ACTUAL)

    def test_to_pandas_raises(self, ts):
        with pytest.raises(ValueError, match="no data"):
            ts.to_pandas()

    def test_to_polars_raises(self, ts):
        with pytest.raises(ValueError, match="no data"):
            ts.to_polars()

    def test_to_list_raises(self, ts):
        with pytest.raises(ValueError, match="no data"):
            ts.to_list()

    def test_to_numpy_raises(self, ts):
        with pytest.raises(ValueError, match="no data"):
            ts.to_numpy()

    def test_to_pyarrow_raises(self, ts):
        pytest.importorskip("pyarrow")
        with pytest.raises(ValueError, match="no data"):
            ts.to_pyarrow()

    def test_head_raises(self, ts):
        with pytest.raises(ValueError, match="no data"):
            ts.head()

    def test_tail_raises(self, ts):
        with pytest.raises(ValueError, match="no data"):
            ts.tail()

    def test_convert_unit_raises(self, ts):
        pytest.importorskip("pint")
        with pytest.raises(ValueError, match="no data"):
            ts.convert_unit("kW")

    def test_coverage_bar_raises(self, ts):
        with pytest.raises(ValueError, match="no data"):
            ts.coverage_bar()

    def test_validate_for_insert_raises(self, ts):
        with pytest.raises(ValueError, match="no data"):
            ts.validate_for_insert()


class TestTimeSeriesMetadataDict:
    """metadata_dict() handles both metadata-only and data-bearing instances."""

    def test_metadata_only(self):
        ts = TimeSeries(
            name="power",
            unit="MW",
            data_type=DataType.FORECAST,
            timeseries_type=TimeSeriesType.OVERLAPPING,
        )
        d = ts.metadata_dict()
        assert d["name"] == "power"
        assert d["unit"] == "MW"
        assert d["data_type"] == DataType.FORECAST.value
        assert d["timeseries_type"] == TimeSeriesType.OVERLAPPING.value
        assert d["shape"] is None
        assert d["num_rows"] == 0

    def test_with_data(self):
        df = pl.DataFrame(
            {
                "valid_time": [None],
                "value": [1.0],
            }
        ).cast({"valid_time": pl.Datetime("us", "UTC")})
        ts = TimeSeries(df, name="power", unit="MW", data_type=DataType.ACTUAL)
        d = ts.metadata_dict()
        assert d["shape"] == DataShape.SIMPLE.value
        assert d["num_rows"] == 1


class TestTimeSeriesMetadataOnlyRepr:
    """Repr renders metadata-only without crashing on None shape."""

    def test_text_repr(self):
        ts = TimeSeries(
            name="power",
            unit="MW",
            data_type=DataType.FORECAST,
            description="Wind power forecast",
        )
        text = repr(ts)
        assert "TimeSeries" in text
        assert "power" in text
        assert "MW" in text
        assert "Shape" not in text
        assert "Rows" not in text

    def test_html_repr(self):
        ts = TimeSeries(name="power", unit="MW")
        html = ts._repr_html_()
        assert "TimeSeries" in html
        assert "power" in html


class TestDescriptorRemoval:
    """The TimeSeriesDescriptor name and conversion helpers are gone."""

    def test_descriptor_not_exported(self):
        import timedatamodel as tdm

        assert not hasattr(tdm, "TimeSeriesDescriptor")

    def test_to_descriptor_removed(self):
        ts = TimeSeries(name="power")
        assert not hasattr(ts, "to_descriptor")

    def test_from_descriptor_removed(self):
        assert not hasattr(TimeSeries, "from_descriptor")
