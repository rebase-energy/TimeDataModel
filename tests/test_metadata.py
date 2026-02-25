import pytest

from timedatamodel import DataType, GeoLocation, Metadata, StorageType


class TestMetadata:
    def test_defaults(self):
        m = Metadata()
        assert m.unit is None
        assert m.data_type is None
        assert m.location is None
        assert m.name is None
        assert m.description is None
        assert m.storage_type == StorageType.FLAT
        assert m.attributes == {}

    def test_full_construction(self):
        loc = GeoLocation(latitude=59.91, longitude=10.75)
        m = Metadata(
            unit="MW",
            data_type=DataType.ACTUAL,
            location=loc,
            name="power",
            description="Power generation",
            storage_type=StorageType.FLAT,
            attributes={"source": "test"},
        )
        assert m.unit == "MW"
        assert m.data_type == DataType.ACTUAL
        assert m.location == loc
        assert m.name == "power"
        assert m.attributes["source"] == "test"

    def test_pint_unit_valid(self):
        m = Metadata(unit="MW")
        u = m.pint_unit
        assert str(u) == "megawatt"

    def test_pint_unit_none(self):
        m = Metadata()
        with pytest.raises(ValueError, match="unit is not set"):
            m.pint_unit

    def test_pint_unit_invalid(self):
        m = Metadata(unit="not_a_real_unit_xyz")
        with pytest.raises(ValueError, match="invalid unit string"):
            m.pint_unit

    def test_frozen(self):
        m = Metadata(unit="MW")
        with pytest.raises(AttributeError):
            m.unit = "kW"  # type: ignore[misc]
