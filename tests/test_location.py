import pytest

import timedatamodel as tdm


class TestGeoLocation:
    def test_valid(self):
        loc = tdm.GeoLocation(latitude=59.91, longitude=10.75)
        assert loc.latitude == 59.91
        assert loc.longitude == 10.75

    def test_boundary_values(self):
        tdm.GeoLocation(latitude=90, longitude=180)
        tdm.GeoLocation(latitude=-90, longitude=-180)

    def test_invalid_latitude(self):
        with pytest.raises(ValueError, match="latitude"):
            tdm.GeoLocation(latitude=91, longitude=0)

    def test_invalid_longitude(self):
        with pytest.raises(ValueError, match="longitude"):
            tdm.GeoLocation(latitude=0, longitude=181)

    def test_frozen(self):
        loc = tdm.GeoLocation(latitude=0, longitude=0)
        with pytest.raises(AttributeError):
            loc.latitude = 1  # type: ignore[misc]


class TestGeoArea:
    @pytest.fixture
    def triangle(self):
        coords = [(60.0, 10.0), (61.0, 11.0), (60.0, 12.0)]
        return tdm.GeoArea.from_coordinates(coords, name="test-area")

    def test_from_coordinates(self, triangle):
        assert triangle.name == "test-area"
        assert triangle.polygon.is_valid

    def test_bounds(self, triangle):
        bounds = triangle.bounds
        assert len(bounds) == 4
        # bounds = (minx, miny, maxx, maxy) = (min_lon, min_lat, max_lon, max_lat)
        assert bounds[0] == 10.0  # min lon
        assert bounds[1] == 60.0  # min lat

    def test_centroid(self, triangle):
        c = triangle.centroid
        assert isinstance(c, tdm.GeoLocation)
        assert -90 <= c.latitude <= 90
        assert -180 <= c.longitude <= 180

    def test_no_name(self):
        area = tdm.GeoArea.from_coordinates([(0, 0), (0, 1), (1, 1), (1, 0)])
        assert area.name is None
