import pytest
import timedatamodel as tdm

# ---- GeoLocation methods --------------------------------------------------


class TestGeoLocationDistance:
    def test_same_point(self):
        loc = tdm.GeoLocation(latitude=59.91, longitude=10.75)
        assert loc.distance_to(loc) == 0.0

    def test_oslo_to_bergen(self):
        oslo = tdm.GeoLocation(latitude=59.91, longitude=10.75)
        bergen = tdm.GeoLocation(latitude=60.39, longitude=5.32)
        dist = oslo.distance_to(bergen)
        assert 300 < dist < 310  # ~305 km

    def test_unit_miles(self):
        oslo = tdm.GeoLocation(latitude=59.91, longitude=10.75)
        bergen = tdm.GeoLocation(latitude=60.39, longitude=5.32)
        km = oslo.distance_to(bergen, unit="km")
        mi = oslo.distance_to(bergen, unit="mi")
        assert abs(mi - km * 0.621371) < 0.1

    def test_invalid_unit(self):
        loc = tdm.GeoLocation(latitude=0, longitude=0)
        with pytest.raises(ValueError, match="unsupported unit"):
            loc.distance_to(loc, unit="meters")

    def test_antipodal(self):
        a = tdm.GeoLocation(latitude=0, longitude=0)
        b = tdm.GeoLocation(latitude=0, longitude=180)
        dist = a.distance_to(b)
        # Half circumference ≈ 20015 km
        assert 20000 < dist < 20100


class TestGeoLocationBearing:
    def test_due_north(self):
        a = tdm.GeoLocation(latitude=0, longitude=0)
        b = tdm.GeoLocation(latitude=10, longitude=0)
        assert abs(a.bearing_to(b) - 0.0) < 0.1

    def test_due_east(self):
        a = tdm.GeoLocation(latitude=0, longitude=0)
        b = tdm.GeoLocation(latitude=0, longitude=10)
        assert abs(a.bearing_to(b) - 90.0) < 0.1

    def test_due_south(self):
        a = tdm.GeoLocation(latitude=10, longitude=0)
        b = tdm.GeoLocation(latitude=0, longitude=0)
        assert abs(a.bearing_to(b) - 180.0) < 0.1

    def test_due_west(self):
        a = tdm.GeoLocation(latitude=0, longitude=0)
        b = tdm.GeoLocation(latitude=0, longitude=-10)
        assert abs(a.bearing_to(b) - 270.0) < 0.1


class TestGeoLocationMidpoint:
    def test_equator(self):
        a = tdm.GeoLocation(latitude=0, longitude=0)
        b = tdm.GeoLocation(latitude=0, longitude=10)
        mid = a.midpoint(b)
        assert abs(mid.latitude) < 0.01
        assert abs(mid.longitude - 5.0) < 0.01

    def test_same_point(self):
        loc = tdm.GeoLocation(latitude=45, longitude=90)
        mid = loc.midpoint(loc)
        assert abs(mid.latitude - 45) < 0.01
        assert abs(mid.longitude - 90) < 0.01


class TestGeoLocationOffset:
    def test_round_trip(self):
        start = tdm.GeoLocation(latitude=60.0, longitude=10.0)
        bearing = start.bearing_to(tdm.GeoLocation(latitude=61.0, longitude=10.0))
        moved = start.offset(100.0, bearing)
        dist_back = moved.distance_to(start)
        assert abs(dist_back - 100.0) < 0.5

    def test_zero_distance(self):
        start = tdm.GeoLocation(latitude=60.0, longitude=10.0)
        moved = start.offset(0.0, 45.0)
        assert abs(moved.latitude - start.latitude) < 0.001
        assert abs(moved.longitude - start.longitude) < 0.001


class TestGeoLocationIsWithin:
    @pytest.fixture
    def square_area(self):
        return tdm.GeoArea.from_coordinates(
            [
                (59.0, 9.0),
                (59.0, 12.0),
                (61.0, 12.0),
                (61.0, 9.0),
            ]
        )

    def test_inside(self, square_area):
        loc = tdm.GeoLocation(latitude=60.0, longitude=10.0)
        assert loc.is_within(square_area) is True

    def test_outside(self, square_area):
        loc = tdm.GeoLocation(latitude=65.0, longitude=10.0)
        assert loc.is_within(square_area) is False


# ---- GeoArea methods ------------------------------------------------------


class TestGeoAreaContains:
    @pytest.fixture
    def large_area(self):
        return tdm.GeoArea.from_coordinates(
            [
                (58.0, 4.0),
                (58.0, 12.0),
                (62.0, 12.0),
                (62.0, 4.0),
            ]
        )

    @pytest.fixture
    def small_area(self):
        return tdm.GeoArea.from_coordinates(
            [
                (59.0, 5.0),
                (59.0, 6.0),
                (60.0, 6.0),
                (60.0, 5.0),
            ]
        )

    def test_contains_point(self, large_area):
        loc = tdm.GeoLocation(latitude=60.0, longitude=8.0)
        assert large_area.contains_point(loc) is True

    def test_not_contains_point(self, large_area):
        loc = tdm.GeoLocation(latitude=70.0, longitude=8.0)
        assert large_area.contains_point(loc) is False

    def test_contains_area(self, large_area, small_area):
        assert large_area.contains_area(small_area) is True
        assert small_area.contains_area(large_area) is False

    def test_overlaps(self, large_area, small_area):
        assert large_area.overlaps(small_area) is True
        assert small_area.overlaps(large_area) is True

    def test_no_overlap(self, large_area):
        far = tdm.GeoArea.from_coordinates(
            [
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
                (1.0, 0.0),
            ]
        )
        assert large_area.overlaps(far) is False


class TestGeoAreaDistance:
    def test_contained_point(self):
        area = tdm.GeoArea.from_coordinates(
            [
                (59.0, 9.0),
                (59.0, 12.0),
                (61.0, 12.0),
                (61.0, 9.0),
            ]
        )
        loc = tdm.GeoLocation(latitude=60.0, longitude=10.0)
        assert area.distance_to(loc) == 0.0

    def test_outside_point(self):
        area = tdm.GeoArea.from_coordinates(
            [
                (59.0, 9.0),
                (59.0, 12.0),
                (61.0, 12.0),
                (61.0, 9.0),
            ]
        )
        loc = tdm.GeoLocation(latitude=70.0, longitude=10.0)
        assert area.distance_to(loc) > 0.0

    def test_overlapping_areas(self):
        a = tdm.GeoArea.from_coordinates(
            [
                (59.0, 9.0),
                (59.0, 12.0),
                (61.0, 12.0),
                (61.0, 9.0),
            ]
        )
        b = tdm.GeoArea.from_coordinates(
            [
                (60.0, 10.0),
                (60.0, 13.0),
                (62.0, 13.0),
                (62.0, 10.0),
            ]
        )
        assert a.distance_to(b) == 0.0


class TestGeoAreaBoundingBox:
    def test_creates_valid_area(self):
        center = tdm.GeoLocation(latitude=60.0, longitude=10.0)
        bbox = tdm.GeoArea.bounding_box(center, 50.0, name="box")
        assert bbox.name == "box"
        assert bbox.polygon.is_valid
        assert bbox.contains_point(center)

    def test_approximate_size(self):
        center = tdm.GeoLocation(latitude=60.0, longitude=10.0)
        bbox = tdm.GeoArea.bounding_box(center, 50.0)
        # Centroid should be close to center
        c = bbox.centroid
        assert abs(c.latitude - center.latitude) < 0.5
        assert abs(c.longitude - center.longitude) < 0.5
