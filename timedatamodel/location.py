from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapely.geometry import Polygon

_EARTH_RADIUS_KM = 6371.0
_KM_TO_MI = 0.621371


@dataclass(frozen=True, slots=True)
class GeoLocation:
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"latitude must be between -90 and 90, got {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(
                f"longitude must be between -180 and 180, got {self.longitude}"
            )

    def distance_to(self, other: GeoLocation, unit: str = "km") -> float:
        """Haversine great-circle distance to *other*."""
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        km = _EARTH_RADIUS_KM * c
        if unit == "km":
            return km
        if unit == "mi":
            return km * _KM_TO_MI
        raise ValueError(f"unsupported unit {unit!r}, use 'km' or 'mi'")

    def bearing_to(self, other: GeoLocation) -> float:
        """Initial bearing in degrees [0, 360) from *self* to *other*."""
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        return math.degrees(math.atan2(x, y)) % 360

    def midpoint(self, other: GeoLocation) -> GeoLocation:
        """Geographic midpoint on the great circle."""
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        bx = math.cos(lat2) * math.cos(lon2 - lon1)
        by = math.cos(lat2) * math.sin(lon2 - lon1)
        lat3 = math.atan2(
            math.sin(lat1) + math.sin(lat2),
            math.sqrt((math.cos(lat1) + bx) ** 2 + by ** 2),
        )
        lon3 = lon1 + math.atan2(by, math.cos(lat1) + bx)
        return GeoLocation(
            latitude=math.degrees(lat3),
            longitude=math.degrees(lon3),
        )

    def offset(self, distance_km: float, bearing_deg: float) -> GeoLocation:
        """New point displaced by *distance_km* along *bearing_deg*."""
        lat1 = math.radians(self.latitude)
        lon1 = math.radians(self.longitude)
        brng = math.radians(bearing_deg)
        d = distance_km / _EARTH_RADIUS_KM
        lat2 = math.asin(
            math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brng)
        )
        lon2 = lon1 + math.atan2(
            math.sin(brng) * math.sin(d) * math.cos(lat1),
            math.cos(d) - math.sin(lat1) * math.sin(lat2),
        )
        return GeoLocation(
            latitude=math.degrees(lat2),
            longitude=math.degrees(lon2),
        )

    def is_within(self, area: GeoArea) -> bool:
        """True if this point is inside *area*."""
        return area.contains_point(self)


@dataclass(frozen=True, slots=True)
class GeoArea:
    polygon: Polygon
    name: str | None = None

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.polygon.bounds

    @property
    def centroid(self) -> GeoLocation:
        c = self.polygon.centroid
        return GeoLocation(latitude=c.y, longitude=c.x)

    @classmethod
    def from_coordinates(cls, coords: list[tuple[float, float]], name: str | None = None) -> GeoArea:
        """Create a GeoArea from a list of (lat, lon) coordinate pairs."""
        try:
            from shapely.geometry import Polygon
        except ImportError:
            raise ImportError(
                "shapely is required for GeoArea. "
                "Install it with: pip install timedatamodel[geo]"
            ) from None
        # Shapely uses (x, y) = (lon, lat), so we swap
        xy_coords = [(lon, lat) for lat, lon in coords]
        return cls(polygon=Polygon(xy_coords), name=name)

    def contains_point(self, location: GeoLocation) -> bool:
        """True if *location* is inside this area."""
        from shapely.geometry import Point
        return self.polygon.contains(Point(location.longitude, location.latitude))

    def contains_area(self, other: GeoArea) -> bool:
        """True if *other* is entirely inside this area."""
        return self.polygon.contains(other.polygon)

    def overlaps(self, other: GeoArea) -> bool:
        """True if this area and *other* share any space."""
        return self.polygon.intersects(other.polygon)

    def distance_to(self, other: GeoLocation | GeoArea) -> float:
        """Approximate distance in km (centroid-based Haversine).

        Returns 0.0 if the point is contained or the areas overlap.
        """
        if isinstance(other, GeoLocation):
            if self.contains_point(other):
                return 0.0
            return self.centroid.distance_to(other)
        if self.overlaps(other):
            return 0.0
        return self.centroid.distance_to(other.centroid)

    @classmethod
    def bounding_box(
        cls,
        center: GeoLocation,
        radius_km: float,
        name: str | None = None,
    ) -> GeoArea:
        """Create a rectangular area centered on *center* with half-side *radius_km*."""
        n = center.offset(radius_km, 0)
        s = center.offset(radius_km, 180)
        e = center.offset(radius_km, 90)
        w = center.offset(radius_km, 270)
        coords = [
            (n.latitude, w.longitude),
            (n.latitude, e.longitude),
            (s.latitude, e.longitude),
            (s.latitude, w.longitude),
        ]
        return cls.from_coordinates(coords, name=name)


Location = GeoLocation | GeoArea


def _location_to_json(location: Location | None) -> dict | None:
    """Serialize a location object to a JSON-safe dict."""
    if location is None:
        return None

    if isinstance(location, GeoLocation):
        return {
            "type": "GeoLocation",
            "latitude": location.latitude,
            "longitude": location.longitude,
        }

    if isinstance(location, GeoArea):
        coords = [
            [lat, lon] for lon, lat in list(location.polygon.exterior.coords)
        ]
        return {
            "type": "GeoArea",
            "name": location.name,
            "coordinates": coords,
        }

    raise TypeError(f"unsupported location type: {type(location).__name__}")


def _location_from_json(payload: dict | None) -> Location | None:
    """Deserialize a location dict produced by _location_to_json()."""
    if payload is None:
        return None

    if not isinstance(payload, dict):
        raise TypeError(
            f"location payload must be a dict or None, got {type(payload).__name__}"
        )

    kind = payload.get("type")
    if kind == "GeoLocation":
        return GeoLocation(
            latitude=float(payload["latitude"]),
            longitude=float(payload["longitude"]),
        )

    if kind == "GeoArea":
        raw_coords = payload.get("coordinates")
        if not isinstance(raw_coords, list) or len(raw_coords) < 3:
            raise ValueError(
                "GeoArea payload must contain at least 3 coordinate pairs"
            )
        coords = [(float(lat), float(lon)) for lat, lon in raw_coords]
        name = payload.get("name")
        return GeoArea.from_coordinates(coords, name=name)

    raise ValueError(f"unknown location type: {kind!r}")
