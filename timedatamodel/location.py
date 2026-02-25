from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon


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
        # Shapely uses (x, y) = (lon, lat), so we swap
        xy_coords = [(lon, lat) for lat, lon in coords]
        return cls(polygon=Polygon(xy_coords), name=name)


Location = GeoLocation | GeoArea
