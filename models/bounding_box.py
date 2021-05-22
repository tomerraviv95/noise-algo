from typing import NamedTuple, Tuple

from shapely.geometry import Polygon


class BoundingBox(NamedTuple):
    south: float
    west: float
    north: float
    east: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self.south, self.west, self.north, self.east

    def to_polygon(self) -> Polygon:
        return Polygon([
            (self.south, self.west),
            (self.south, self.east),
            (self.north, self.east),
            (self.north, self.west)
        ])

    def to_polygon_scaled(self) -> Polygon:
        alpha = 0.05
        lon_dist = self.east - self.west
        lat_dist = self.north - self.south
        return Polygon([
            (self.south + alpha * lat_dist, self.west + alpha * lon_dist),
            (self.south + alpha * lat_dist, self.east - alpha * lon_dist),
            (self.north - alpha * lat_dist, self.east - alpha * lon_dist),
            (self.north - alpha * lat_dist, self.west + alpha * lon_dist)
        ])

    def get_end_of_map_polygon(self) -> Polygon:
        boundary = self.to_polygon().boundary
        return boundary.buffer(0.0006)
