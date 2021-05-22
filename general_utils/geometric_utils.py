from typing import List, Dict

import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import cascaded_union

from general_utils.geographic_utils import circle_around_point
from general_utils.python_utils import make_iterable

EPS = 1e-10


def point_in_multipolygon(points: Dict, multipolygon: Polygon) -> np.ndarray:
    """Returns whether points are contained in multipolygon

    :param points: 2D numpy array of points
    :param multipolygon: multipolygon
    :return: 2D numpy array of booleans
    """
    contains = np.zeros(len(points)).astype(bool)
    for point_index in range(len(points)):
        contains[point_index] = multipolygon.contains(Point(points[point_index]))
    return contains


def create_buffered_polygons(polygons: List[Polygon], buffer_size_in_geometric_units: float) -> MultiPolygon:
    """Creates a buffered union of polygons

    :param polygons: list of polygons
    :param buffer_size_in_geometric_units: buffer size in geometric units (same as polygons units)
    :return: buffered union of all polygons
    """
    buffered_polygons = [polygon.buffer(buffer_size_in_geometric_units) for polygon in polygons]
    cu = cascaded_union(buffered_polygons)
    return MultiPolygon(make_iterable(cu))


def create_buffered_polygon_around_coord(coord: np.ndarray, buffer_in_meters: float) -> Polygon:
    """Create multipolygon from coordinate and buffer

    :param coord: input coordinate
    :param buffer_in_meters: buffer in meters
    :return: multipolygon which is coordinate and buffer around it
    """
    buffered_coord = np.rad2deg(circle_around_point(np.deg2rad(coord), buffer_in_meters))
    return Polygon(buffered_coord)


def external_tangents_of_two_circles(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float) -> np.ndarray:
    """Given two circles, returns the two external tangents of the two circles

    :param c1: center of the first circle
    :param r1: radius of the first circle
    :param c2: center of the second circle
    :param r2: radius of the second circle
    :return: four points representing the two tangents
    """
    if np.any(c1 == c2):
        c1 = c1 + EPS

    c1 = c1.reshape(2, 1)
    c2 = c2.reshape(2, 1)

    dc = c2 - c1
    dr = r2 - r1
    d = np.linalg.norm(dc)

    XY = dc / d
    R = dr / d

    k = np.array([-1, 1])

    ab = R * XY + k * np.dot(np.array([[0, -1], [1, 0]]), XY) * np.sqrt(1 - R ** 2)
    c = r1 - np.dot(ab.T, c1)
    circles = np.hstack((c1, c2))
    x_p = (circles[0].reshape(-1, 1) * ab[1] ** 2 - (np.prod(ab, axis=0).reshape(-1, 1) * circles[1]).T - ab[
        0] * c.T) / (np.linalg.norm(ab, axis=0) ** 2)
    y_p = (-ab[0] / ab[1]) * x_p - c.T / ab[1]

    return np.dstack((x_p.T, y_p.T))


def circle_points(center: np.ndarray, radius: float, steps: int = 60, endpoint: bool = False) -> np.ndarray:
    """Returns points which approximate the circle

    :param center: center of the circle
    :param radius: radius of the circle
    :param steps: number of points representing the circle, default is 60
    :param endpoint: whether to add the last point to the approximation, default is false
    :return:
    """
    points = [
        [center[0] + np.cos(angle) * radius, center[1] + np.sin(angle) * radius]
        for angle in np.linspace(0, 2 * np.pi, steps, endpoint=endpoint)]
    return np.array(points).reshape(-1, 2)


def merge_circles(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float) -> np.ndarray:
    """Given two circles, creates a merged polygon by adding the area between the external tangents of the two circles

    :param c1: center of the first circle
    :param r1: radius of the first circle
    :param c2: center of the second circle
    :param r2: radius of the second circle
    :return: 2D numpy array containing exterior point of the merged polygon created
    """
    if is_circle_contained_completely(c1, r1, c2, r2):
        return circle_points(c2, r2, endpoint=True)

    if is_circle_contained_completely(c2, r2, c1, r1):
        return circle_points(c1, r1, endpoint=True)

    lines_np = external_tangents_of_two_circles(c1, r1, c2, r2)
    rect = np.vstack([lines_np[0], lines_np[1, ::-1]])

    p1 = Polygon(circle_points(c1, r1))
    p2 = Polygon(circle_points(c2, r2))
    p3 = Polygon(rect)

    p: Polygon = cascaded_union([p1, p2, p3])
    return np.array(p.exterior.xy).T


def is_circle_contained_completely(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float) -> bool:
    """Returns whether circle 1 in completely inside circle 2

    :param c1: center of the first circle
    :param r1: radius of the first circle
    :param c2: center of the second circle
    :param r2: radius of the second circle
    :return: true iff circle 1 in completely inside circle 2
    """
    return np.linalg.norm(c1 - c2) + r1 <= r2
