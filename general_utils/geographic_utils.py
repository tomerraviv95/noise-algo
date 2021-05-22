import math
from typing import Tuple

import numpy as np

EARTH_RADIUS_IN_KM = 6371
METERS_PER_KM = 1000
NM_TO_METERS = 1e-9


def distance(first_coordinate_in_rad: np.ndarray, second_coordinate_in_rad: np.ndarray) -> float:
    """Calculates distance between two points

    :param first_coordinate_in_rad: first coordinate in radians
    :param second_coordinate_in_rad: second coordinate in radians
    :return: distance in meters
    """
    p1 = first_coordinate_in_rad.reshape(2)
    p2 = second_coordinate_in_rad.reshape(2)
    d_lat = p2[0] - p1[0]
    d_lon = p2[1] - p1[1]

    a = np.sin(d_lat / 2) * np.sin(d_lat / 2) + \
        np.cos(p1[0]) * np.cos(p2[0]) * \
        np.sin(d_lon / 2) * np.sin(d_lon / 2)

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = EARTH_RADIUS_IN_KM * METERS_PER_KM * c

    return d


def distance_approx(first_coordinate_in_rad: np.ndarray, second_coordinate_in_rad: np.ndarray) -> float:
    """Calculates approximation of distance between two coordinates

    :param first_coordinate_in_rad: first coordinate in radians
    :param second_coordinate_in_rad: second coordinate in radians
    :return: approximated distance in meters
    """
    p1 = first_coordinate_in_rad.reshape(2)
    p2 = second_coordinate_in_rad.reshape(2)
    dy = 12430 * np.abs(p1[0] - p2[0]) / np.pi
    dx = 24901 * np.abs(p1[1] - p2[1]) / (2 * np.pi) * np.cos((p1[0] + p2[0]) / 2)
    return np.sqrt(np.power(dx, 2) + np.power(dy, 2)) * 1609.34


def destination_coord_from_start_coord_and_angle(start_in_rad: np.ndarray,
                                                 angle_in_radians: float,
                                                 distance_in_meters: float) -> np.ndarray:
    """Calculates a coordinate from a start coordinate, an angle and a distance

    :param start_in_rad: start coordinate/s in radians (2D numpy array)
    :param angle_in_radians: angle of destination coordinate in radians (clockwise, north is zero)
    :param distance_in_meters: distance of destination point in meters
    :return: destination coordinate/s in radians (2D numpy array)
    """
    distance_in_km = distance_in_meters / METERS_PER_KM
    start_in_rad = start_in_rad.reshape(-1, 2)
    dist = np.zeros(start_in_rad.shape)
    dist[:, 0] = np.arcsin(np.sin(start_in_rad[:, 0]) * np.cos(distance_in_km / EARTH_RADIUS_IN_KM) +
                           np.cos(start_in_rad[:, 0]) * np.sin(distance_in_km / EARTH_RADIUS_IN_KM) *
                           np.cos(angle_in_radians))
    dist[:, 1] = start_in_rad[:, 1] + np.arctan2(
        np.sin(angle_in_radians) * np.sin(distance_in_km / EARTH_RADIUS_IN_KM) * np.cos(start_in_rad[:, 0]),
        np.cos(distance_in_km / EARTH_RADIUS_IN_KM) - np.sin(start_in_rad[:, 0]) * np.sin(dist[:, 0]))
    return dist


def circle_around_point(coordinate_in_rad: np.ndarray, radius_in_meters: float,
                        resolution: int = 60) -> np.ndarray:
    """Generates points around the given point in given distance

    :param coordinate_in_rad: given coordinate in radians
    :param radius_in_meters: radius from coordinate in meters
    :param resolution: amount of coordinates in the output
    :return: 2D array of coordinates in radians
    """
    points = [
        destination_coord_from_start_coord_and_angle(coordinate_in_rad, angle, radius_in_meters)
        for angle in np.linspace(0, 2 * np.pi, resolution, endpoint=False)]
    return np.array(points).reshape(-1, 2)


def delta_east_and_north(point1: np.ndarray, point2: np.ndarray) -> Tuple[float, float]:
    """
    Calculates approximated difference in lat and lon in rad
    :param point1: lat,lon in rad
    :param point2: lat,lon in rad
    :return: lat,lon difference
    """
    d_lat = np.rad2deg(point2[0] - point1[0])
    d_lon = np.rad2deg(point2[1] - point1[1])
    d_north = 60 * d_lat
    d_east = d_lon * 60 * np.cos((point2[0] + point1[0]) / 2)
    return d_east * NM_TO_METERS, d_north * NM_TO_METERS


def calculate_bearing(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculates the bearing, the degree between the vector from point1 to point2 and a unit vector directed at the north
    :param point1: lat,lon in rad
    :param point2: lat,lon in rad
    :return: degree in radians
    """
    d_east, d_north = delta_east_and_north(point1, point2)
    angle_in_radians = np.arctan2(d_east, d_north) % (2 * math.pi)
    return angle_in_radians
