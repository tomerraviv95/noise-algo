from typing import List, Dict, Tuple
from shapely.geometry import LineString, MultiPoint, MultiLineString, Point, GeometryCollection
from shapely.ops import split
import numpy as np


def get_all_roads_in_radius(roads_linestrings: Dict[Tuple, LineString], current_road_center: Tuple[float, float]) -> \
        List[LineString]:
    """
    Find all roads in some radius for fast query
    :param roads_linestrings: dict from centers of roads to the roads
    :param current_road_center: queried road centers
    :return:list of roads linestrings
    """
    roads_centers = np.array([road_center for road_center in roads_linestrings.keys()])
    distance_to_centers = np.linalg.norm(roads_centers - current_road_center, axis=1)
    close_by_centers_mask = np.bitwise_and(distance_to_centers < 2 * roads_linestrings[current_road_center].length,
                                           distance_to_centers > 0)
    relevant_centers = roads_centers[close_by_centers_mask].tolist()
    return [road_linestring for road_center, road_linestring in roads_linestrings.items() if
            list(road_center) in relevant_centers]


def get_representative_points_from_geometry(intersections: List, intersection_geometry: GeometryCollection):
    """
    Extract the representative point/points from the intersection geometry, based on it's type.
    Add to the intersections list.
    :param intersections:
    :param intersection_geometry:
    """
    if type(intersection_geometry) == Point:
        intersections.append(intersection_geometry)
    elif type(intersection_geometry) == MultiPoint:
        for point in intersection_geometry:
            intersections.append(point)
    elif type(intersection_geometry) == LineString:
        intersections.append(intersection_geometry.centroid)
    elif type(intersection_geometry) == MultiLineString:
        for linestring in intersection_geometry:
            intersections.append(linestring.centroid)
    else:
        to_add = [point for point in intersection_geometry if type(point) == Point]
        if len(to_add) == 0:
            raise Exception("Encountered error in intersection of roads_heralds format")
        intersections.append(to_add[0])


def calculate_intersecting_linestrings(roads_linestrings: Dict[Tuple, LineString]) -> List[LineString]:
    """
    Break all linestrings into elemental linestrings by the intersection nodes.
    :param roads_linestrings: dict from centers of roads to the roads
    :return: list of all the elemental linestrings (breaked by the intersection points)
    """
    intersected_linestrings = []
    for current_road_center, current_linestring in roads_linestrings.items():
        # query all nearby roads
        relevant_linestrings = get_all_roads_in_radius(roads_linestrings, current_road_center)

        # check if nearby road intersects the current road. If it does, add one relevant point from the
        # intersection geometry to a list
        intersections = []
        for relevant_linestring in relevant_linestrings:
            if current_linestring.intersects(relevant_linestring):
                intersection_geometry = current_linestring.intersection(relevant_linestring)
                get_representative_points_from_geometry(intersections, intersection_geometry)

        # take all the points and split the linestring into multiple linestring, with the intersections as the junctions
        if len(intersections) > 0:
            intersections_multipoint = MultiPoint(intersections)
            splitted_linestring = split(current_linestring, intersections_multipoint)
            for linestring in splitted_linestring:
                intersected_linestrings.append(linestring)

    return intersected_linestrings
