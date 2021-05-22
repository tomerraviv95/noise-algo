from shapely.geometry import MultiLineString, Polygon, LineString
from collections import defaultdict
from typing import Dict, Tuple
import numpy as np


def edge_is_contained(edge: Tuple[int, int], is_contained_in_patient_circle: np.ndarray) -> bool:
    # both inner and outer nodes are in the effective polygon
    return is_contained_in_patient_circle[edge[0]] and is_contained_in_patient_circle[edge[1]]


def edge_crosses(edge: Tuple[int, int], is_contained_in_patient_circle: np.ndarray) -> bool:
    # one of the inner and outer nodes are in the effective polygon
    return is_contained_in_patient_circle[edge[0]] or is_contained_in_patient_circle[edge[1]]


def assign_edges_labels(edges_to_roads_dict: Dict[Tuple, LineString], is_contained_in_patient_circle: np.ndarray,
                        patient_effective_polygon: Polygon, no_entrance_polygon: Polygon) -> Dict[Tuple[int,int],str]:
    """
    Assigns labels to edges:
    'contained' - edge is contained in patient_effective_polygon
    'safe_crossing' - edge crosses the patient_effective_polygon, and is outside the dangerous zone
    'danger_crossing' - edge crosses the patient_effective_polygon, and is outside the dangerous zone
    If edge is marked, aka crosses the polygon, save the contained node first and the second is the outer node
    """

    edges_labels = {}
    for edge in edges_to_roads_dict.keys():
        if edge_is_contained(edge, is_contained_in_patient_circle):
            pass
        elif edge_crosses(edge, is_contained_in_patient_circle):
            road = edges_to_roads_dict[edge]
            intersection_line = patient_effective_polygon.intersection(road)
            if type(intersection_line) == MultiLineString:
                intersection_line = max(intersection_line, key=lambda line: line.length)

            if is_contained_in_patient_circle[edge[1]]:
                edge = (edge[1], edge[0])

            if intersection_line.intersects(no_entrance_polygon):
                edges_labels[edge] = 'danger_crossing'
            else:
                edges_labels[edge] = 'safe_crossing'

    return edges_labels


def get_inner_node(edge: Tuple[int, int], is_contained_in_patient_circle: np.ndarray) -> int:
    if is_contained_in_patient_circle[edge[0]]:
        return edge[0]
    return edge[1]


def get_outer_node(edge: Tuple[int, int], is_contained_in_patient_circle: np.ndarray) -> int:
    if not is_contained_in_patient_circle[edge[0]]:
        return edge[0]
    return edge[1]


def assign_nodes_labels(edges_labels: Dict[Tuple, str], is_contained_in_patient_circle: np.ndarray) -> Dict[int, str]:
    """
    Loop over all the labeled edges, and label nodes based on the actual label
    :param edges_labels: dict from the edges (tuples of ints, each int is a node) to the underlying road
    :param is_contained_in_patient_circle: mask of nodes contained the polygon
    :return: dict from the nodes to the label
    """
    nodes_labels = defaultdict(str)
    for edge in edges_labels.keys():
        if edges_labels[edge] == 'safe_crossing' and \
                nodes_labels[get_inner_node(edge, is_contained_in_patient_circle)] != 'danger_marked':
            nodes_labels[get_inner_node(edge, is_contained_in_patient_circle)] = 'safe_marked'

        if edges_labels[edge] == 'danger_crossing':
            nodes_labels[get_outer_node(edge, is_contained_in_patient_circle)] = 'danger_marked'
    return nodes_labels
