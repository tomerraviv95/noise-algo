from typing import Dict, Tuple

from shapely.geometry import MultiPoint, LineString, Polygon
import numpy as np

from roads_heralds.roads_to_networks.roads_utils import get_road_by_edge_key


def get_edges_ending_at_node(node: int, edges_labels: Dict[Tuple, str]):
    edges = []
    for edge in edges_labels.keys():
        if edge[1] == node:
            edges.append(edge)
    return edges


def update_nodes_plotting_locs(nodes_labels: Dict[int, str],
                               edges_labels: Dict[Tuple[int, int], str],
                               edges_to_roads_dict: Dict[Tuple[int, int], LineString],
                               nodes_geo_locs: Dict[int, Tuple[float, float]],
                               patient_effective_polygon: Polygon):
    """
    Move every safe-marked node to the boundary of the effective polygon, if there is not filtered edge that ends
    at this node
    :param nodes_labels: dict mapping from node to it's label
    :param edges_labels: dict mapping from edge to it's label
    :param edges_to_roads_dict: dict mapping from tuple of ints to the road linestring
    :param nodes_geo_locs: dict mapping from node to it's geo location in degrees lat,lon
    :param patient_effective_polygon: effective polygon
    """
    index = len(nodes_geo_locs)
    new_nodes_labels = nodes_labels.copy()
    for node, node_label in nodes_labels.items():
        if nodes_labels[node] == 'danger_marked':
            edges_ending_at_node = get_edges_ending_at_node(node, edges_labels)
            current_labels = [edges_labels[edge] for edge in edges_ending_at_node]
            # if there is even one filtered edge, keep the block at the same point
            if 'filtered' in current_labels:
                continue

            # else all edges are dangerous (have not been filtered before), so the blockage must be kept but can be
            # moved inward, to the intersection with the polygon itself
            for edge in edges_ending_at_node:
                road = get_road_by_edge_key(edge, edges_to_roads_dict)
                intersection = road.intersection(patient_effective_polygon.boundary)
                if type(intersection) == MultiPoint:
                    intersection = intersection[0]
                nodes_geo_locs[index] = np.array(intersection).reshape(2)
                new_nodes_labels[index] = node_label
                index += 1
            nodes_geo_locs.pop(node, None)
            new_nodes_labels.pop(node, None)
    return new_nodes_labels
