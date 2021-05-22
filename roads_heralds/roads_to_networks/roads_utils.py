import numpy as np
from shapely.geometry import LineString
from collections import defaultdict
from typing import Dict, Tuple, List
import networkx as nx


def get_roads_linestrings_dict(roads: List[List[Tuple]]) -> Dict[Tuple, LineString]:
    """
    Format the roads from list structure to a dict structure, with their centers as keys
    :param roads: list of lists with points
    :return: dict mapping from road center to the actual road as a linestring
    """
    roads_linestrings = {}
    for i, road in enumerate(roads):
        current_linestring = LineString(road)
        roads_linestrings[(current_linestring.centroid.coords.xy[0][0],
                           current_linestring.centroid.coords.xy[1][0])] = current_linestring
    return roads_linestrings


def extract_start_end_points_of_linestring(linestring: LineString) -> Tuple[Tuple, Tuple]:
    """
    Returns the first and last points of the intersecting linestring. Those are the nodes.
    :param linestring: the linestring of the road, sequence of points
    :return: first,last points
    """
    start = (linestring.coords.xy[0][0], linestring.coords.xy[1][0])
    end = (linestring.coords.xy[0][-1], linestring.coords.xy[1][-1])
    return start, end


def gather_intersecting_nodes(intersected_linestrings: List[LineString]) -> List[Tuple]:
    """
    Extract all first and last points in the intersected linestring into a list
    :param intersected_linestrings: list of intersecting roads
    :return: list of intersection points
    """
    intersected_nodes = []
    for intersected_linestring in intersected_linestrings:
        start, end = extract_start_end_points_of_linestring(intersected_linestring)
        intersected_nodes.append(start)
        intersected_nodes.append(end)
    return intersected_nodes


def create_edges_to_roads_dict(nodes_to_id_dict: Dict[Tuple, int], intersected_linestrings: List[LineString]) -> \
        Dict[Tuple, LineString]:
    """
    Creates dict from the edges (tuples of ints, each int is a node) to the underlying road, that is a linestring with multiple nodes
    :param nodes_to_id_dict: dict mapping from the node location in geo to it's id (an int) in the graph, numbered from 0 to len(nodes)-1
    :param intersected_linestrings: list of intersecting roads
    :return: the mentioned dict
    """
    edges_to_roads_dict = defaultdict(LineString)
    for linestring in intersected_linestrings:
        start, end = extract_start_end_points_of_linestring(linestring)
        start_id = nodes_to_id_dict[start]
        end_id = nodes_to_id_dict[end]
        edges_to_roads_dict[(start_id, end_id)] = linestring

    return edges_to_roads_dict


def set_geo_locs(G: nx.Graph, nodes_to_id_dict: Dict[Tuple, int]):
    nx.set_node_attributes(G, {id: node for node, id in nodes_to_id_dict.items()}, 'geo_locs')


def set_edges_grid_locs(G: nx.Graph, edges_to_roads_dict: Dict[Tuple, LineString]):
    nx.set_edge_attributes(G, edges_to_roads_dict, 'roads')


def get_geo_locs(G: nx.Graph) -> Dict:
    geo_locs = list(nx.get_node_attributes(G, 'geo_locs').values())
    return {i: geo_locs[i] for i in range(len(geo_locs))}


def get_roads(G: nx.Graph) -> Dict:
    return nx.get_edge_attributes(G, 'roads')


def get_road_by_edge_key(edge: Tuple[int, int], edges_to_roads_dict: Dict[Tuple[int, int], LineString]):
    """
    Collect the road by the edge key. If the edge was ordered as inner and outer nodes, then we need to switch the order
    :param edge: tuple of ints
    :param edges_to_roads_dict: dict from the edges (tuples of ints, each int is a node) to the underlying road,
           that is a linestring with multiple nodes
    :return: the corresponding road
    """
    try:
        road = edges_to_roads_dict[edge]
    except KeyError:
        road = edges_to_roads_dict[(edge[1], edge[0])]
    return road
