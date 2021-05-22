from typing import Dict, Tuple, List
from shapely.geometry import Polygon, LineString, MultiPolygon

from models.scenario import Scenario
import networkx as nx

from roads_heralds.roads_to_networks.roads_utils import get_road_by_edge_key


def edge_prevents_to_reach_patient(inner_subgraph: nx.Graph, inner_node: int, marked_nodes_set: List[int],
                                   edges_to_roads_dict: Dict[Tuple, LineString], no_entrance_polygon: Polygon) -> bool:
    """
    Checks if the edge is essential to prevent an outsider from reaching the patient via the roads networks
    :param inner_subgraph: the part of the network that is inside the effective polygon
    :param inner_node: the inner node of the crossing edge
    :param marked_nodes_set: all the marked nodes
    :param edges_to_roads_dict: dict from the edges (tuples of ints, each int is a node) to the underlying road
    :param no_entrance_polygon:
    :return: bool
    """
    # calculate all paths to the leaf nodes in the inner subgraph
    paths = []
    for node in inner_subgraph:
        if inner_subgraph.degree(node) == 1:  # it's a leaf
            paths.append(nx.shortest_path(inner_subgraph, inner_node, node))

    # if one of the paths intersects the no entrance polygon
    for path in paths:
        for edge in zip(path[:-1], path[1:]):
            road = get_road_by_edge_key(edge, edges_to_roads_dict)
            if road.intersects(no_entrance_polygon):
                return True
            if edge[1] in marked_nodes_set:
                break
    return False


def gather_marked_nodes(marked_edges: List[Tuple], edges_labels: Dict[Tuple, str]) -> Tuple[List, List]:
    """
    Gathers the marked nodes from currently marked edges
    :param marked_edges: all the marked edges
    :param edges_labels: dict mapping from edges to the labels
    :return: all marked nodes and the non marked nodes
    """
    marked_nodes = []
    non_marked_nodes = []
    for marked_edge in marked_edges:
        marked_node = marked_edge[0] if edges_labels[marked_edge] == 'safe_crossing' else marked_edge[1]
        non_marked_node = marked_edge[1] if edges_labels[marked_edge] == 'safe_crossing' else marked_edge[0]
        marked_nodes.append(marked_node)
        non_marked_nodes.append(non_marked_node)
    return marked_nodes, non_marked_nodes


def connected_to_village_or_end_of_map(outer_subgraph: nx.Graph, edges_to_roads_dict: Dict[Tuple, LineString],
                                       scenario: Scenario, villages: MultiPolygon) -> bool:
    """
    Checks if the edge is connected to a village or reaches the end of map
    :param outer_subgraph: the part of the network that is outside the effective polygon
    :param edges_to_roads_dict: dict from the edges (tuples of ints, each int is a node) to the underlying road, that is a linestring with multiple nodes
    :param scenario: current scenario
    :param villages: the villages
    :return: bool
    """

    end_of_map_polygon = scenario.bbox.get_end_of_map_polygon()
    overall_polygon = end_of_map_polygon.union(villages)

    for edge in outer_subgraph.edges():
        road = get_road_by_edge_key(edge, edges_to_roads_dict)
        if road.intersects(overall_polygon):
            return True
    return False


def post_process(G: nx.Graph, edges_labels: Dict[Tuple, str], edges_to_roads_dict: Dict[Tuple, LineString],
                 scenario: Scenario, no_entrance_polygon: Polygon, villages: MultiPolygon, nodes_geo_locs) -> Dict[
    Tuple, str]:
    """
    The post processing of the labeled edges. Filters crossing edges that:
    (1) do not prevent outsiders from reaching the entrance polygon
    (2) are not connected to villages or the edges of the map
    :param G: roads network
    :param edges_labels: dict mapping from edges to the labels
    :param edges_to_roads_dict: dict mapping from edges to the underlying roads
    :param scenario: Scenario object
    :param no_entrance_polygon: danger polygon
    :param villages: villages multipolygon
    :return: the updated edges_labels dict
    """
    marked_edges = [edge for edge, edge_label in edges_labels.items() if
                    edge_label in ['danger_crossing', 'safe_crossing']]

    marked_nodes, non_marked_nodes = gather_marked_nodes(marked_edges, edges_labels)
    marked_nodes_set = list(set(marked_nodes.copy()))

    for marked_edge in marked_edges:

        # remove marked edges
        G_copy = G.copy()
        G_copy.remove_edges_from(marked_edges)

        # find if inner component reaches the no entrance polygon close to patient
        inner_node = marked_edge[0]
        inner_connected_components = nx.node_connected_component(G_copy, inner_node)
        inner_subgraph = G_copy.subgraph(inner_connected_components)

        if not edge_prevents_to_reach_patient(inner_subgraph,
                                              inner_node,
                                              marked_nodes_set,
                                              edges_to_roads_dict,
                                              no_entrance_polygon):
            edges_labels[marked_edge] = 'filtered'
            continue

        # find if outer component reaches a village or the end of map
        outer_node = marked_edge[1]
        G_copy.remove_edges_from(inner_subgraph.edges())
        G_copy.add_edge(*marked_edge, roads=get_road_by_edge_key(marked_edge, edges_to_roads_dict))
        outer_connected_components = nx.node_connected_component(G_copy, outer_node)
        outer_subgraph = G_copy.subgraph(outer_connected_components)

        if not connected_to_village_or_end_of_map(outer_subgraph, edges_to_roads_dict, scenario, villages):
            edges_labels[marked_edge] = 'filtered'

    return edges_labels
