import os

import networkx as nx

from dir_definitions import ROADS_NETWORKS_DIR
from general_utils.patient_utils import get_patient_filtered_polygons
from models.scenario import Scenario
from roads_heralds.roads_to_networks.intersect import calculate_intersecting_linestrings
from roads_heralds.roads_to_networks.merge import merge_junctions
from roads_heralds.roads_to_networks.roads_utils import set_geo_locs, set_edges_grid_locs, \
    create_edges_to_roads_dict, \
    gather_intersecting_nodes, get_roads_linestrings_dict
from visualization.visualizer import draw
from algo_config.algo_config import AlgorithmConfig


def generate_roads_network():
    scenario = Scenario(heralds=None)

    # hotfix - get merge close intersections at ends or starts of roads_heralds
    roads = merge_junctions(scenario.roads)

    # get a dict with roads_heralds' centers as keys, and the linestrings of the roads_heralds as values
    roads_linestrings = get_roads_linestrings_dict(roads)

    # calculate all segments, parts of the above linestrings splitted at the intersections
    intersected_linestrings = calculate_intersecting_linestrings(roads_linestrings)

    # gather all nodes from above linestrings
    intersected_nodes = gather_intersecting_nodes(intersected_linestrings)

    # the unique nodes are the junctions
    nodes = list(set(intersected_nodes))

    # mapping dict, from node geo location to id
    nodes_to_id_dict = {node: i for i, node in enumerate(nodes)}

    # dict from the edges (tuples of nodes id) to the linestrings of the roads_heralds
    edges_to_roads_dict = create_edges_to_roads_dict(nodes_to_id_dict, intersected_linestrings)

    G = nx.Graph()

    # add all nodes and edges
    G.add_nodes_from(list(nodes_to_id_dict.values()))
    G.add_edges_from(list(edges_to_roads_dict.keys()))

    # set the geo location as attribute per node
    set_geo_locs(G, nodes_to_id_dict)

    # set the linestrings of the underlying roads_heralds as the attribute of an edge
    set_edges_grid_locs(G, edges_to_roads_dict)

    # save the graph
    save_path = os.path.join(ROADS_NETWORKS_DIR, f'roads_graph_{AlgorithmConfig().get_name()}.pkl')
    if not os.path.isdir(ROADS_NETWORKS_DIR):
        os.mkdir(ROADS_NETWORKS_DIR)
    nx.write_gpickle(G, save_path)

    # draw graph
    patient_contagion_polygon, patient_effective_polygon = get_patient_filtered_polygons(scenario)
    draw(scenario=scenario, graph=G,
         patient_contagion_polygon=patient_contagion_polygon,
         patient_effective_polygon=patient_effective_polygon,
         to_file=f'roads_only/{AlgorithmConfig().get_name()}')
