from shapely.geometry import MultiPolygon, Polygon

from general_utils.patient_utils import get_no_entrance_polygon
from roads_heralds.block_roads.assign_labels import assign_edges_labels, assign_nodes_labels
from roads_heralds.block_roads.fix_nodes_locs import update_nodes_plotting_locs
from roads_heralds.block_roads.post_process import post_process

from roads_heralds.roads_to_networks.roads_utils import get_geo_locs, get_roads

from models.scenario import Scenario
from general_utils.geometric_utils import point_in_multipolygon
import networkx as nx


def block_roads(G: nx.Graph, villages: MultiPolygon, patient_effective_polygon: Polygon):
    scenario = Scenario(heralds=None)
    no_entrance_polygon = get_no_entrance_polygon(scenario)

    # get the geo locations
    nodes_geo_locs = get_geo_locs(G)

    # get the linestrings of the underlying roads_heralds as the attribute of an edge
    edges_to_roads_dict = get_roads(G)

    # find for each intersection if it is contained in the patient effective polygon
    is_contained_in_patient_polygon = point_in_multipolygon(nodes_geo_locs, patient_effective_polygon)

    # calculate label per edge
    edges_labels = assign_edges_labels(edges_to_roads_dict,
                                       is_contained_in_patient_polygon,
                                       patient_effective_polygon,
                                       no_entrance_polygon)

    # post process the selected edges
    edges_labels = post_process(G, edges_labels,
                                edges_to_roads_dict,
                                scenario,
                                no_entrance_polygon,
                                villages,nodes_geo_locs)

    # for all post-processed filtered edges, assign the relevant node the label
    nodes_labels = assign_nodes_labels(edges_labels, is_contained_in_patient_polygon)

    # get the display location of nodes. if able - push node to the boundary of the effective polygon
    nodes_labels = update_nodes_plotting_locs(nodes_labels, edges_labels, edges_to_roads_dict, nodes_geo_locs,
                                              patient_effective_polygon)

    block_output = {"scenario": scenario,
                    "nodes_labels": nodes_labels,
                    "edges_labels": edges_labels,
                    "edges_to_roads_dict": edges_to_roads_dict,
                    "nodes_geo_locs": nodes_geo_locs}
    return block_output
