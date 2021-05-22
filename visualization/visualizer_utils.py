from shapely.geometry import LineString

from algo_config.algo_config import AlgorithmConfig
from roads_heralds.roads_to_networks.roads_utils import get_road_by_edge_key
from general_utils.geometric_utils import create_buffered_polygon_around_coord
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

NODES_COLORS_DICT = {'irrelevant': 'black', 'regular': 'blue', 'danger_marked': 'orange', 'safe_marked': 'orange'}
SIZES_DICT = {'irrelevant': 1, 'regular': 5, 'danger_marked': 40, 'safe_marked': 40}
EDGES_COLORS_DICT = {'irrelevant': 'gray', 'filtered': 'gray', 'contained': 'gray', 'danger_crossing': 'orange',
                     'safe_crossing': 'orange'}


def plot_labeled_edges(edges_labels: Dict[Tuple, str], edges_to_roads_dict: Dict[Tuple[int, int], LineString]):
    for edge in edges_labels.keys():
        if edges_labels[edge] in EDGES_COLORS_DICT.keys():
            road = np.array(get_road_by_edge_key(edge, edges_to_roads_dict))
            plt.plot(road[:, 1], road[:, 0],
                     color=EDGES_COLORS_DICT[edges_labels[edge]],
                     zorder=1,
                     alpha=0.35)


def plot_labeled_nodes(nodes_geo_locs: Dict[int, Tuple[float, float]], nodes_labels: Dict[int, str]):
    for node in nodes_labels.keys():
        plt.scatter(nodes_geo_locs[node][1], nodes_geo_locs[node][0],
                    color=NODES_COLORS_DICT[nodes_labels[node]],
                    s=SIZES_DICT[nodes_labels[node]],
                    zorder=12)


def plot_filtered_centers(filtered_clusters_centers: Dict[int, np.ndarray], patient_loc: np.ndarray):
    clusters_centers_array = np.array(list(filtered_clusters_centers.values()))
    clusters_priorities = np.argsort(np.linalg.norm(clusters_centers_array - patient_loc, axis=1))
    MAX_ALPHA = 0.5
    MIN_ALPHA = 0.2
    DELTA = (MAX_ALPHA - MIN_ALPHA) / (len(filtered_clusters_centers) - 1)
    for i, (label, cluster_center) in enumerate(filtered_clusters_centers.items()):
        priority = np.where(clusters_priorities == i)[0][0]
        alpha = MAX_ALPHA - priority * DELTA
        plt.scatter(cluster_center[1], cluster_center[0], s=100, zorder=5, c='lime', alpha=alpha)
        circle = create_buffered_polygon_around_coord(cluster_center,
                                                      AlgorithmConfig().get_value(
                                                          'noise_herald_effective_radius') * alpha)
        plt.fill(circle.exterior.xy[1], circle.exterior.xy[0], c="lime", alpha=alpha, zorder=5)
        full_circle = create_buffered_polygon_around_coord(cluster_center,
                                                           AlgorithmConfig().get_value(
                                                               'noise_herald_effective_radius'))
        plt.plot(full_circle.exterior.xy[1], full_circle.exterior.xy[0], alpha=1.0, zorder=5, linewidth=3, c='lime')
