from collections import defaultdict
from typing import Dict, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import math
from shapely.geometry import MultiLineString

from algo_config.algo_config import AlgorithmConfig
from general_utils.geometric_utils import create_buffered_polygon_around_coord
from general_utils.math_utils import normalize_vector, angle_between
import networkx as nx

NORMAL_TO_CENTER_VECTOR_DEG = 45


def calculate_front_points(patient_loc: np.ndarray, villages_boundary: MultiLineString) -> np.ndarray:
    """
    Taking first and last point in an edge of the villages boundary if the edge's
    normal creates a degree of -NORMAL_TO_CENTER_VECTOR_DEG to NORMAL_TO_CENTER_VECTOR_DEG
    with the line that connects the center of the edge to the patient
    :param patient_loc: location of the patient in lat,lon
    :param villages_boundary: villages boundary
    :return: all front points (points on edges directed towards the patient location)
    """
    front_points = []
    for edge in villages_boundary:
        edge_points = np.array(edge)
        for point1, point2 in zip(edge_points[:-1], edge_points[1:]):
            edge_vec = point2 - point1
            edge_mid = (point2 + point1) / 2
            normal_vec = normalize_vector(np.array([-edge_vec[1], edge_vec[0]]))
            mid_edge_to_patient_vec = normalize_vector(patient_loc - edge_mid)
            phi = math.degrees(angle_between(normal_vec, mid_edge_to_patient_vec))
            if phi < NORMAL_TO_CENTER_VECTOR_DEG:
                concat_points = np.concatenate([point1, point2]).reshape(2, 2)
                front_points.extend(concat_points)

    front_points = np.array(front_points)
    return front_points


def cluster_front_points(front_points: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Cluster with Agglomerative by distance (bottom-up), stop merging cluster if above the distance threshold
    :param front_points: array of all 2-d points
    :return: a dict mapping cluster label to the relevant cluster points
    """
    clusters_labels_dict = defaultdict(np.ndarray)
    clustering = AgglomerativeClustering(n_clusters=None,
                                         distance_threshold=float(AlgorithmConfig().get_value(
                                             'front_points_distance_threshold')),
                                         compute_full_tree=True).fit(front_points)

    labels = clustering.labels_
    for label in np.unique(labels):
        clusters_labels_dict[label] = front_points[labels == label]

    return clusters_labels_dict


def calculate_clusters_centers(clusters_labels_dict: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    Calculates the clusters centers
    :param clusters_labels_dict: a dict mapping cluster label to the relevant cluster points
    :return: a dict mapping cluster label to the relevant cluster center
    """
    clusters_centers = {}
    for label, points in clusters_labels_dict.items():
        cluster_center = points.mean(0)
        clusters_centers[label] = cluster_center
    return clusters_centers


def create_centers_graph(clusters_centers: Dict[int, np.ndarray]) -> nx.Graph:
    """
    Create a graph with clusters centers as nodes, and edges exist if the intersection between a herald placed at the
    centers is nonzero. No weight is used.
    :param clusters_centers: a dict mapping cluster label to the relevant cluster center
    :return: a clusters centers graph
    """
    noise_herald_effective_radius = AlgorithmConfig().get_value('noise_herald_effective_radius')
    n_clusters = len(clusters_centers.keys())
    A = np.zeros([n_clusters, n_clusters])
    for i, (label, cluster_center) in enumerate(clusters_centers.items()):
        circle = create_buffered_polygon_around_coord(cluster_center, noise_herald_effective_radius)
        for j, (label2, cluster_center2) in enumerate(clusters_centers.items()):
            circle2 = create_buffered_polygon_around_coord(cluster_center2, noise_herald_effective_radius)
            A[i, j] = circle.intersects(circle2)

    G = nx.from_numpy_matrix(A)
    return G


def calculate_min_set_cover(clusters_centers: Dict[int, np.ndarray], G: nx.Graph) -> List[int]:
    """
    Calculates the min set cover -> minimum set of nodes that cover all other nodes.
    A chosen node covers all it's neighbors.
    :param clusters_centers: a dict mapping cluster label to the relevant cluster center
    :param G: a clusters centers graph
    :return: clusters labels whose centers are included in the min set cover
    """
    set_cover = []
    # greedy algo_config for set cover
    while len(G.nodes) > 1:
        degrees = np.array(G.degree(G.nodes))[:, 1]
        max_degree_ind = np.argmax(degrees)
        max_degree_node = list(G.nodes)[max_degree_ind]
        set_cover.append(max_degree_node)
        G.remove_nodes_from(list(G.neighbors(max_degree_node)))

    if len(G.nodes) == 1:
        set_cover.append(list(G.nodes)[0])

    min_set_labels = [list(clusters_centers.keys()).index(cluster_ind) for cluster_ind in set_cover]
    return min_set_labels
