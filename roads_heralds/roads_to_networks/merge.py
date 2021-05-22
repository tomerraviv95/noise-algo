from typing import List, Tuple

from sklearn.cluster import AgglomerativeClustering
import numpy as np

from algo_config.algo_config import AlgorithmConfig


def merge_junctions(roads: List[List[Tuple]]) -> List[List[Tuple]]:
    """
    Merge close by end-nodes using Agglomerative clustering - the clusters stop merging when their centers surpass some distance threshold
    :param roads: list of lists with points
    :return: list of lists with points, with nearby points replaced with the same point
    """
    # merge only the endpoints of the roads
    endpoints = []
    for road in roads:
        endpoints.append(road[0])
        endpoints.append(road[-1])
    endpoints = np.array(endpoints)
    clustering = AgglomerativeClustering(n_clusters=None,
                                         distance_threshold=float(AlgorithmConfig().get_value(
                                             'intersection_points_distance_threshold')),
                                         compute_full_tree=True).fit(endpoints)

    labels = clustering.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_count = 0

    for label, count in zip(unique_labels, counts):
        if count > 1:
            class_indices = np.where(label == labels)[0]
            endpoints[class_indices, :] = np.tile(endpoints[class_indices[0]], (len(class_indices), 1))
            total_count += count - 1

    # apply the replacement
    for i in range(len(endpoints)):
        # first index is for the road index.
        # Since we stacked the endpoints in following manner: start1,end1,start2,end2,...
        # we access the matching endpoint by using the i//2.
        # second index is for the start/end. See that when i is even then we replace the start of the road,
        # and when i is odd we replace the end of the road
        roads[i // 2][-(i % 2)] = list(endpoints[i])

    return roads
