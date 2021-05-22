import os
import pickle
from typing import List, BinaryIO

import networkx as nx
import numpy as np
from networkx.utils import UnionFind
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import cascaded_union

from algo_config.algo_config import AlgorithmConfig
from dir_definitions import BENCHMARK_DIR, VILLAGES_DIR
from models.scenario import Scenario
from general_utils.geographic_utils import circle_around_point
from general_utils.geometric_utils import merge_circles, circle_points


def merge_multiple_circles(centers: np.ndarray, radius: float) -> MultiPolygon:
    """Given a set of circles with the same radius, creates a multipolygon which represents the merged circles

    :param centers: circles' centers
    :param radius: circles' radius
    :return: multipolygon representing the merged circles
    """
    dist = cdist(centers, centers)
    g = nx.Graph()
    d_max = 2 * radius
    g.add_nodes_from(range(len(centers)))
    for i in range(len(centers)):
        for j in range(i):
            if dist[i, j] <= d_max:
                g.add_edge(i, j, weight=dist[i, j])

    mst = nx.minimum_spanning_edges(g)
    edges = list(mst)

    uf = UnionFind()

    polygons = [Polygon(circle_points(coord, radius)) for coord in centers]
    for edge in edges:
        src0 = uf[edge[0]]
        src1 = uf[edge[1]]
        uf.union(edge[0], edge[1])
        new_src = uf[edge[0]]

        poly = merge_circles(centers[edge[0]], radius, centers[edge[1]], radius)
        union_poly = cascaded_union([Polygon(poly), polygons[src0], polygons[src1]])
        polygons[new_src] = union_poly

    return MultiPolygon([polygons[uf[s.pop()]] for s in uf.to_sets()])


def save_villages_outline(mp: MultiPolygon, buildings: List[Point]):
    """Saves villages outlines to a file

    :param mp: villages outlines as a multipolygon
    :param buildings: buildings centers
    :return:
    """
    if not os.path.isdir(VILLAGES_DIR):
        os.mkdir(VILLAGES_DIR)

    villages = [{"count": len([bldg for bldg in buildings if polygon.contains(bldg)]),
                 "polygon": polygon}
                for polygon in mp]
    with open(os.path.join(VILLAGES_DIR, f'villages_{AlgorithmConfig().get_name()}.pickle'), 'wb') as f:
        pickle.dump(villages, f)


def _read_villages_outline(file: BinaryIO, filter_count=None) -> MultiPolygon:
    """Reads villages outlines from a file

    :param file: villages file
    :param filter_count: filter out villages which less buildings than this number
    :return: villages outlines as a multipolygon
    """
    data = pickle.load(file)
    if filter_count is None:
        mp = MultiPolygon([p["polygon"] for p in data])
    else:
        mp = MultiPolygon([p["polygon"] for p in data if p["count"] > filter_count])
    return mp


def get_villages_outline() -> MultiPolygon:
    """Reads villages outlines from benchmark file
    :return: villages outlines as a multipolygon
    """
    villages_outline_path = os.path.join(VILLAGES_DIR, f'villages_{AlgorithmConfig().get_name()}.pickle')
    if not os.path.isfile(villages_outline_path):
        create_villages_outline()

    with open(villages_outline_path, 'rb') as f:
        return _read_villages_outline(f, filter_count=AlgorithmConfig().get_value('village_min_buildings'))


def create_villages_outline():
    scenario = Scenario(heralds=None)
    buildings_centers = [Polygon(building).centroid for building in scenario.buildings if len(building) > 2]
    buildings_centers_xy = [c.xy for c in buildings_centers]
    b = np.array(buildings_centers_xy).reshape(-1, 2)
    circle_p = circle_around_point(np.deg2rad(b[0]), AlgorithmConfig().get_value('villages_radius')
                                   , AlgorithmConfig().get_value('villages_resolution'))
    d = np.linalg.norm(np.rad2deg(circle_p[0]) - b[0])

    mp = merge_multiple_circles(b, d)
    save_villages_outline(mp, buildings_centers)


if __name__ == "__main__":

    for file in os.listdir(BENCHMARK_DIR):
        benchmark_num = file.split('.')[0].split('_')[-1]

        AlgorithmConfig().load_config(os.path.join(BENCHMARK_DIR, file))
        create_villages_outline()
