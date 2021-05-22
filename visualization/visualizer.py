import os
from typing import Optional, Dict

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from dir_definitions import FIGURES_DIR
from models.scenario import Scenario
from noise_heralds.segment import SegmentedAreas
from visualization.visualizer_utils import plot_filtered_centers, plot_labeled_edges, plot_labeled_nodes

plt.style.use('dark_background')


def draw(scenario: Scenario,
         seg_areas: SegmentedAreas = None,
         patient_contagion_polygon: Polygon = None,
         patient_effective_polygon: Polygon = None,
         graph: Optional[nx.Graph] = None,
         blocks_output: Optional[Dict] = None,
         noise_output: Optional[Dict] = None,
         to_file: Optional[str] = None):
    """Draws all the objects in pyplot"""

    if seg_areas is not None:
        for p in seg_areas.los:
            plt.fill(p.exterior.xy[1], p.exterior.xy[0], c="orangered", alpha=0.2)

        for p in seg_areas.villages:
            plt.fill(p.exterior.xy[1], p.exterior.xy[0], c="grey", alpha=0.05)

    if scenario.buildings is not None:
        for b in scenario.buildings:
            c = np.array(b)
            plt.plot(c[:, 1], c[:, 0], color='blue', alpha=0.45)

    if scenario.roads is not None:
        for c in scenario.roads:
            c = np.array(c)
            plt.plot(c[:, 1], c[:, 0], c='gray', alpha=0.35, zorder=3, linewidth=2)

    if scenario.patient is not None:
        plt.scatter(scenario.patient.location[1], scenario.patient.location[0], marker='+', c='red', zorder=10)

    if patient_contagion_polygon is not None:
        x, y = patient_contagion_polygon.exterior.coords.xy
        patient_exterior_points = np.concatenate([np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)],
                                                 axis=1)
        plt.plot(patient_exterior_points[:, 1], patient_exterior_points[:, 0],
                 c='red', alpha=0.6, linewidth=3, zorder=11)

    if patient_effective_polygon is not None:
        x, y = patient_effective_polygon.exterior.coords.xy
        patient_interior_points = np.concatenate([np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)], axis=1)
        plt.plot(patient_interior_points[:, 1], patient_interior_points[:, 0], c='magenta', alpha=0.4,
                 linewidth=3, zorder=11)

    if scenario.bbox is not None:
        plt.xlim(scenario.bbox.west, scenario.bbox.east)
        plt.ylim(scenario.bbox.south, scenario.bbox.north)
        p = scenario.bbox.get_end_of_map_polygon()
        plt.plot(p.interiors[0].xy[1], p.interiors[0].xy[0], c="yellow", alpha=0.25)

    if blocks_output is not None:
        nodes_geo_locs = blocks_output["nodes_geo_locs"]
        nodes_labels = blocks_output["nodes_labels"]
        plot_labeled_nodes(nodes_geo_locs, nodes_labels)

        edges_to_roads_dict = blocks_output["edges_to_roads_dict"]
        edges_labels = blocks_output["edges_labels"]
        plot_labeled_edges(edges_labels, edges_to_roads_dict)

    if noise_output is not None:
        filtered_clusters_centers = noise_output["filtered_clusters_centers"]
        plot_filtered_centers(filtered_clusters_centers,scenario.patient.location)

    if graph is not None:
        geo_locs = nx.get_node_attributes(graph, 'geo_locs')
        geo_locs = np.array(list(geo_locs.values()))
        plt.scatter(geo_locs[:, 1], geo_locs[:, 0], c='red', s=0.5, zorder=5)

        roads = nx.get_edge_attributes(graph, 'roads_heralds')
        for road in roads.values():
            plt.plot(np.array(road)[:, 1], np.array(road)[:, 0], alpha=0.8, zorder=4)

    if to_file is not None:
        total_path = os.path.join(FIGURES_DIR, to_file + '.png')
        parent_path = os.path.abspath(os.path.join(total_path, os.pardir))
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
        plt.savefig(total_path)
        plt.close()
    else:
        plt.show()
