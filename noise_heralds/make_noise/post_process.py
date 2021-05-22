from models.scenario import Scenario
from general_utils.geographic_utils import destination_coord_from_start_coord_and_angle, calculate_bearing
from general_utils.geometric_utils import create_buffered_polygon_around_coord
from typing import Dict
import numpy as np
from shapely.geometry import Point

CENTERS_MOVEMENT_RATIO = 0.1


def move_clusters_centers_towards_patient(clusters_centers: Dict[int, np.ndarray],
                                          patient_loc: np.ndarray) -> None:
    """
    Move the cluster center towards the patient
    :param clusters_centers: a dict mapping cluster label to the relevant cluster center
    :param patient_loc: location of the patient in lat,lon
    :return: none, modifies the dict inplace
    """
    for label, cluster_center in clusters_centers.items():
        clusters_centers[label] = cluster_center + CENTERS_MOVEMENT_RATIO * (patient_loc - cluster_center)


def push_centers_out_of_contagion_polygon(clusters_centers: Dict[int, np.ndarray], scenario: Scenario):
    """
    Push centers out of contagion polygon of the patient
    :param clusters_centers: a dict mapping cluster label to the relevant cluster center
    :param scenario: the scenario object
    :return: none, modifies the dict inplace
    """
    patient_effective_polygon = create_buffered_polygon_around_coord(scenario.patient.location,
                                                                     scenario.patient.contagion_radius)
    for label, cluster_center in clusters_centers.items():
        if Point(cluster_center).intersects(patient_effective_polygon):
            start_in_rad = np.deg2rad(scenario.patient.location)
            angle_in_radians = calculate_bearing(scenario.patient.location, cluster_center)
            moved_center = destination_coord_from_start_coord_and_angle(start_in_rad,
                                                                        angle_in_radians,
                                                                        scenario.patient.contagion_radius)
            clusters_centers[label] = np.rad2deg(moved_center).reshape(-1)
