from typing import Tuple

from shapely.geometry import Polygon, MultiPolygon

from algo_config.algo_config import AlgorithmConfig
from general_utils.geometric_utils import create_buffered_polygon_around_coord
from models.scenario import Scenario
from noise_heralds.villages.village_outliner import get_villages_outline


def get_no_entrance_polygon(scenario: Scenario) -> Polygon:
    """
    Calculates the no entrance polygon
    :param scenario: Scenario object
    :return: no entrance polygon
    """
    no_entrance_polygon = create_buffered_polygon_around_coord(scenario.patient.location,
                                                               AlgorithmConfig().get_value('no_entrance_polygon_ratio')
                                                               * scenario.patient.effective_radius)
    return no_entrance_polygon


def get_patient_filtered_polygons(scenario: Scenario) -> Tuple[Polygon, Polygon]:
    """
    Filters the effective polygon of the patient with the villages polygons.
    :param scenario: current scenario
    :return: the filtered polygon
    """
    villages = get_villages_outline()
    patient_contagion_polygon = create_buffered_polygon_around_coord(scenario.patient.location,
                                                                     scenario.patient.contagion_radius)
    patient_effective_polygon = create_buffered_polygon_around_coord(scenario.patient.location,
                                                                     scenario.patient.effective_radius)

    # only remove from the patient's effective polygon those villages outside the polygon, touching it's boundary
    # if one wishes to consider the inner villages simply remove the whole villages multipolygons instead
    for village_polygon in villages:
        if not village_polygon.within(patient_effective_polygon):
            patient_effective_polygon = patient_effective_polygon.difference(village_polygon)

    if type(patient_effective_polygon) == MultiPolygon:
        patient_effective_polygon = max(patient_effective_polygon, key=lambda polygon: polygon.area)

    return patient_contagion_polygon, patient_effective_polygon
