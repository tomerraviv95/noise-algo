import os
from typing import List, Tuple

import numpy as np

from algo_config.algo_config import AlgorithmConfig
from dir_definitions import RESOURCES_DIR
from models.bounding_box import BoundingBox
from models.herald import Herald
from models.patient import Patient
from general_utils.geo_data_retriever import load_bounds, load_geographic_data


class Scenario:
    def __init__(self, heralds: List[Herald] = None):
        self._config = AlgorithmConfig().get_config()
        self.bbox = self.load_bbox()
        self.buildings = self.load_buildings()
        self.roads = self.load_roads()
        self.patient = self.load_patient()
        self.heralds = heralds if heralds is not None else []

    def load_buildings(self) -> List[List[Tuple[float, float]]]:
        buildings_file = os.path.join(RESOURCES_DIR, f'buildings/buildings_{self._config["area_name"]}.json')
        return load_geographic_data(buildings_file)

    def load_bbox(self) -> BoundingBox:
        bounds_file = os.path.join(RESOURCES_DIR, f'bounds/bounds_{self._config["area_name"]}.json')
        return load_bounds(bounds_file)

    def load_roads(self) -> List[List[Tuple[float, float]]]:
        roads_file = os.path.join(RESOURCES_DIR, f'roads/roads_{self._config["area_name"]}.json')
        return load_geographic_data(roads_file)

    def load_patient(self) -> Patient:
        if self._config['patient_location_type'] == 'relative':
            location = np.array([self.bbox.south * self._config['patient_location_south'] + self.bbox.north * (
                    1 - self._config['patient_location_south']),
                                 self.bbox.west * self._config['patient_location_west'] + self.bbox.east * (
                                         1 - self._config['patient_location_west'])])
        else:
            location = np.array([self._config['patient_location_south'], self._config['patient_location_west']])
        return Patient(location, self._config['patient_contagion_radius'], self._config['patient_effective_radius'])
