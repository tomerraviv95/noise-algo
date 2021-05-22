import json
import os
from enum import Enum
from pathlib import Path
from typing import Tuple, Dict, List

import requests

from dir_definitions import RESOURCES_DIR
from models.bounding_box import BoundingBox


class GeographicType(Enum):
    BUILDING = 1
    ROAD = 2

    def get_overpass_entity_name(self) -> str:
        if self.name == "BUILDING":
            return "building"
        elif self.name == "ROAD":
            return "highway"
        return ""

    def get_entities_name(self) -> str:
        return f"{self.name.lower()}s"


def retrieve_data_from_server(data_type: GeographicType, bounds: BoundingBox) -> Dict:
    """Retrieves the data of the given type which is located inside the bounding box

    :param data_type: type of the requested geographic entity
    :param bounds: bounding box of the requested area
    :return: all geographic entities of the given type in the area
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:25];
    (
      way["{data_type.get_overpass_entity_name()}"]{bounds.as_tuple()};
      relation["{data_type.get_overpass_entity_name()}"]{bounds.as_tuple()};
    );
    out geom;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    return response.json()


def data_to_coords_collection(data: Dict) -> List[List[Tuple[float, float]]]:
    """Creates an xy coordinates collection of all the entities in data retrieved from overpass-api

    :param data: retrieved data
    :return: coordinates collection of all the entities in the data
    """
    coords = []

    for element in data['elements']:
        if element['type'] == 'way':
            coords.append(geometry_to_coords(element['geometry']))
        elif element['type'] == 'relation':
            # a relation is a multipolygon and therefore has multiple geometries
            coords += [geometry_to_coords(member['geometry']) for member in element['members']]

    return coords


def geometry_to_coords(geometry: List[Dict[str, float]]) -> List[Tuple[float, float]]:
    """Transforms coordinates if the following way:
    from geometry format: [{'lat': lat0, 'lon': lon0}, {'lat': lat1, 'lon': lon1}, ...]
    to xy format: [(lat0, lon0), (lat1, lon1), ...]

    :param geometry: the given coordinates of the geometry
    :return: the formatted coordinates
    """
    return [(g['lat'], g['lon']) for g in geometry]


def geometry_to_xy_coords(geometry: List[Dict[str, float]]) -> List[Tuple]:
    """Transforms coordinates if the following way:
    from geometry format: [{'lat': lat0, 'lon': lon0}, {'lat': lat1, 'lon': lon1}, ...]
    to xy format: [[lat0, lat1, ...], [lon0, lon1, ...]]

    :param geometry: the given coordinates of the geometry
    :return: the formatted coordinates
    """
    return list(zip(*[(g['lat'], g['lon']) for g in geometry]))


def save_geographic_data(data: List[List[Tuple]], area_name: str, geo_type: GeographicType) -> None:
    """Saves geographic data to json file

    :param data: geographic data in format list of polygons/lines (which are represented by list of coordinates)
    :param area_name: area name
    :param geo_type: geographic type of data to save
    :return: none
    """
    dir_path = os.path.join(RESOURCES_DIR, geo_type.get_entities_name())
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    file_path = os.path.join(dir_path, f'{geo_type.get_entities_name()}_{area_name}.json')
    with open(file_path, "w") as f:
        json.dump(data, f)


def load_geographic_data(filename: str) -> List[List[Tuple]]:
    """Loads geographic data from json file

    :param filename: input file path
    :return: geographic data in format list of polygons/lines (which are represented by list of coordinates)
    """
    with open(filename, "r") as f:
        return json.load(f)


def load_bounds(filename: str) -> BoundingBox:
    """Loads bounds from config file

    :param filename: input file path
    :return: bounding box
    """
    if not os.path.exists(filename):
        raise FileNotFoundError("Bounds file doesn't exist")

    with open(filename) as f:
        bounds = json.load(f)

    return BoundingBox(**bounds)


def retrieve_and_save_geo_data(area_name: str) -> None:
    """Retrieves and saves buildings and roads

    :param area_name: requested area name
    :return: none
    """
    bounds_file = os.path.join(RESOURCES_DIR, f'bounds/bounds_{area_name}.json')
    bbox = load_bounds(bounds_file)

    print("Retrieving buildings")
    buildings_data = retrieve_data_from_server(GeographicType.BUILDING, bbox)

    print("Retrieving roads")
    roads_data = retrieve_data_from_server(GeographicType.ROAD, bbox)

    print("Parsing data")
    buildings = data_to_coords_collection(buildings_data)
    roads = data_to_coords_collection(roads_data)

    print("Saving data")
    save_geographic_data(buildings, area_name, GeographicType.BUILDING)
    save_geographic_data(roads, area_name, GeographicType.ROAD)


if __name__ == "__main__":
    areas_names = [str(i) for i in range(1, 11)]
    for area_name in areas_names:
        retrieve_and_save_geo_data(area_name)
