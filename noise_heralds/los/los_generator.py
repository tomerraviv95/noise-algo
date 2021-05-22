import os
import pickle
from typing import Tuple, List

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from algo_config.algo_config import AlgorithmConfig
from dir_definitions import LOS_DIR, BENCHMARK_DIR
from models.bounding_box import BoundingBox
from models.scenario import Scenario
from noise_heralds.los.dtm_loader import get_elevation
from noise_heralds.los.los_utils import find_los


def evaluate_grid_cells_centers(bounds: BoundingBox, grid_size: int) -> np.ndarray:
    """Returns the centers of all grid cells
    :param bounds: scenario bbox
    :param grid_size: resolution of the output grid
    :return: 2d array of all grid centers
    """
    x = np.linspace(bounds.south, bounds.north, grid_size + 1)
    y = np.linspace(bounds.west, bounds.east, grid_size + 1)
    x_diff = x[1] - x[0]
    y_diff = y[1] - y[0]
    xx, yy = np.meshgrid(x[:-1] + x_diff / 2, y[:-1] + y_diff / 2)
    return np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))


def generate_elevation_grid(bounds: BoundingBox, grid_size: int) -> np.ndarray:
    """Creates an elevation grid
    :param bounds: scenario bbox
    :param grid_size: resolution of the output grid
    :return: grid which contains elevation in meters
    """
    grid_points = evaluate_grid_cells_centers(bounds, grid_size)
    points = grid_points.reshape(-1, 2)
    elevation = get_elevation(points).reshape(grid_size, grid_size).T
    return elevation


def transform_coords_geo_to_grid(grid_size: int, bounds: BoundingBox, coords: np.ndarray) -> np.ndarray:
    """Transform coordinates from geo coordinate system to heatmap coordinate system
    :param bounds: scenario bbox
    :param grid_size: resolution of the output grid
    :param coords: 2D array of geo points
    :return: 2D array of points in grid coordinate system
    """
    coords = coords.reshape(-1, 2)
    lat_diff = bounds.east - bounds.west
    lon_diff = bounds.north - bounds.south
    x = grid_size - ((coords[:, 0] - bounds.south) / lon_diff) * grid_size
    y = ((coords[:, 1] - bounds.west) / lat_diff) * grid_size
    return np.vstack((x, y)).T


def generate_los_grid(patient_height: int, above_surface_height: int):
    """Creates a line-of-sight grid
    :param bounds: scenario bbox
    :param grid_size: resolution of the output grid
    :param patient_height: height of patient agl (in meters)
    :param above_surface_height: height agl from which we check if there's a LOS to the patient (in meters)
    :return: binary grid which states whether there's a LOS to the patient
    """
    grid_size = AlgorithmConfig().get_value('grid_size')
    scenario = Scenario(heralds=None)
    bounds = scenario.bbox
    patient_loc = scenario.patient.location
    elevations = generate_elevation_grid(bounds, grid_size)
    patient_coord_in_grid = transform_coords_geo_to_grid(grid_size, bounds, patient_loc).reshape(2).astype(int)
    patient_elevation = elevations[patient_coord_in_grid[0], patient_coord_in_grid[1]]
    patient_xyz = np.hstack((patient_coord_in_grid, patient_elevation))
    vis = find_los(elevations, patient_xyz, patient_height, above_surface_height)
    return vis


def binary_grid_to_multipolygons(vis: np.ndarray) -> Tuple[MultiPolygon, MultiPolygon]:
    """Creates multipolygons which represents true and false valued polygons of a binary grid
    :param bounds: scenario bbox
    :param grid_size: resolution of the output grid
    :param vis: binary grid which states whether there's a LOS to the patient
    :return: a tuple which contains false polygons and true polygons
    """
    grid_size = AlgorithmConfig().get_value('grid_size')
    scenario = Scenario(heralds=None)
    bounds = scenario.bbox
    cells = evaluate_grid_cells_centers(bounds, grid_size)
    polygons: List[List[Polygon]] = [[], []]

    for i in range(vis.shape[0] - 1):
        for j in range(vis.shape[1] - 1):
            cell = np.array([[i * vis.shape[0] + j],
                             [(i + 1) * vis.shape[0] + j],
                             [(i + 1) * vis.shape[0] + j + 1],
                             [i * vis.shape[0] + j + 1]])
            cell_geo = cells[cell].reshape(-1, 2)
            polygons[vis[j, i]].append(Polygon(cell_geo))

    return MultiPolygon(polygons[0]).buffer(0), MultiPolygon(polygons[1]).buffer(0)


def save_los(mp: Tuple[MultiPolygon, MultiPolygon]) -> None:
    """Saves true and false multipolygons to a file
    :param mp: tuple of multipolygons representing the LOS
    """
    if not os.path.isdir(LOS_DIR):
        os.mkdir(LOS_DIR)

    with open(os.path.join(LOS_DIR, f'los_mp_{AlgorithmConfig().get_name()}.pkl'), 'wb') as f:
        pickle.dump(mp, f)


def create_los():
    """
    Creates the LOS if no such file exists
    """
    los = generate_los_grid(50, 5)
    mp = binary_grid_to_multipolygons(los)
    save_los(mp)


def get_los() -> Tuple[MultiPolygon, MultiPolygon]:
    """Reads true and false multipolygons from a file
    :return: tuple of multipolygons representing the LOS
    """
    los_path = os.path.join(LOS_DIR, f'los_mp_{AlgorithmConfig().get_name()}.pkl')
    if not os.path.isfile(los_path):
        create_los()

    with open(los_path, 'rb') as f:
        los: Tuple[MultiPolygon, MultiPolygon] = pickle.load(f)
    return los


if __name__ == '__main__':
    for file in os.listdir(BENCHMARK_DIR):
        AlgorithmConfig().load_config(os.path.join(BENCHMARK_DIR, file))
        create_los()
