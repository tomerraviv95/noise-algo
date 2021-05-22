import json
import math
import os
from typing import Optional, Dict

import numpy as np

from dir_definitions import DTM_DIR, BOUNDS_DIR

SAMPLES = 3601  # SRTM1


def get_elevation(coords: np.ndarray) -> np.ndarray:
    """Retrieves elevation for every coordinate

    :param coords: a 2D numpy array of coordinates
    :return: a numpy vector of elevations in meters
    """
    elevation = np.zeros(coords.shape[0])
    hgt = _load_dtm_from_files(coords)

    for i, coord in enumerate(coords):
        hgt_file = _get_file_name(*coord)
        if hgt_file:
            elevation[i] = _read_elevation_from_array(hgt[hgt_file], *coord)
        else:
            elevation[i] = None
    return elevation


def _load_dtm_from_files(coords: np.ndarray) -> Dict[str, np.ndarray]:
    """Loads DTM data from files for the given coordinates

    :param coords: a 2D numpy array of coordinates
    :return: a dictionary which maps filename to DTM numpy array
    """
    hgt_files = list(set([_get_file_name(*coord) for coord in coords]))
    if None in hgt_files:
        hgt_files.remove(None)
    return {file: _open_hgt_file(file) for file in hgt_files}


def _read_elevation_from_array(elevations: np.ndarray, lat, lon) -> int:
    """Given a DTM and a coordinate returns the elevation

    :param elevations: DTM numpy array
    :param lat: latitude
    :param lon: longitude
    :return: elevation in meters
    """
    lat_row = int(round((lat - int(lat)) * (SAMPLES - 1), 0))
    lon_row = int(round((lon - int(lon)) * (SAMPLES - 1), 0))

    return elevations[SAMPLES - 1 - lat_row, lon_row].astype(int)


def _open_hgt_file(filename: str) -> np.ndarray:
    """Given a DTM file name returns a DTM numpy array

    :param filename: DTM file name
    :return: DTM numpy array
    """
    size = os.path.getsize(filename)
    dim = int(math.sqrt(size / 2))

    assert dim * dim * 2 == size, 'Invalid file size'

    return np.fromfile(filename, np.dtype('>i2'), dim * dim).reshape((dim, dim))


def _get_file_name(lat, lon) -> Optional[str]:
    """Returns the file name of the given coordinate. If it doesn't exist, returns none.

    :param lat: latitude
    :param lon: longitude
    :return: file name
    """
    ns = 'N' if lat >= 0 else 'S'
    ew = 'E' if lon >= 0 else 'W'

    hgt_file = "%(ns)s%(lat)02d%(ew)s%(lon)03d.hgt" % {'lat': abs(lat), 'lon': abs(lon), 'ns': ns, 'ew': ew}
    hgt_file_path = os.path.join(DTM_DIR, hgt_file)
    if os.path.isfile(hgt_file_path):
        return hgt_file_path
    else:
        print(f"{hgt_file_path} not found")
        return None


if __name__ == '__main__':
    bound_files = [f for f in os.listdir(BOUNDS_DIR) if f.endswith('.json')]
    points = []
    for bf in bound_files:
        with open(os.path.join(BOUNDS_DIR, bf)) as f:
            bounds = json.load(f)
        points += [(bounds["south"], bounds["west"]),
                   (bounds["south"], bounds["east"]),
                   (bounds["north"], bounds["east"]),
                   (bounds["north"], bounds["west"])]

    coords = np.array(points)
    elevation = get_elevation(coords)
    print(elevation)
