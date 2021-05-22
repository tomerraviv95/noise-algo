import numpy as np


def vectorized(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """
    finds the list of pixels that the line between start to end is going through
    :param start: N-dim numpy array of the start point
    :param end: N-dim numpy array of the end point
    :return: array of the desired cells
    """
    vec = end - start
    advance_idx = np.argmax(np.abs(vec))
    amount = np.abs(start[advance_idx] - end[advance_idx]) + 1
    cells = np.linspace(start, end, int(amount))

    return cells


def los3d(elevations: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """
    checks if object in end is visible from start
    :param elevations: height of topography
    :param start: 3d numpy array of start point (values according to coordinates in elevations matrix)
    :param end: 3d numpy array of end point (values according to coordinates in elevations matrix)
    :return: True if a line of sight exists, else false
    """
    start = start.reshape(-1, 3)
    end = end.reshape(-1, 3)
    visible = np.zeros(start.shape[0]).astype(bool)
    for i in range(start.shape[0]):
        cells = vectorized(start[i], end[i])
        visible[i] = np.all(elevations[cells[:, 0].astype(int), cells[:, 1].astype(int)] <= cells[:, 2])
    return visible


def object_visibility(elevations: np.ndarray, location: np.ndarray) -> np.ndarray:
    start = np.tile(location.reshape(-1, 3), (elevations.size, 1))
    x = np.linspace(0, elevations.shape[0], elevations.shape[0], endpoint=False)
    y = np.linspace(0, elevations.shape[1], elevations.shape[1], endpoint=False)
    xx, yy = np.meshgrid(x, y)
    end = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1), elevations.T.reshape(-1, 1)))
    return los3d(elevations, start, end)


def find_los(elevations: np.ndarray, coord: np.ndarray, coord_delta_height: float,
             seen_from_height: float) -> np.ndarray:
    """

    :param elevations:
    :param coord:
    :param coord_delta_height: agl, meters
    :param seen_from_height: agl, meters
    :return:
    """
    coord = coord.reshape(-1)
    coord[2] += coord_delta_height
    elevation_with_height = elevations + seen_from_height
    start = np.tile(coord, (elevation_with_height.size, 1))

    x = np.linspace(0, elevation_with_height.shape[0], elevation_with_height.shape[0], endpoint=False)
    y = np.linspace(0, elevation_with_height.shape[1], elevation_with_height.shape[1], endpoint=False)
    xx, yy = np.meshgrid(x, y)
    end = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1), elevation_with_height.T.reshape(-1, 1)))
    result = los3d(elevations, start, end).reshape(elevations.shape)
    return np.rot90(result)
