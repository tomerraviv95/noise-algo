import numpy as np


def angle_between(v1: np.ndarray, v2: np.ndarray):
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def normalize_vector(v: np.ndarray):
    return v / np.linalg.norm(v)
