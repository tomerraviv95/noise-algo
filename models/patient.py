import numpy as np


class Patient:
    def __init__(self, location: np.ndarray, contagion_radius: float, effective_radius: float):
        self._location = location
        self._contagion_radius = contagion_radius
        self._effective_radius = effective_radius

    @property
    def location(self) -> np.ndarray:
        return self._location

    @property
    def contagion_radius(self) -> float:
        return self._contagion_radius

    @property
    def effective_radius(self) -> float:
        return self._effective_radius
