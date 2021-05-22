from enum import Enum

import numpy as np


class CharacterType(Enum):
    CHARISMATIC = 0
    SHY = 1


class Herald:
    def __init__(self, location: np.ndarray, effective_radius: float,
                 character: CharacterType = CharacterType.CHARISMATIC):
        self._location = location
        self._effective_radius = effective_radius
        self._character = character

    @property
    def location(self) -> np.ndarray:
        return self._location

    @property
    def effective_radius(self) -> float:
        return self._effective_radius

    @property
    def character(self) -> CharacterType:
        return self._character
