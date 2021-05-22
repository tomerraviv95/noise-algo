import os
import pickle
from typing import NamedTuple

from shapely.geometry import MultiPolygon, Polygon

from noise_heralds.los.los_generator import get_los
from noise_heralds.villages.village_outliner import get_villages_outline
from dir_definitions import SEGMENTED_AREAS_DIR, BENCHMARK_DIR
from algo_config.algo_config import AlgorithmConfig


class SegmentedAreas(NamedTuple):
    villages: MultiPolygon
    los: MultiPolygon


def calc_areas() -> SegmentedAreas:
    """Calculates segmented areas (from LOS and villages)

    :return: segmented areas object
    """
    los = get_los()
    los = [mp.buffer(0) for mp in los]  # make sure multipolygon valid

    villages = get_villages_outline()
    villages = villages.buffer(0.001)

    if type(villages) == Polygon:
        villages = MultiPolygon([villages])

    if type(los[1]) == Polygon:
        los[1] = MultiPolygon([los[1]])

    return SegmentedAreas(villages=villages,
                          los=los[1])


def save_segmented_areas(areas: SegmentedAreas) -> None:
    """Saves segmented areas to a file

    :param areas: segmented areas
    :return: none
    """
    if not os.path.isdir(SEGMENTED_AREAS_DIR):
        os.mkdir(SEGMENTED_AREAS_DIR)

    segmented_areas_path = os.path.join(SEGMENTED_AREAS_DIR, f'areas_{AlgorithmConfig().get_name()}.pkl')
    with open(segmented_areas_path, 'wb') as f:
        pickle.dump(areas, f)


def read_segmented_areas() -> SegmentedAreas:
    """Reads segmented areas from a file
    :return: segmented areas
    """
    segmented_areas_path = os.path.join(SEGMENTED_AREAS_DIR, f'areas_{AlgorithmConfig().get_name()}.pkl')
    if not os.path.isfile(segmented_areas_path):
        seg_areas = calc_areas()
        save_segmented_areas(seg_areas)

    with open(segmented_areas_path, 'rb') as f:
        return pickle.load(f)


def create_segmentation():
    seg_areas = calc_areas()
    save_segmented_areas(seg_areas)


if __name__ == '__main__':
    alg_config = AlgorithmConfig()
    for benchmark_file in os.listdir(BENCHMARK_DIR):
        alg_config.load_config(os.path.join(BENCHMARK_DIR, benchmark_file))
        create_segmentation()
