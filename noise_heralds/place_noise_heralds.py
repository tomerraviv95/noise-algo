import os

from tqdm import tqdm

from algo_config.algo_config import AlgorithmConfig
from dir_definitions import BENCHMARK_DIR
from general_utils.patient_utils import get_patient_filtered_polygons
from models.scenario import Scenario
from noise_heralds.make_noise.core import place_heralds_main
from noise_heralds.segment import read_segmented_areas
from visualization.visualizer import draw

if __name__ == '__main__':

    alg_config = AlgorithmConfig()

    for file in tqdm(os.listdir(BENCHMARK_DIR)):
        alg_config.load_config(os.path.join(BENCHMARK_DIR, file))
        scenario = Scenario(None)

        # calculate patient's contagion and effective polygons
        patient_contagion_polygon, patient_effective_polygon = get_patient_filtered_polygons(scenario)

        # read the segmented areas
        seg_areas = read_segmented_areas()

        # place heralds
        noise_output = place_heralds_main(seg_areas)

        # draw the output
        draw(scenario=scenario,
             seg_areas=seg_areas,
             patient_contagion_polygon=patient_contagion_polygon,
             patient_effective_polygon=patient_effective_polygon,
             noise_output=noise_output,
             to_file=f"noise_only/{AlgorithmConfig().get_name()}")
