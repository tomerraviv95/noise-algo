import argparse
import os

from tqdm import tqdm

from algo_config.algo_config import AlgorithmConfig
from dir_definitions import BENCHMARK_DIR
from general_utils.patient_utils import get_patient_filtered_polygons
from models.scenario import Scenario

from noise_heralds.make_noise.core import place_heralds_main
from noise_heralds.segment import read_segmented_areas, SegmentedAreas
from roads_heralds.block_roads.core import block_roads
from roads_heralds.convert_roads_to_networks import read_network_graph
from visualization.visualizer import draw

benchmark_files = [os.path.join(BENCHMARK_DIR, file)
                   for file in os.listdir(BENCHMARK_DIR) if file.endswith('.yaml')]

if __name__ == "__main__":
    alg_config = AlgorithmConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("-scenarios", help="scenarios to run", nargs="+", type=int)
    args = parser.parse_args()

    if args.scenarios is not None:
        benchmark_files = [os.path.join(BENCHMARK_DIR, f"benchmark_{file}.yaml") for file in args.scenarios]
    else:
        benchmark_files = sorted(benchmark_files)

    for benchmark_file in tqdm(benchmark_files):
        alg_config.load_config(os.path.join(BENCHMARK_DIR, benchmark_file))
        scenario = Scenario(heralds=None)

        # calculate patient's contagion and effective polygons
        patient_contagion_polygon, patient_effective_polygon = get_patient_filtered_polygons(scenario)

        # read the segmented areas (villages and LOS map)
        seg_areas: SegmentedAreas = read_segmented_areas()

        # load the roads graph
        G = read_network_graph()

        # calculate noise heralds locations
        noise_output = place_heralds_main(seg_areas)

        # calculate roads heralds locations
        blocks_output = block_roads(G, seg_areas.villages, patient_effective_polygon)

        # draw the solution
        draw(scenario=scenario,
             seg_areas=seg_areas,
             patient_contagion_polygon=patient_contagion_polygon,
             patient_effective_polygon=patient_effective_polygon,
             blocks_output=blocks_output,
             noise_output=noise_output,
             to_file=f'full/{alg_config.get_name()}')
