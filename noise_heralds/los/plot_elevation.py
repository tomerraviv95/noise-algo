import argparse
import os

from scipy.ndimage import label
from tqdm import tqdm
from algo_config.algo_config import AlgorithmConfig
from dir_definitions import BENCHMARK_DIR
from models.scenario import Scenario
from noise_heralds.los.los_generator import generate_elevation_grid
import matplotlib.pyplot as plt
import skimage.morphology
from skimage.morphology import square
import numpy as np

benchmark_files = [os.path.join(BENCHMARK_DIR, file)
                   for file in os.listdir(BENCHMARK_DIR) if file.endswith('.yaml')]

if __name__ == '__main__':
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

    for file in tqdm(os.listdir(BENCHMARK_DIR)):
        AlgorithmConfig().load_config(os.path.join(BENCHMARK_DIR, file))
        scenario = Scenario(heralds=None)
        structure = square(11)
        elevations = generate_elevation_grid(scenario.bbox, 100)

        white_tophat_elevations = skimage.morphology.white_tophat(elevations, selem=structure)

        maxima_elevations = skimage.morphology.h_maxima(white_tophat_elevations, h=10, selem=structure)

        peaks_mask = np.zeros_like(elevations)
        peaks_elevations = elevations.copy()
        inds = np.where(maxima_elevations)
        for ind_x, ind_y in zip(*inds):
            val = 0.7 * white_tophat_elevations[ind_x, ind_y]
            binarized_image = white_tophat_elevations > val
            labeled_array, num_features = label(binarized_image)
            cur_label = labeled_array[ind_x, ind_y]
            peaks_mask += labeled_array == cur_label
        peaks_mask = peaks_mask > 0
        peaks_elevations[peaks_mask] = 0

        f = plt.figure()
        f.add_subplot(2, 2, 1)
        plt.title('Original Image')
        plt.imshow(elevations, cmap='hot')

        f.add_subplot(2, 2, 2)
        plt.imshow(white_tophat_elevations, cmap='hot')
        plt.title('White Tophat (Original Minus Opening)')

        f.add_subplot(2, 2, 3)
        plt.imshow(maxima_elevations, cmap='hot')
        plt.title('H-maxima Transform')

        f.add_subplot(2, 2, 4)
        plt.title('Peaks Image')
        plt.imshow(peaks_elevations, cmap='hot')
        plt.show(block=True)
