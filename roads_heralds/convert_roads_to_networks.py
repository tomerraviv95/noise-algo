import os
import argparse

from tqdm import tqdm

from algo_config.algo_config import AlgorithmConfig
from dir_definitions import BENCHMARK_DIR, ROADS_NETWORKS_DIR
from roads_heralds.roads_to_networks.core import generate_roads_network
import networkx as nx

benchmark_files = [os.path.join(BENCHMARK_DIR, file)
                   for file in os.listdir(BENCHMARK_DIR) if file.endswith('.yaml')]


def read_network_graph() -> nx.Graph:
    graph_path = os.path.join(ROADS_NETWORKS_DIR, f'roads_graph_{AlgorithmConfig().get_name()}.pkl')
    if not os.path.isfile(graph_path):
        generate_roads_network()
    G = nx.read_gpickle(graph_path)
    return G


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
        read_network_graph()
