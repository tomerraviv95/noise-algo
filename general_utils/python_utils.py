import pickle as pkl
import sys
from algo_config.algo_config import AlgorithmConfig


def make_iterable(obj: object) -> object:
    try:
        iter(obj)
    except TypeError:
        return [obj]
    return obj


def save_pkl(path: str, array: object):
    with open(path, 'wb') as f:
        pkl.dump(array, f)


def load_pkl(path: str):
    with open(path, 'rb') as f:
        return pkl.load(f)


def load_config(benchmark_file: str):
    config = AlgorithmConfig()
    try:
        config.load_config(benchmark_file)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
    return config
