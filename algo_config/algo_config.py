import os
from typing import Dict

import yaml

from dir_definitions import BENCHMARK_DIR


class AlgorithmConfig:
    __instance = None

    def __new__(cls):
        if AlgorithmConfig.__instance is None:
            AlgorithmConfig.__instance = object.__new__(cls)
            AlgorithmConfig.__instance.config = None
            AlgorithmConfig.__instance.load_default_config()
        return AlgorithmConfig.__instance

    def load_default_config(self):
        files = [file for file in os.listdir(BENCHMARK_DIR) if file.endswith('.yaml')]
        if len(files) == 0:
            raise FileNotFoundError("No benchmark files")
        files = sorted(files)
        self.load_config(os.path.join(BENCHMARK_DIR, files[0]))

    def load_config(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            self.config_name = os.path.splitext(os.path.basename(config_path))[0]

    def get_config(self) -> Dict:
        return self.config

    def get_value(self, value: str):
        return self.config[value]

    def get_name(self) -> str:
        return self.config_name.replace("benchmark_", "")
