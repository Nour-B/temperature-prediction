import pandas as pd
from omegaconf import OmegaConf


class Loader:

    def __init__(self):
        self.config = OmegaConf.load("./app/configs/config.yaml")

    def load_raw_data(self):
        raw_data_path = self.config.data.raw_data_path
        raw_data = pd.read_csv(raw_data_path)
        return raw_data

    def load_clean_data(self):
        clean_data_path = self.config.data.clean_data_path
        clean_data = pd.read_csv(clean_data_path)
        return clean_data
