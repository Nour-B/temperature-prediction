import pandas as pd
from omegaconf import OmegaConf


class Cleaner:

    def __init__(self):
        self.config = OmegaConf.load("./app/configs/config.yaml")
        self.data = self.config.data.raw_data_path
        self.clean_data_path = self.config.data.clean_data_path

    def clean_data(self, data):
        # Convert columns to the correct data type.
        data["date"] = pd.to_datetime(data["date"], format="%Y%m%d")
        data["month"] = data["date"].dt.month
        data["year"] = data["date"].dt.year

        # drop null values
        data = data.dropna(subset=['mean_temp'])

        # save the data
        # data.to_csv(self.clean_data_path, index=None)

        return data
