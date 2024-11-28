import pandas as pd


class Cleaner:
    def __init__(self, path):
        self.path = path

    def clean_data(self, data):
        # Convert columns to the correct data type
        data["date"] = pd.to_datetime(data["date"], format="%Y%m%d")
        data["month"] = data["date"].dt.month
        data["year"] = data["date"].dt.year

        # drop null values
        data = data.dropna(subset=["mean_temp"])

        # save the cleaned data
        data.to_csv(self.path, index=None)

        return data
