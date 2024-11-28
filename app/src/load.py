import pandas as pd


class Loader:
    def __init__(self, path):
        self.path = path

    def load_data(self):
        data = pd.read_csv(self.path)
        return data
