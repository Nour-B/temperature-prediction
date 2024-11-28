import os

import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


class Trainer:
    def __init__(self, models, test_size, random_state):
        self.models = models
        self.test_size = test_size
        self.random_state = random_state

    def prepare_data(self, data):
        feature_selection = ["month", "cloud_cover", "sunshine", "precipitation", "pressure", "global_radiation"]
        target_var = "mean_temp"

        # Subset features and target sets
        X = data[feature_selection]
        y = data[target_var]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Impute missing values and scale the data
        pipeline = Pipeline(steps=[("imputation_mean", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])

        # Fit on the training data
        X_train = pipeline.fit_transform(X_train)

        # Transform on the test data
        X_test = pipeline.transform(X_test)

        return X, y, X_train, X_test, y_train, y_test

    def train_save_models(self, X_train, y_train):
        model_map = {
            "LinearRegression": LinearRegression,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "RandomForestRegressor": RandomForestRegressor,
        }
        for m in self.models:
            model_name = model_map[m.name]
            model = model_name(**m.params)
            model.fit(X_train, y_train)
            model_file_path = os.path.join(m.store_path, m.store_filename)
            joblib.dump(model, model_file_path)
