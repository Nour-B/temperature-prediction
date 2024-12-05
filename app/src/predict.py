import logging
import os

import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


class Predictor:
    def __init__(self, models, metric_name):
        self.models = models
        self.metric_name = metric_name

    def load_model(self, model_path, model_name):
        model_file_path = os.path.join(model_path, model_name)
        return joblib.load(model_file_path)

    def evaluate_model_mlflow(self, X_test, y_test):
        mlflow.set_tracking_uri(os.environ["TRACKING_URI"])
        mlflow.set_experiment("Predicting Temperature in London")

        with mlflow.start_run() as run:
            metric = {
                "root_mean_squared_error": root_mean_squared_error,
                "mean_squared_error": mean_squared_error,
                "mean_absolute_error": mean_absolute_error,
            }[self.metric_name]

            for m in self.models:
                model = self.load_model(m.store_path, m.store_filename)
                y_pred = model.predict(X_test)
                met = metric(y_test, y_pred)

                # Print evaluation results
                print("\n============= Model Evaluation Results ==============")
                print(f"Model: {m.name}")
                print(f"{self.metric_name}: {float(met)}")
                print("=====================================================\n")
                # Log performance
                mlflow.log_params(m.params)
                mlflow.log_metric("rmse" + "_" + m.name, met)
                mlflow.sklearn.log_model(sk_model=model, artifact_path=m.name, registered_model_name=m.name)
                # Register the model
                print(f"{run.info.run_id}")
                # model_name = m.name
                # model_version = 1
                # model_uri = f"models:/{model_name}/{model_version}"
                # mlflow.register_model(model_uri, model_name)

                logging.info("MLflow tracking completed successfully")
