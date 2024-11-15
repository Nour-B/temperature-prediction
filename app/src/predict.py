import os
import joblib
import mlflow

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error
from omegaconf import OmegaConf


class Predictor:

    def __init__(self):
        self.config = OmegaConf.load("./app/configs/config.yaml")
        self.models = self.config.models
        self.metric_name = self.config.evaluate.metric

    def load_model(self,model_path,model_name):
        model_file_path = os.path.join(model_path, model_name)
        return joblib.load(model_file_path)
            
      
   
    def evaluate_model(self, X_test, y_test):
      
        metric = {
                "root_mean_squared_error": root_mean_squared_error,
                "mean_squared_error": mean_squared_error,
                "mean_absolute_error": mean_absolute_error
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
            

    def evaluate_model_mlflow(self, X_test, y_test):

        mlflow.set_tracking_uri("http://localhost:8080")
        mlflow.set_experiment("Predicting Temperature in London")   

        with mlflow.start_run() as run:   


            metric = {
                    "root_mean_squared_error": root_mean_squared_error,
                    "mean_squared_error": mean_squared_error,
                    "mean_absolute_error": mean_absolute_error
                }[self.metric_name]

            for m in self.models:
                model = self.load_model(m.store_path, m.store_filename)
                y_pred = model.predict(X_test)
                metric = metric(y_test, y_pred)
                # Print evaluation results
                print("\n============= Model Evaluation Results ==============")
                print(f"Model: {m.name}")
                print(f"Metric: {metric:.4f}")
                print("=====================================================\n")
                
                # Log performance 
                mlflow.log_params(m.params)
                mlflow.log_metric("rmse" + "_" + m.name, metric)
                mlflow.sklearn.log_model(model, m.name)



    
                   


