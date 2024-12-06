# import joblib
import os

import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


class Temperature(BaseModel):
    month: int
    cloud_cover: float
    sunshine: float
    precipitation: float
    pressure: float
    global_radiation: float


app = FastAPI()

mlflow.tracking.set_tracking_uri(os.environ["TRACKING_URI"])


@app.get("/")
def index():
    return {"messeage": "Predicting Temperature"}


@app.post("/DecisionTreeRegressor")
def predict_temperature_dtr_model(data: Temperature):
    # Load the model from the Model Registry
    model_name = "DecisionTreeRegressor"
    model_version = 1
    dtr_model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
    df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    pred = dtr_model.predict(df)
    return {"Predicted tempreature": float(pred)}


@app.post("/LinearRegression")
def predict_temperature_lin_reg_model(data: Temperature):
    model_name = "LinearRegression"
    model_version = 1
    lin_reg_model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
    df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    pred = lin_reg_model.predict(df)
    return {"Predicted tempreature": float(pred)}


@app.post("/RandomForestRegressor")
def predict_temperature_rfr_model(data: Temperature):
    model_name = "RandomForestRegressor"
    model_version = 1
    rfr_model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
    df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    pred = rfr_model.predict(df)
    return {"Predicted tempreature": float(pred)}
