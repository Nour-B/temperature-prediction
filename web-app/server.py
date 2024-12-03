# import joblib
import os

import mlflow
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
    dtr_model = mlflow.pyfunc.load_model("runs:/8585b1f562df48aea605f9cace4f70b0/DecisionTreeRegressor")
    df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    pred = dtr_model.predict(df)
    return {"Predicted tempreature": float(pred)}


@app.post("/LinearRegression")
def predict_temperature_lin_reg_model(data: Temperature):
    lin_reg_model = mlflow.pyfunc.load_model("runs:/8585b1f562df48aea605f9cace4f70b0/LinearRegression")
    df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    pred = lin_reg_model.predict(df)
    return {"Predicted tempreature": float(pred)}


@app.post("/RandomForestRegressor")
def predict_temperature_rfr_model(data: Temperature):
    rfr_model = mlflow.pyfunc.load_model("runs:/8585b1f562df48aea605f9cace4f70b0/RandomForestRegressor")
    df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    pred = rfr_model.predict(df)
    return {"Predicted tempreature": float(pred)}
