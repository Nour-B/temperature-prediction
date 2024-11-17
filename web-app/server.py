from pydantic import BaseModel
from fastapi import FastAPI
import joblib
import pandas as pd

class Temperature(BaseModel):
    month: int
    cloud_cover: float
    sunshine: float
    precipitation: float
    pressure: float
    global_radiation: float


app = FastAPI()
dtr_model = joblib.load('models/DecisionTreeRegressor.pkl')
lin_reg_model = joblib.load('models/LinearRegression.pkl')
rfr_model = joblib.load('models/RandomForestRegressor.pkl')


@app.get('/')
def index():
    return {'messeage':'Predicting Temperature'}

@app.post('/predict_temperature')
def predict_temperature(data: Temperature):
    df = pd.DataFrame([data.dict().values()], 
                          columns=data.dict().keys())
    pred = dtr_model.predict(df)
    return {"Predicted tempreature": float(pred)}






