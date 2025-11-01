from fastapi import FastAPI
import uvicorn
from enum import Enum
import mlflow.sklearn
import pandas as pd
import numpy as np
from fastapi import APIRouter
app = FastAPI()

from scripts.session_3.schemas.request import PredictRequest
from scripts.session_3.schemas.response import PredictResponse

mlflow.set_tracking_uri('http://localhost:8080')
model_name = "Housing_price_predictor"
model_version = "1"
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

housing_router = APIRouter(prefix="/housing")

@housing_router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    sample_data = {
        "Avg. Area Income": [request.avg_area_income],
        "Avg. Area House Age": [request.avg_area_house_age],
        "Avg. Area Number of Rooms": [request.avg_area_number_of_rooms],
        "Avg. Area Number of Bedrooms": [request.avg_area_number_of_bedrooms],
        "Area Population": [request.area_population],
    }
    df = pd.DataFrame(sample_data)
    predictions = model.predict(df)[0]
    return PredictResponse(result=predictions)