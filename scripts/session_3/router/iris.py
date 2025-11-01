from fastapi import FastAPI
import uvicorn
from enum import Enum
import mlflow.sklearn
import pandas as pd
import numpy as np
from fastapi import APIRouter
app = FastAPI()

from scripts.session_3.schemas.request import IrisPredictRequest
from scripts.session_3.schemas.response import IrisPredictResponse

mlflow.set_tracking_uri('http://localhost:8080')
model_name = "iris_model"
model_version = "1"
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri) 

iris_router = APIRouter(prefix="/iris")

@iris_router.post("/iris_predict", response_model=IrisPredictResponse)
def irisPredict(request: IrisPredictRequest) -> IrisPredictResponse:
    sample_data = {
        "sepal length (cm)": [request.sepal_length],
        "sepal width (cm)": [request.sepal_width],
        "petal length (cm)": [request.petal_length],
        "petal width (cm)": [request.petal_width],

    }
    df = pd.DataFrame(sample_data)
    df = df.to_numpy()[:, (2,3)]
    predictions = model.predict(df)[0]
    return IrisPredictResponse(result=predictions)