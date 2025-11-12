from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = [1.0]
    return model

@pytest.fixture
def mock_pandas():
  mock_pandas = MagicMock()
  mock_pandas.DataFrame.return_value = {"test": [1, 2, 3]}
  with patch("scripts.session_3.router.predict.pd", mock_pandas):
    yield mock_pandas

@pytest.fixture
def mock_mlflow_server(mock_model):
    mlflow_server = MagicMock()
    mlflow_server.sklearn.load_model.return_value = mock_model
    with patch("scripts.session_3.router.predict.mlflow", mlflow_server):
        yield mlflow_server


def test_get_model(mock_mlflow_server, mock_model):
    from scripts.session_3.router.predict import get_model

    model = get_model()
    assert model is not None
    assert model == mock_model


def test_func_predict(mock_mlflow_server, mock_model):
    from scripts.session_3.router.predict import func_predict
    from scripts.session_3.schemas.request import HousingPredictionRequest

    request = HousingPredictionRequest(
        average_area_income=100000,
        average_area_house_age=10,
        average_area_number_of_rooms=3,
        average_area_number_of_bedrooms=2,
        area_population=100000,
    )
    response = func_predict(request)
    assert response.predicted_price == 1.0


def test_router(mock_mlflow_server, mock_model,mock_pandas):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from scripts.session_3.router.predict import housing_router

    app = FastAPI()
    app.include_router(housing_router)
    client = TestClient(app)
    response = client.post(
        "/housing/predict",
        json={
            "average_area_income": 100000,
            "average_area_house_age": 10,
            "average_area_number_of_rooms": 3,
            "average_area_number_of_bedrooms": 2,
            "area_population": 100000,
        },
    )
    assert response.status_code == 200
    assert response.json() == {"predicted_price": 1.0}

    mock_pandas.DataFrame.assert_called_once_with({
        "Avg. Area Income": [100000],
        "Avg. Area House Age": [10],
        "Avg. Area Number of Rooms": [3],
        "Avg. Area Number of Bedrooms": [2],
        "Area Population": [100000]
    })