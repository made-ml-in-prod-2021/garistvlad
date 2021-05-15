import os
import sys
from unittest.mock import patch

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        os.path.pardir,
    )
)  # .../api
sys.path.append(BASE_DIR)

import pytest
from fastapi.testclient import TestClient

from app.main import app, PRETRAINED_MODELS_DIR
from app.custom_datamodels import HeartRequestModel
from app.utils.loader import load_pickled
from app.utils.inference import predict_pipeline


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def data_sample():
    features = {
        'age': 35, 'sex': 0, 'cp': 0,
        'trestbps': 138, 'chol': 183, 'fbs': 0,
        'restecg': 1, 'thalach': 182, 'exang': 0,
        'oldpeak': 1.4, 'slope': 2, 'ca': 0, 'thal': 2
    }
    y_proba = predict_pipeline(
        data=HeartRequestModel(**features).to_pandas(),
        transformer=load_pickled(os.path.join(PRETRAINED_MODELS_DIR, "transformer.pickle")),
        classifier=load_pickled(os.path.join(PRETRAINED_MODELS_DIR, "classifier.pickle")),
    )
    return features, y_proba


def test_model_and_transformer_load_on_startup(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() is True, (
        "fitted classifier and transformer should be loaded on startup"
    )


def test_root_page_loaded(client):
    expected_description = "MADE. ML in production. HA #2: REST service"
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["description"] == expected_description, (
        "Description is not the same as expected"
    )


@patch("app.main.check_health")
def test_predict_raise_500_error_if_pretrained_not_loaded(mock_check_health, data_sample, client):
    # expected result
    expected_status_code = 500
    expected_json_response = {"detail": "Model should be loaded for making predictions"}
    # mock external dependencies
    mock_check_health.return_value = False
    # test
    resp = client.post("/predict", json=data_sample[0])
    assert resp.status_code == expected_status_code
    assert resp.json() == expected_json_response


@pytest.mark.parametrize(
    ["feature_name", "broken_value", "expected_status_code"],
    [
        pytest.param("trestbps", "high", 422, id="WrongDataType"),
        pytest.param("sex", 2, 400, id="NotBinaryFeature"),
        pytest.param("cp", 4, 400, id="WrongCategory"),
        pytest.param("age", 200, 400, id="NumericalOutOfRange"),
    ]
)
def test_predict_raise_400_if_data_not_valid(
        feature_name, broken_value, expected_status_code,
        data_sample, client
):
    # not valid data
    broken_data = data_sample[0].copy()
    broken_data[feature_name] = broken_value
    # test
    resp = client.post("/predict", json=broken_data)
    assert resp.status_code == expected_status_code
    assert "detail" in resp.json()


def test_predict_returns_accurate_probability(data_sample, client):
    expected_status_code = 200
    expected_response = {"predicted_probability": data_sample[1]}
    resp = client.post("/predict", json=data_sample[0])
    assert resp.status_code == expected_status_code
    assert resp.json() == expected_response
