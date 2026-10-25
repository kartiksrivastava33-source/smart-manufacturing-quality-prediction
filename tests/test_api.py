
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"]


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "Type": "L",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 100
        }
    )

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["result"] in ["Failure", "No Failure"]
    assert 0 <= result["failure_probability"] <= 1
    assert result["threshold"] == 0.4


def test_invalid_machine_type():
    response = client.post(
        "/predict",
        json={
            "Type": "Z",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 100
        }
    )

    assert response.status_code == 422


def test_missing_required_field():
    response = client.post(
        "/predict",
        json={
            "Type": "L",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40.0
        }
    )

    assert response.status_code == 422

def test_invalid_temperature():
    response = client.post(
        "/predict",
        json={
            "Type": "L",
            "Air temperature [K]": -10.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 100
        }
    )

    assert response.status_code == 422
