import requests

BASE_URL = "http://localhost:8000"


def test_health_check():
    response = requests.get(f"{BASE_URL}/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

    if data["model_loaded"]:
        assert "model_name" in data
    else:
        assert "message" in data


def test_bad_health_check():
    response = requests.get(f"{BASE_URL}/ealth")
    assert response.status_code == 404
