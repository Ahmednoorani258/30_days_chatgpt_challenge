# tests/test_api.py
from fastapi.testclient import TestClient
import app

client = TestClient(app)

def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "Welcome" in res.json()["message"]

def test_predict():
    payload = {"features": [1500.0]}  # Adjust to your model's expected input
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    assert "prediction" in res.json()