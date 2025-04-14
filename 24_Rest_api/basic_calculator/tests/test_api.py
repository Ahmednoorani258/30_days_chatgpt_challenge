# tests/test_api.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "Welcome" in res.json()["message"]

def test_add():
    payload = {"a": 2.0, "b": 3.0}
    res = client.post("/add", json=payload)
    assert res.status_code == 200
    assert res.json() == {"result": 5.0}