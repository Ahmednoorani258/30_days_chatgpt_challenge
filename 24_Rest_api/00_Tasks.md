# 🚀 Day 24: Build a REST API to Serve Your ML Model

## 🎯 Goal
Create a RESTful API using **FastAPI** (or Flask, if you prefer) that loads one of your trained machine-learning models (from earlier days) and serves predictions over HTTP. By the end of today, you’ll have a working API with automatic documentation and simple client examples.

---

## 📚 Why This Matters
1. **Deployability**: Expose your model to other services or front-end apps.
2. **Scalability**: REST APIs are the backbone of modern microservices.
3. **Professionalism**: Adds a critical skill to your portfolio—productionizing ML.

---

## 🛠️ Tools & Libraries
- **FastAPI**: Modern, high-performance Python web framework.
- **Uvicorn**: ASGI server for running FastAPI.
- **Pydantic**: Data validation and settings management.
- **Joblib or Pickle**: Model serialization.
- **Requests**: For client testing.

#### Install Dependencies:

# pip install fastapi uvicorn joblib pydantic scikit-learn


🔧 Step-by-Step Tasks
1️⃣ Choose Your Model
Select a trained model from previous days—e.g., your Linear Regression from Day 10 or a text classifier from Day 19. Ensure you have a serialized file, e.g.:


<!-- joblib.dump(model, "model.joblib") -->

2️⃣ Create the API Project Structure
my_api/
├── app.py
├── model/
│   └── model.joblib
├── requirements.txt
└── tests/
    └── test_api.py

    
3️⃣ Implement the FastAPI App
In app.py:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("model/model.joblib")

# Define request schema
class PredictRequest(BaseModel):
    features: list[float]

# Define response schema
class PredictResponse(BaseModel):
    prediction: float

app = FastAPI(title="ML Model API")

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Convert features to array
    x = np.array(request.features).reshape(1, -1)
    # Make prediction
    y_pred = model.predict(x)[0]
    return PredictResponse(prediction=y_pred)

@app.get("/")
def root():
    return {"message": "Welcome to the ML Model API!"}

```
4️⃣ Run and Test the API Locally
Start the server:

bash
Copy
Edit
uvicorn app:app --reload
Visit http://127.0.0.1:8000/docs to see Swagger UI docs auto-generated by FastAPI.

Test the /predict endpoint directly in the browser UI.

5️⃣ Write Automated Tests
In tests/test_api.py:

``` python
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "Welcome" in res.json()["message"]

def test_predict():
    payload = {"features": [1500.0]}  # adjust to your model's expected input
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    assert "prediction" in res.json()
Run tests:


pytest test
```

6️⃣ Package Dependencies
Freeze your dependencies:

<!-- pip freeze > requirements.txt -->
## ✅ Day 24 Checklist

| **Task**                                      | **Done?** |
|-----------------------------------------------|-----------|
| Serialized your ML model (joblib or pickle)   | ☐         |
| Created FastAPI project structure             | ☐         |
| Implemented `/predict` POST endpoint          | ☐         |
| Added root GET endpoint                       | ☐         |
| Ran server and tested via Swagger UI          | ☐         |
| Wrote automated tests for endpoints           | ☐         |
| Updated `requirements.txt`                    | ☐         |