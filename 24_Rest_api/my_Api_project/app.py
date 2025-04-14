# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("model/model.joblib")

# Define request schema (what data we expect)
class PredictRequest(BaseModel):
    features: list[float]

# Define response schema (what data we send back)
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