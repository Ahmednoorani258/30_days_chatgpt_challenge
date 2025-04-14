# app.py
from fastapi import FastAPI
from pydantic import BaseModel

# Define request schema (what data we expect)
class AddRequest(BaseModel):
    a: float
    b: float

# Define response schema (what data we send back)
class AddResponse(BaseModel):
    result: float

app = FastAPI(title="Calculator API")

@app.post("/add", response_model=AddResponse)
def add_numbers(request: AddRequest):
    # Add the numbers
    sum_result = request.a + request.b
    return AddResponse(result=sum_result)

@app.get("/")
def root():
    return {"message": "Welcome to the Calculator API!"}