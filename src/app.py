# src/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd

# <<< --- ADD THIS LINE --- >>>
# Tell the app where to find the MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 1. Define the input data schema using Pydantic
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    IsAlone: int

# 2. Initialize the FastAPI application
app = FastAPI(title="Titanic Survival Prediction API", version="1.0")

# 3. Load the model from the MLflow Model Registry
model_uri = "models:/TitanicSurvivalModel/latest"
print(f"Loading model from: {model_uri}")
try:
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# 4. Define the prediction endpoint
@app.post("/predict")
def predict(passenger: Passenger):
    if model is None:
        return {"error": "Model not loaded. Please check the server logs."}

    input_df = pd.DataFrame([passenger.dict()])
    prediction = model.predict(input_df)
    survival_prediction = int(prediction[0])

    return {"prediction": survival_prediction, "prediction_label": "Survived" if survival_prediction == 1 else "Did not survive"}

# Optional: A root endpoint for health checks
@app.get("/")
def read_root():
    return {"status": "API is running"}