from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import yaml

# Load config and model
with open("params.yml", "r") as f:
    config = yaml.safe_load(f)

model = joblib.load("model.joblib")

app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using a trained Random Forest model",
    version="1.0"
)

# Define input schema
class HouseFeatures(BaseModel):
    features: dict

@app.post("/predict")
def predict(data: HouseFeatures):
    try:
        df = pd.DataFrame([data.features])
        prediction = model.predict(df)
        return {"prediction": prediction[0]}
    except Exception as e:
        return {"error": str(e)}
