from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(title="🏡 House Price Prediction API")

templates = Jinja2Templates(directory="templates")

# Load model and feature columns
model = joblib.load("model.joblib")
columns = joblib.load("columns.pkl")

# Pydantic schema for API
class HouseFeatures(BaseModel):
    first_flr: float = Field(..., example=856)
    second_flr: float = Field(..., example=450)
    bedrooms: int = Field(..., example=3)
    total_rooms: int = Field(..., example=6)
    garage: float = Field(..., example=480)
    living_area: float = Field(..., example=1500)
    lot_area: float = Field(..., example=7500)
    quality: int = Field(..., ge=1, le=10, example=7)

# Web UI endpoint
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

# Web form endpoint
@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    first_flr: float = Form(...),
    second_flr: float = Form(...),
    bedrooms: int = Form(...),
    total_rooms: int = Form(...),
    garage: float = Form(...),
    living_area: float = Form(...),
    lot_area: float = Form(...),
    quality: int = Form(...)
):
    try:
        input_df = pd.DataFrame([{
            "1stFlrSF": first_flr,
            "2ndFlrSF": second_flr,
            "BedroomAbvGr": bedrooms,
            "TotRmsAbvGrd": total_rooms,
            "GarageArea": garage,
            "GrLivArea": living_area,
            "LotArea": lot_area,
            "OverallQual": quality
        }])

        input_df = input_df.reindex(columns=columns, fill_value=0)
        prediction = model.predict(input_df)[0]
        result = f"${round(prediction, 2):,}"

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": result
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": f"Error: {str(e)}"
        })

# JSON API for Swagger & clients
@app.post("/predict", response_model=dict)
def predict_api(features: HouseFeatures):
    try:
        input_df = pd.DataFrame([{
            "1stFlrSF": features.first_flr,
            "2ndFlrSF": features.second_flr,
            "BedroomAbvGr": features.bedrooms,
            "TotRmsAbvGrd": features.total_rooms,
            "GarageArea": features.garage,
            "GrLivArea": features.living_area,
            "LotArea": features.lot_area,
            "OverallQual": features.quality
        }])
        input_df = input_df.reindex(columns=columns, fill_value=0)
        prediction = model.predict(input_df)[0]
        result = f"${round(prediction, 2):,}"

        return {"predicted_price": result}

    except Exception as e:
        return {"error": str(e)}
