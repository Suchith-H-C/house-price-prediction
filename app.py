from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="🏡 House Price Predictor")

# Setup template directory
templates = Jinja2Templates(directory="templates")

# Load model and feature columns
model = joblib.load("model.joblib")
columns = joblib.load("columns.pkl")

# Input schema for API
class HouseFeatures(BaseModel):
    first_flr: float
    second_flr: float
    bedrooms: int
    total_rooms: int
    garage: float
    living_area: float
    lot_area: float
    quality: int

# Home route: UI
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Form route: Handle POST from browser form
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
        data = pd.DataFrame([{
            "1stFlrSF": first_flr,
            "2ndFlrSF": second_flr,
            "BedroomAbvGr": bedrooms,
            "TotRmsAbvGrd": total_rooms,
            "GarageArea": garage,
            "GrLivArea": living_area,
            "LotArea": lot_area,
            "OverallQual": quality
        }])
        data = data.reindex(columns=columns, fill_value=0)
        prediction = model.predict(data)[0]
        result = f"${prediction:,.2f}"
        return templates.TemplateResponse("index.html", {"request": request, "prediction": result})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "prediction": f"Error: {str(e)}"})

# Swagger-friendly JSON route
@app.post("/predict")
async def predict_api(features: HouseFeatures):
    try:
        data = pd.DataFrame([{
            "1stFlrSF": features.first_flr,
            "2ndFlrSF": features.second_flr,
            "BedroomAbvGr": features.bedrooms,
            "TotRmsAbvGrd": features.total_rooms,
            "GarageArea": features.garage,
            "GrLivArea": features.living_area,
            "LotArea": features.lot_area,
            "OverallQual": features.quality
        }])
        data = data.reindex(columns=columns, fill_value=0)
        prediction = model.predict(data)[0]
        return {"predicted_price": round(prediction, 2)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
