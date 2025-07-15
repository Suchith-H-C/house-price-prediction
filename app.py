from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model once at startup
model = joblib.load("model.joblib")

# Replace with your actual feature list
FEATURE_NAMES = [
    "1stFlrSF", "2ndFlrSF", "BedroomAbvGr", "TotRmsAbvGrd",
    "GarageArea", "GrLivArea", "LotArea", "OverallQual"
]

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
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
        # Map inputs to feature names
        input_data = pd.DataFrame([{
            "1stFlrSF": first_flr,
            "2ndFlrSF": second_flr,
            "BedroomAbvGr": bedrooms,
            "TotRmsAbvGrd": total_rooms,
            "GarageArea": garage,
            "GrLivArea": living_area,
            "LotArea": lot_area,
            "OverallQual": quality
        }])

        prediction = model.predict(input_data)[0]
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": f"${round(prediction, 2)}"
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": f"Error: {str(e)}"
        })
