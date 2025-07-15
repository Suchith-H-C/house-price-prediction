from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model and feature columns
model = joblib.load("model.joblib")
columns = joblib.load("columns.pkl")

class HouseFeatures(BaseModel):
    first_flr: float
    second_flr: float
    bedrooms: int
    total_rooms: int
    garage: float
    living_area: float
    lot_area: float
    quality: int

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Don’t send prediction key at all initially
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
    quality: int = Form(...),
):
    try:
        features = HouseFeatures(
            first_flr=first_flr,
            second_flr=second_flr,
            bedrooms=bedrooms,
            total_rooms=total_rooms,
            garage=garage,
            living_area=living_area,
            lot_area=lot_area,
            quality=quality
        )

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
        result = f"${round(prediction, 2):,.2f}"

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": result
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e)
        })
