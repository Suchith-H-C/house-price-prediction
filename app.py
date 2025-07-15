from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib
from prometheus_fastapi_instrumentator import Instrumentator


app = FastAPI()
Instrumentator().instrument(app).expose(app)
templates = Jinja2Templates(directory="templates")

# Load model and expected column order
model = joblib.load("model.joblib")
columns = joblib.load("columns.pkl")

# JSON input schema for API clients
class HouseFeatures(BaseModel):
    first_flr: float
    second_flr: float
    bedrooms: int
    total_rooms: int
    garage: float
    living_area: float
    lot_area: float
    quality: int

# Home route to render form
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# HTML form submission (from index.html)
@app.post("/predict", response_class=HTMLResponse)
async def predict_form(
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

# Optional: JSON-based prediction endpoint (for curl/Postman)
@app.post("/predict-json", response_model=dict)
async def predict_json(features: HouseFeatures):
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
        return {"predicted_price": round(prediction, 2)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
