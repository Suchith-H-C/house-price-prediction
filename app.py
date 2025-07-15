from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model and column names
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
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    first_flr: float = Form(None),
    second_flr: float = Form(None),
    bedrooms: int = Form(None),
    total_rooms: int = Form(None),
    garage: float = Form(None),
    living_area: float = Form(None),
    lot_area: float = Form(None),
    quality: int = Form(None),
):
    try:
        # Determine if it's a JSON (Swagger) request
        if request.headers.get("content-type", "").startswith("application/json"):
            json_data = await request.json()
            features = HouseFeatures(**json_data)
        else:
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

        # Create DataFrame in the correct feature order
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

        if request.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse(content={"predicted_price": result})
        else:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "prediction": result
            })

    except Exception as e:
        if request.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse(status_code=500, content={"error": str(e)})
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": f"Error: {str(e)}"
        })
