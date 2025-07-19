# main.py

import pandas as pd
import joblib
import threading
import time
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

app = FastAPI()
Instrumentator().instrument(app).expose(app)

templates = Jinja2Templates(directory="templates")

# Load model and columns
model = joblib.load("model.joblib")
columns = joblib.load("columns.pkl")

# 📊 Prometheus metrics
drift_share_metric = Gauge("data_drift_share", "Share of drifted features")
drift_detected_metric = Gauge("data_drift_detected", "1 if data drift is detected, else 0")

# 📈 Drift monitoring logic
def run_drift_analysis():
    try:
        reference = pd.read_csv("data/reference.csv")
        current = pd.read_csv("data/current.csv")

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        result = report.as_dict()

        drift_result = result["metrics"][0]["result"]
        drift_share = drift_result.get("share_drifted", 0)
        drift_detected = 1 if drift_result.get("dataset_drift", False) else 0

        drift_share_metric.set(drift_share)
        drift_detected_metric.set(drift_detected)

        print(f"✅ Drift analyzed: share={drift_share}, detected={bool(drift_detected)}")

    except Exception as e:
        print(f"❌ Error in drift analysis: {e}")

def drift_monitor_loop():
    while True:
        run_drift_analysis()
        time.sleep(60)

# Start the drift monitoring thread
threading.Thread(target=drift_monitor_loop, daemon=True).start()

# 🏠 Home route (form-based UI)
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 🔮 HTML form-based prediction
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

# 🔌 API prediction via JSON
class HouseFeatures(BaseModel):
    first_flr: float
    second_flr: float
    bedrooms: int
    total_rooms: int
    garage: float
    living_area: float
    lot_area: float
    quality: int

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

# 📊 Expose /metrics endpoint for Prometheus
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
