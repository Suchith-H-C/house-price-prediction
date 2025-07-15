from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the trained model
model = joblib.load("model.joblib")

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            OverallQual: int = Form(...),
            GrLivArea: float = Form(...),
            GarageCars: int = Form(...),
            TotalBsmtSF: float = Form(...),
            FullBath: int = Form(...),
            YearBuilt: int = Form(...)):

    # Example input order should match the trained model
    input_data = pd.DataFrame([{
        "OverallQual": OverallQual,
        "GrLivArea": GrLivArea,
        "GarageCars": GarageCars,
        "TotalBsmtSF": TotalBsmtSF,
        "FullBath": FullBath,
        "YearBuilt": YearBuilt
    }])

    prediction = model.predict(input_data)[0]
    return templates.TemplateResponse("index.html", {"request": request, "result": f"${prediction:,.2f}"})
