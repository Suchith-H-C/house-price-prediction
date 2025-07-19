# monitor_drift.py

import pandas as pd
from fastapi import FastAPI
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import threading
import time

app = FastAPI()

# Define Prometheus metrics
drift_share_metric = Gauge("data_drift_share", "Share of drifted features")
drift_detected_metric = Gauge("data_drift_detected", "1 if data drift is detected, else 0")

# Function to run drift detection and update Prometheus metrics
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

# Expose the /metrics endpoint
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Background thread to periodically analyze drift every 60 seconds
def drift_monitor_loop():
    while True:
        run_drift_analysis()
        time.sleep(60)

# Start background thread
threading.Thread(target=drift_monitor_loop, daemon=True).start()
