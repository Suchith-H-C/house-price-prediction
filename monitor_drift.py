import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# ✅ Ensure reports directory exists
os.makedirs("reports", exist_ok=True)

# Load reference and current data
reference = pd.read_csv("data/reference.csv")  # Sample from training data
current = pd.read_csv("data/current.csv")      # Simulate recent data

# Generate drift report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)

# ✅ Save the report
report.save_html("reports/drift_report.html")

print("✅ Drift report generated: reports/drift_report.html")
