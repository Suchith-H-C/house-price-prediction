import os
import pandas as pd

# Ensure required folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ✅ Corrected processed file path
processed_path = "data/processed/train_processed.csv"

# Check if processed data file exists
if not os.path.exists(processed_path):
    print(f"❌ Error: '{processed_path}' not found. Please run the preprocessing pipeline first.")
    exit(1)

# Load processed dataset
df = pd.read_csv(processed_path)

# Create reference and current datasets
reference_path = "data/reference.csv"
current_path = "data/current.csv"

# Save drift datasets
df.head(500).to_csv(reference_path, index=False)
df.sample(500, random_state=42).to_csv(current_path, index=False)

print("✅ Drift data generated:")
print(f"   - Reference: {reference_path}")
print(f"   - Current  : {current_path}")
