# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Generate data + start FastAPI server
CMD ["sh", "-c", "python generate_drift_data.py && uvicorn app:app --host 0.0.0.0 --port 8000"]
