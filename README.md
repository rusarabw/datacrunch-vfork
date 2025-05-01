# Use a lightweight Python base image
FROM python:3.10-slim

# Install system dependencies (needed for XGBoost)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY deployment/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code
COPY deployment/ ./

# Expose port for FastAPI
EXPOSE 8000

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
