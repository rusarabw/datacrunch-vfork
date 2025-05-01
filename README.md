# AgroChill Crop Price Forecasting System

Welcome to the AgroChill crop price forecasting system! This repository contains the complete implementation of a modular, scalable, and containerized machine learning pipeline designed to predict weekly crop prices four weeks ahead, leveraging weather and market data.

## 🚀 Overview
This system includes:
- Data ingestion (via REST API)
- Feature engineering
- Per-commodity XGBoost models
- Rolling 4-week forecasts
- FastAPI service for real-time predictions
- Docker container for easy deployment

## 📦 Repository Structure
```
project-root/
├── deployment/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                # FastAPI server entrypoint
│   ├── model_inference.py     # Model loading & prediction logic
│   ├── data_pipeline.py       # Data handling
│   ├── features.py            # Feature engineering
│   ├── models/                # Saved XGBoost models (one per commodity)
│   └── data/                  # Initial training data (optional)
├── image_name.txt             # DockerHub image URI
├── Documentation.pdf          # Technical write-up
├── Presentation/              # Presentation materials
└── README.md                  # This file
```

## ⚙️ Setup Instructions

### 1️⃣ Build the Docker Image
```bash
docker build -t agrochill-forecast ./deployment
```

### 2️⃣ Run the Container
```bash
docker run -p 8000:8000 agrochill-forecast
```

### 3️⃣ Access the API
Once running, the FastAPI server will be available at:
```
http://localhost:8000/docs
```
This provides an interactive Swagger UI to test endpoints:
- `POST /api/data/weather` → Ingest new weather data
- `POST /api/data/prices` → Ingest new price data
- `POST /api/predict` → Request 4-week rolling price forecasts

## 📊 Features
✅ Per-commodity XGBoost models (high accuracy, low latency)  
✅ Rolling multi-step predictions using latest data  
✅ Lightweight CPU-only deployment  
✅ Modular, maintainable Python codebase  
✅ Automated evaluation with RMSE and NRMSE metrics  
✅ Dockerized for easy shipping and scaling

## 🔍 How to Contribute
If you want to extend or improve this system:
1. Clone the repository
2. Create a virtual environment
3. Install dependencies from `requirements.txt`
4. Modify code inside the `deployment/` folder
5. Rebuild the Docker image and test

## 📄 License
This project is for educational and competition use. Please review licensing details before using it in commercial applications.

## 🤝 Acknowledgments
Thanks to the AgroChill team and DataCrunch organizers for the problem statement and dataset.

---

If you need a badge section or GitHub Actions CI setup, let me know and I can generate that too! 🚀

