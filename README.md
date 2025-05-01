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

## ⚙️ Docker Setup Instructions

### Prerequisites
Make sure Docker is installed and running on your machine.

### 1️⃣ Build the Docker Image
```bash
cd deployment
docker build -t agrochill-forecast .
```

### 2️⃣ Run the Docker Container
```bash
docker run -d -p 8000:8000 --name agrochill-container agrochill-forecast
```

### 3️⃣ Access the FastAPI Application
Once the container is running, open your browser and go to:
```
http://localhost:8000/docs
```
Use the Swagger UI to test API endpoints:
- `POST /api/data/weather` → Ingest new weather data
- `POST /api/data/prices` → Ingest new price data
- `POST /api/predict` → Get a 4-week rolling forecast

### 🛠 Useful Docker Commands
- Check Docker images:
```bash
docker images
```
- List running containers:
```bash
docker ps
```
- Stop a container:
```bash
docker stop agrochill-container
```
- Remove a container:
```bash
docker rm agrochill-container
```
- Remove an image:
```bash
docker rmi agrochill-forecast
```

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

If you need a CI/CD pipeline setup or a Docker Compose example, let me know!

