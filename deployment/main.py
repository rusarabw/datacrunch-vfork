from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import datetime
import uvicorn
import os
import shutil
import atexit
from model_inference import predict_prices
from model_training import pipeline as data_pipeline_instance
from model_training import train_from_scratch, train_incremental_on_price, train_incremental_on_weather
from model_evaluation import evaluate_models, evaluate_model

app = FastAPI(title="Crop Price Forecast API")

class WeatherRecord(BaseModel):
    date: datetime
    region: str
    temperature: float
    rainfall: float
    humidity: float
    yield_impact: float

class PriceRecord(BaseModel):
    date: datetime
    region: str
    commodity: str
    price: float
    type: str = None

@app.post("/api/data/weather")
def ingest_weather(data: List[WeatherRecord]):
    """Ingest new weather data records."""
    records = [record.model_dump() for record in data]
    data_pipeline_instance.add_weather_data(records)
    train_incremental_on_weather()
    overall_rmse, overall_nrmse, _ = evaluate_models(data_pipeline_instance)
    if (overall_nrmse > 1.55):
        train_from_scratch()
    return {
        "status": "success", 
        "ingested": len(records),
        "rmse": overall_rmse,
        "nrmse": overall_nrmse
    }

@app.get("/api/data/weather")
def get_weather(region: str):
    """Get new weather data records."""
    results = data_pipeline_instance.get_latest_weather(region)
    weather_data = results[['date', 'region', 'temperature', 'rainfall', 'humidity', 'yield_impact']].to_dict(orient='records')
    return {"status": "success", "weather data": weather_data}

@app.post("/api/data/prices")
def ingest_prices(data: List[PriceRecord]):
    """Ingest new price data records."""
    records = [record.model_dump() for record in data]
    data_pipeline_instance.add_price_data(records)
    commodities = set(record.commodity for record in data)
    evaluations = []
    retrain = False
    for commodity in commodities:
        train_incremental_on_price(commodity)
        eval_commodity, rmse, nrmse = evaluate_model(commodity)
        if (nrmse > 1.55):
            retrain = True
        evaluations.append({
            'commodity': eval_commodity,
            'rmse': rmse,
            'nrmse': nrmse
        })
    if (retrain):
        train_from_scratch()
    return {
        "status": "success", 
        "ingested": len(records),
        "evaluations": evaluations
    }

@app.get("/api/data/price")
def get_prices(commodity:str, region: str):
    """Get new price data records."""
    results = data_pipeline_instance.get_latest_price(commodity, region)
    price_data = [{'date': date, 'region': region, 'commodity': commodity, 'price': price, 'type': type} 
                    for date, region, commodity, price, type in results]
    return {"status": "success", "price data": price_data}

@app.post("/api/predict")
def forecast_prices(commodity: str, region: str):
    """Return a rolling 4-week price forecast for the given commodity and region."""
    results = predict_prices(commodity, region, weeks=4)
    forecast = [{"date": date, "predicted_price": price} for date, price in results]
    _, rmse, nrmse = evaluate_model(commodity)
    return {
        "commodity": commodity, 
        "region": region, 
        "forecast": forecast,
        "rmse": rmse,
        "nrmse": nrmse
    }

@app.post("/api/evaluate")
def evaluate():
    """Evalute all the models and return rmse and nrmse"""
    overall_rmse, overall_nrmse, evaluations = evaluate_models(data_pipeline_instance)
    return {
        "status": "success", 
        "rmse": overall_rmse,
        "nrmse": overall_nrmse,
        "evaluations": evaluations
    }

@app.post("/api/evaluate_model")
def evaluate_single(commodity: str):
    """Evalute a model, given commodity and return rmse and nrmse"""
    _, rmse, nrmse = evaluate_model(commodity)
    return {
        "status": "success",
        "commodity": commodity, 
        "rmse": rmse,
        "nrmse": nrmse
    }

@app.post("/api/train_scratch")
def train_scratch():
    train_from_scratch()
    return {"status": "success"}

def cleanup_models(): 
    models_path = "models" 
    if os.path.exists(models_path): 
        shutil.rmtree(models_path) 
        print("Cleaned up models folder.")

atexit.register(cleanup_models)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)