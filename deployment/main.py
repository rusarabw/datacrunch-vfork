from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date as pd_date
import uvicorn
import os
import shutil
import atexit

from model_inference import predict_prices
from model_training import pipeline as data_pipeline_instance
from model_training import train_from_scratch, train_incremental_on_price, train_incremental_on_weather
from model_evaluation import evaluate_models, evaluate_model

app = FastAPI(title="Crop Price Forecast API")


class WeatherData(BaseModel):
    rainfall: Optional[float] = Field(None, example=5.2)
    humidity: Optional[float] = Field(None, example=78.3)
    temp: Optional[float] = Field(None, example=29.4)
    yield_impact: Optional[float] = Field(default=None)

class WeatherRecordInput(BaseModel):
    date: pd_date = Field(..., example="2025-04-16")
    region: str = Field(..., example="Valhalla")
    weatherData: WeatherData

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2025-04-16",
                "region": "Valhalla",
                "weatherData": {
                    "rainfall": 5.2,
                    "humidity": 78.3,
                    "temp": 29.4
                }
            }
        }

class PriceData(BaseModel):
    price: float = Field(..., example=86.4)

class PriceRecordInput(BaseModel):
    date: pd_date = Field(..., example="2025-04-16")
    crop: str = Field(..., example="Cantaloupe")
    region: str = Field(..., example="Valhalla")
    type: Optional[str] = Field(default=None)
    priceData: PriceData

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2025-04-16",
                "crop": "Cantaloupe",
                "region": "Valhalla",
                "priceData": {"price": 86.4}
            }
        }

class PredictionRequest(BaseModel):
    crop: str = Field(..., example="Cantaloupe")
    region: str = Field(..., example="Valhalla")

@app.post("/api/data/weather", responses={
    200: {"description": "Weather data stored successfully"},
    400: {"description": "Missing required fields"},
    422: {"description": "Invalid JSON schema"}
})
def ingest_weather(data: List[WeatherRecordInput]):
    try:
        weather_records = []
        for entry in data:
            record = {
                "date": entry.date,
                "region": entry.region,
                "temperature": entry.weatherData.temp,
                "rainfall": entry.weatherData.rainfall,
                "humidity": entry.weatherData.humidity,
                "yield_impact": entry.weatherData.yield_impact
            }
            weather_records.append(record)

        data_pipeline_instance.add_weather_data(weather_records)
        train_incremental_on_weather()
        overall_rmse, overall_nrmse, _ = evaluate_models(data_pipeline_instance)
        if overall_nrmse > 1.55:
            train_from_scratch()

        return {
            "status": "success",
            "ingested": len(weather_records),
            "rmse": overall_rmse,
            "nrmse": overall_nrmse
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/data/weather")
def get_weather(region: str):
    """Get new weather data records."""
    results = data_pipeline_instance.get_latest_weather(region)
    weather_data = results[['date', 'region', 'temperature', 'rainfall', 'humidity', 'yield_impact']].to_dict(orient='records')
    return {"status": "success", "weather data": weather_data}

@app.post("/api/data/prices", responses={
    200: {"description": "Price data stored successfully"},
    400: {"description": "Missing required fields"},
    422: {"description": "Invalid JSON schema"}
})
def ingest_prices(data: List[PriceRecordInput]):
    try:
        price_records = []
        for entry in data:
            record = {
                "date": entry.date,
                "commodity": entry.crop,
                "region": entry.region,
                "price": entry.priceData.price,
                "type": entry.type
            }
            price_records.append(record)

        data_pipeline_instance.add_price_data(price_records)
        commodities = set(record["commodity"] for record in price_records)
        evaluations = []
        retrain = False
        for commodity in commodities:
            train_incremental_on_price(commodity)
            eval_commodity, rmse, nrmse = evaluate_model(commodity)
            if nrmse > 1.55:
                retrain = True
            evaluations.append({
                'commodity': eval_commodity,
                'rmse': rmse,
                'nrmse': nrmse
            })
        if retrain:
            train_from_scratch()

        return {
            "status": "success",
            "ingested": len(price_records),
            "evaluations": evaluations
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/data/price")
def get_prices(commodity:str, region: str):
    """Get new price data records."""
    results = data_pipeline_instance.get_latest_price(commodity, region)
    price_data = [{'date': date, 'region': region, 'commodity': commodity, 'price': price, 'type': type} 
                    for date, region, commodity, price, type in results]
    return {"status": "success", "price data": price_data}


@app.post("/api/predict", responses={
    200: {
        "description": "Price predictions for the specified crop and region",
        "content": {
            "application/json": {
                "example": {
                    "crop": "Cantaloupe",
                    "region": "Valhalla",
                    "predictions": [
                        {"prediction_index": 0, "date": "2025-04-18", "price": 85.2}
                    ]
                }
            }
        }
    },
    400: {"description": "Missing required fields or invalid crop/region"},
    422: {"description": "Invalid request format"}
})
def forecast_prices(payload: PredictionRequest):
    try:
        results = predict_prices(payload.crop, payload.region, weeks=4)
        forecast = [
            {"prediction_index": i, "date": date, "price": price} 
            for i, (date, price) in enumerate(results)
        ]
        return {
            "crop": payload.crop,
            "region": payload.region,
            "predictions": forecast
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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