import xgboost as xgb
import pandas as pd
from datetime import timedelta
from features import prepare_features_for_inference
from model_training import pipeline as data_pipeline_instance

# Model cache to store loaded XGBoost models
model_cache = {}

def load_model_for_commodity(commodity: str):
    """Load the XGBoost model for the given commodity from disk, if not already loaded."""
    key = commodity.lower().replace(" ", "_")
    if key in model_cache:
        return model_cache[key]
    model = xgb.XGBRegressor()
    model.load_model(f"models/{key}.json")
    model.n_jobs = 1  # use single thread for inference
    model_cache[key] = model
    return model

def predict_prices(commodity: str, region: str, weeks: int = 4):
    """
    Generate a rolling forecast of `weeks` weeks ahead for the given commodity and region.
    Returns a list of (date, predicted_price) tuples.
    """
    model = load_model_for_commodity(commodity)
    # Get the latest known date for this commodity-region
    latest_price_data = data_pipeline_instance.get_latest_price(commodity, region, n=1)
    latest_weather_data = data_pipeline_instance.get_latest_weather(region, n=1)
    if latest_price_data is None:
        return []
    if latest_weather_data is None:
        return []
    latest_price_date = latest_price_data.iloc[-1]['date']
    latest_weather_date = latest_weather_data.iloc[-1]['date']
    if latest_price_date and latest_weather_date: 
        current_date = max(latest_price_date, latest_weather_date) 
    elif latest_price_date: 
        current_date = latest_price_date 
    elif latest_weather_date: 
        current_date = latest_weather_date 
    else: 
        return []

    predictions = []
    # Iterate for each week to forecast
    for _ in range(weeks):
        X_next = prepare_features_for_inference(data_pipeline_instance, commodity, region, current_date)
        if X_next is None:
            break  # not enough data to continue forecasting
        # Predict the next week's price
        pred_price = float(model.predict(X_next)[0])
        next_date = current_date + timedelta(days=7)
        predictions.append((next_date.strftime("%Y-%m-%d"), pred_price))
        # Update pipeline with this predicted price as if it were new actual data
        data_pipeline_instance.add_price_data([{
            'date': next_date, 'region': region,
            'commodity': commodity, 'price': pred_price,
            'type': None  # commodity type not needed for forecasting
        }])
        current_date = next_date
    # Remove the injected predicted data from pipeline to restore state (optional/cleanup)
    for date_str, _ in predictions:
        data_pipeline_instance.price_df = data_pipeline_instance.price_df[~(
            (data_pipeline_instance.price_df['commodity'] == commodity) &
            (data_pipeline_instance.price_df['region'] == region) &
            (data_pipeline_instance.price_df['date'] == pd.to_datetime(date_str))
        )]
    return predictions
