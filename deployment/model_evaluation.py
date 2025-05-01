import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from data_pipeline import DataPipeline
from features import prepare_training_data

WEATHER_EVAL_DATA_PATH='data/weather_eval_data.csv'
PRICE_EVAL_DATA_PATH='data/price_eval_data.csv'

# Function to load model
model_cache = {}
def load_model(commodity: str):
    key = commodity.lower().replace(" ", "_")
    if key in model_cache:
        return model_cache[key]
    model = xgb.XGBRegressor()
    model.load_model(f"models/{key}.json")
    model.n_jobs = 1
    model_cache[key] = model
    return model

def evaluate_models(pipeline_train: DataPipeline):
    """ Evaluates all models using the evaluation dataset.""" 
    pipeline_eval = DataPipeline(price_data_path=PRICE_EVAL_DATA_PATH, weather_data_path=WEATHER_EVAL_DATA_PATH)
    evaluations = []
    all_y_true = []
    all_y_pred = []

    for commodity in pipeline_train.price_df['commodity'].unique():
        X_eval, y_true = prepare_training_data(
            pipeline_eval.price_df, pipeline_eval.weather_df, commodity
        )

        if len(X_eval) == 0:
            print(f"Skipping {commodity}: not enough evaluation data.")
            continue

        model = load_model(commodity)
        y_pred = model.predict(X_eval)

        rmse = root_mean_squared_error(y_true, y_pred)
        range_y = np.max(y_true) - np.min(y_true)
        nrmse = 0.0 if range_y == 0 else rmse / range_y

        evaluations.append({
            'commodity': commodity,
            'rmse': rmse,
            'nrmse': nrmse
        })

        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(y_pred.tolist())

    overall_rmse = root_mean_squared_error(all_y_true, all_y_pred)
    range_all = np.max(all_y_true) - np.min(all_y_true)
    overall_nrmse = 0.0 if range_all == 0 else overall_rmse / range_all

    return overall_rmse, overall_nrmse, evaluations

def evaluate_model(commodity: str): 
    """Evaluates a single commodity model using the evaluation dataset.""" 
    pipeline_eval = DataPipeline(price_data_path=PRICE_EVAL_DATA_PATH, weather_data_path=WEATHER_EVAL_DATA_PATH)
    X_eval, y_true = prepare_training_data(pipeline_eval.price_df, pipeline_eval.weather_df, commodity)

    if len(X_eval) == 0:
        print(f"Skipping {commodity}: not enough evaluation data.")
        return None

    model = load_model(commodity)
    y_pred = model.predict(X_eval)

    rmse = root_mean_squared_error(y_true, y_pred)
    range_y = np.max(y_true) - np.min(y_true)
    nrmse = 0.0 if range_y == 0 else rmse / range_y

    print(f"Evaluation for {commodity} - RMSE: {rmse:.2f}, NRMSE: {nrmse:.4f}")

    return commodity, rmse, nrmse