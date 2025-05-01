import os
import xgboost as xgb
from data_pipeline import DataPipeline
from features import prepare_training_data

WEATHER_TRAIN_DATA_PATH='data/weather_train_data.csv'
PRICE_TRAIN_DATA_PATH='data/price_train_data.csv'

# Initialize pipeline and load historical training data
pipeline = DataPipeline(price_data_path=PRICE_TRAIN_DATA_PATH, weather_data_path=WEATHER_TRAIN_DATA_PATH)

def train_from_scratch():
    os.makedirs('models', exist_ok=True)
    commodities = pipeline.price_df['commodity'].unique()

    # Train an XGBoost model for each commodity
    for commodity in commodities:
        X_df, y = prepare_training_data(pipeline.price_df, pipeline.weather_df, commodity)
        model = xgb.XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=100, max_depth=6, learning_rate=0.1,
            tree_method='hist',
            eval_metric='rmse',
            n_jobs=4
        )
        model.fit(X_df, y)

        # Save model to file (JSON format)
        fname = commodity.lower().replace(" ", "_") + ".json"
        model.save_model(os.path.join('models', fname))
        print(f"Trained model for {commodity}, saved to models/{fname}")

def train_incremental_on_weather():
    """
    Incrementally train models only when new weather data is added,
    since weather can affect multiple regions and thus multiple commodities.
    """
    os.makedirs('models', exist_ok=True)
    commodities = pipeline.price_df['commodity'].unique()

    for commodity in commodities:
        X_df, y = prepare_training_data(pipeline.price_df, pipeline.weather_df, commodity)
        if X_df.empty:
            continue

        fname = commodity.lower().replace(" ", "_") + ".json"
        model_path = os.path.join('models', fname)
        dtrain = xgb.DMatrix(X_df, label=y)

        if os.path.exists(model_path):
            booster = xgb.Booster()
            booster.load_model(model_path)
            booster = xgb.train(
                params={
                    'objective': 'reg:squarederror',
                    'learning_rate': 0.05,
                    'tree_method': 'hist',
                    'eval_metric': 'rmse'
                },
                dtrain=dtrain,
                num_boost_round=10,
                xgb_model=booster
            )
            booster.save_model(model_path)
            print(f"Incrementally updated model for {commodity} due to weather data, saved to models/{fname}")

def train_incremental_on_price(commodity: str):
    """
    Incrementally train the model for a specific commodity when new price data is added.
    """
    os.makedirs('models', exist_ok=True)
    X_df, y = prepare_training_data(pipeline.price_df, pipeline.weather_df, commodity)
    if X_df.empty:
        return

    fname = commodity.lower().replace(" ", "_") + ".json"
    model_path = os.path.join('models', fname)
    dtrain = xgb.DMatrix(X_df, label=y)

    if os.path.exists(model_path):
        booster = xgb.Booster()
        booster.load_model(model_path)
        booster = xgb.train(
            params={
                'objective': 'reg:squarederror',
                'learning_rate': 0.05,
                'tree_method': 'hist',
                'eval_metric': 'rmse'
            },
            dtrain=dtrain,
            num_boost_round=10,
            xgb_model=booster
        )
        booster.save_model(model_path)
        print(f"Incrementally updated model for {commodity} due to new price data, saved to models/{fname}")