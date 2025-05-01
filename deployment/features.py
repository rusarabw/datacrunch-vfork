import pandas as pd

def prepare_training_data(price_df: pd.DataFrame, weather_df: pd.DataFrame, commodity: str, num_lags: int = 4):
    """
    Prepare supervised learning features and target for a given commodity.
    Returns X_df (features) and y_series (target prices).
    """
    # filter data for this commodity
    df_price = price_df[price_df['commodity'] == commodity].copy()

    # merge with weather on date and region (attach weather features to each price record)
    df_merged = pd.merge(df_price, weather_df, on=['date', 'region'], how='left')
    df_merged.sort_values(['region', 'date'], inplace=True)

    # create lag features for price (1 to 4 weeks back) within each region
    for lag in range(1, num_lags+1):
        df_merged[f'price_lag{lag}'] = df_merged.groupby('region')['price'].shift(lag)

    # Create lag-1 features for weather variables (previous period's weather)
    for col in ['temperature', 'rainfall', 'humidity', 'yield_impact']:
        df_merged[f'{col}_lag1'] = df_merged.groupby('region')[col].shift(1)

    # Drop rows with any NaN in required lag features (e.g., first records of each series)
    lag_cols = [f'price_lag{i}' for i in range(1, num_lags+1)]
    weather_lag_cols = [f'{col}_lag1' for col in ['temperature','rainfall','humidity','yield_impact']]
    df_lags = df_merged.dropna(subset=lag_cols + weather_lag_cols)

    # Time features: month and year
    df_lags['month'] = df_lags['date'].dt.month
    df_lags['year'] = df_lags['date'].dt.year

    # One-hot encode region and month (include all categories to be consistent)
    all_regions = sorted(price_df['region'].unique())
    df_lags['region'] = pd.Categorical(df_lags['region'], categories=all_regions)
    region_dummies = pd.get_dummies(df_lags['region'], prefix='region')
    all_months = list(range(1, 13))
    df_lags['month'] = pd.Categorical(df_lags['month'], categories=all_months)
    month_dummies = pd.get_dummies(df_lags['month'], prefix='month')

    # Add dummy columns to dataframe
    df_lags = pd.concat([df_lags, region_dummies, month_dummies], axis=1)

    # Define feature set X and target y
    drop_cols = ['date', 'commodity', 'type', 'region', 'temperature', 'rainfall', 'humidity', 'yield_impact', 'month']

    # We drop the original current-period weather columns and non-feature columns
    X_df = df_lags.drop(columns=drop_cols + ['price'])
    y_series = df_lags['price']
    return X_df, y_series

def prepare_features_for_inference(pipeline, commodity: str, region: str, current_date):
    """
    Create a feature vector for predicting the next week's price for the given commodity and region.
    `current_date` is the last date we have data for.
    Returns a DataFrame (single row) of features.
    """
    num_lags = 4
    recent_prices = pipeline.get_latest_price(commodity, region, n=num_lags)
    if recent_prices is None or len(recent_prices) < num_lags:
        # Not enough history to forecast
        return None
    recent_prices = recent_prices.sort_values('date')

    # Prepare price lag features from the last known prices
    price_values = recent_prices['price'].tolist()[-num_lags:]
    feats = {}

    # price_lag1 = most recent price, price_lag4 = 4th most recent
    for i in range(1, num_lags+1):
        feats[f'price_lag{i}'] = price_values[-i] if i <= len(price_values) else None

    # Weather lag features: use last known weather (at current_date) as the previous period's weather
    last_weather = pipeline.get_weather_on_date(region, current_date)
    if last_weather is None:
        # If no weather exactly on current_date, use the latest available before that date
        wdf_region = pipeline.weather_df[(pipeline.weather_df['region'] == region) & 
                                         (pipeline.weather_df['date'] <= current_date)]
        if not wdf_region.empty:
            last_date = wdf_region['date'].max()
            last_weather = pipeline.get_weather_on_date(region, last_date)

    if last_weather:
        feats['temperature_lag1'] = last_weather['temperature']
        feats['rainfall_lag1']   = last_weather['rainfall']
        feats['humidity_lag1']   = last_weather['humidity']
        feats['yield_impact_lag1'] = last_weather['yield_impact']
    else:
        # If no weather data at all, default to neutral values (could also skip weather influence)
        feats['temperature_lag1'] = 0.0
        feats['rainfall_lag1']   = 0.0
        feats['humidity_lag1']   = 0.0
        feats['yield_impact_lag1'] = 0.0

    # Time features for the next week to forecast
    next_date = current_date + pd.Timedelta(days=7)  # assuming weekly interval ~7 days
    feats['year'] = next_date.year
    month_val = next_date.month

    # Region one-hot encoding
    all_regions = sorted(pipeline.price_df['region'].unique())
    for reg in all_regions:
        feats[f'region_{reg}'] = 1 if reg == region else 0

    # Month one-hot encoding
    for m in range(1, 13):
        feats[f'month_{m}'] = 1 if m == month_val else 0
        
    # Convert features dict to DataFrame
    X_next = pd.DataFrame([feats])
    return X_next
