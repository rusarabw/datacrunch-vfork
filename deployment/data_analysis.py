import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

WEATHER_TRAIN_DATA_PATH='data/weather_train_data.csv'
PRICE_TRAIN_DATA_PATH='data/price_train_data.csv'

# Load price and weather datasets
df_price = pd.read_csv(PRICE_TRAIN_DATA_PATH)
df_price['Date'] = pd.to_datetime(df_price['Date'])
df_price.rename(columns={'Region': 'region', 'Commodity': 'commodity', 'Date': 'date'}, inplace=True)

# Load weather data
df_weather = pd.read_csv(WEATHER_TRAIN_DATA_PATH)
df_weather['Date'] = pd.to_datetime(df_weather['Date'])
df_weather.rename(columns={'Region': 'region', 'Date': 'date'}, inplace=True)

# Merge datasets
df_price = df_price.reset_index(drop=True)
df_weather = df_weather.reset_index(drop=True)

df_merged = pd.merge(df_price, df_weather, on=['date', 'region'], how='left')
df_merged.sort_values(['region', 'commodity', 'date'], inplace=True)

# --- 1. Basic Statistics ---
print("=== General Info ===")
print(df_merged.info())
print("\n=== Missing Values ===")
print(df_merged.isnull().sum())
print("\n=== Price Statistics ===")
print(df_merged['Price per Unit (Silver Drachma/kg)'].describe())

# --- 2. Unique Combinations ---
print("\n=== Unique Regions and Commodities ===")
print(f"Regions: {df_merged['region'].nunique()}")
print(f"Commodities: {df_merged['commodity'].nunique()}")
print(f"Total (region, commodity) combinations: {df_merged.groupby(['region', 'commodity']).ngroups}")

# --- 3. Time Series Lengths Per Combination ---
time_series_lengths = df_merged.groupby(['region', 'commodity']).size().reset_index(name='count')
plt.figure(figsize=(10,6))
sns.histplot(time_series_lengths['count'], bins=30, kde=True)
plt.title('Distribution of Time Series Lengths (Region-Commodity)')
plt.xlabel('Number of Weeks')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.savefig('test/images/time_series_length_distribution.png')
plt.close()

# --- 4. Price Volatility Analysis ---
volatility = df_merged.groupby(['region', 'commodity'])['Price per Unit (Silver Drachma/kg)'].std().reset_index()
volatility.rename(columns={'Price per Unit (Silver Drachma/kg)': 'price_std'}, inplace=True)
plt.figure(figsize=(10,6))
sns.histplot(volatility['price_std'], bins=30, kde=True)
plt.title('Distribution of Price Volatility (Standard Deviation)')
plt.xlabel('Price Std Dev')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.savefig('test/images/price_volatility_distribution.png')
plt.close()

# --- 5. Sample Representative Time Series ---
sample_combinations = df_merged.groupby(['region', 'commodity']).size().sort_values(ascending=False).head(5).index.tolist()

for region, commodity in sample_combinations:
    df_subset = df_merged[(df_merged['region'] == region) & (df_merged['commodity'] == commodity)]
    plt.figure(figsize=(12,6))
    plt.plot(df_subset['date'], df_subset['Price per Unit (Silver Drachma/kg)'], label=f'{commodity} - {region}', color=color_pal[0])
    plt.title(f'Price Over Time: {commodity} in {region}')
    plt.xlabel('Date')
    plt.ylabel('Price per Unit (Silver Drachma/kg)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'test/images/sample_timeseries_{commodity}_{region}.png')
    plt.close()

# --- 6. Correlation Between Weather and Price (Example Analysis) ---
correlations = []
for region, commodity in df_merged.groupby(['region', 'commodity']).groups.keys():
    df_temp = df_merged[(df_merged['region'] == region) & (df_merged['commodity'] == commodity)]
    if len(df_temp) < 20:
        continue
    corr_temp = df_temp['Price per Unit (Silver Drachma/kg)'].corr(df_temp['Temperature (K)'])
    corr_rain = df_temp['Price per Unit (Silver Drachma/kg)'].corr(df_temp['Rainfall (mm)'])
    correlations.append({
        'region': region,
        'commodity': commodity,
        'temp_corr': corr_temp,
        'rain_corr': corr_rain
    })

correlations_df = pd.DataFrame(correlations)

print("\nTop 5 Region-Commodity pairs where Temperature most impacts Price:")
print(correlations_df.sort_values('temp_corr', key=lambda x: x.abs(), ascending=False).head(5))

print("\nTop 5 Region-Commodity pairs where Rainfall most impacts Price:")
print(correlations_df.sort_values('rain_corr', key=lambda x: x.abs(), ascending=False).head(5))

# Save all artifacts
correlations_df.to_csv('test/docs/weather_price_correlations.csv', index=False)
time_series_lengths.to_csv('test/docs/time_series_lengths.csv', index=False)
volatility.to_csv('test/docs/price_volatility.csv', index=False)

print("\n✅ In-depth EDA complete: charts and summaries saved!")
