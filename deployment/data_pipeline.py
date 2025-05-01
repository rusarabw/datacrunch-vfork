import pandas as pd

class DataPipeline:
    """
    Stores historical weather and price data
    Ingest new, incoming weather and price data
    """
    def __init__(self, price_data_path: str, weather_data_path: str):
        # load data
        self.weather_df = pd.read_csv(weather_data_path)
        self.price_df = pd.read_csv(price_data_path)

        # rename columns
        self.weather_df.rename(columns={
            'Date': 'date',
            'Region': 'region',
            'Temperature (K)': 'temperature',
            'Rainfall (mm)': 'rainfall',
            'Humidity (%)': 'humidity',
            'Crop Yield Impact Score': 'yield_impact'
        }, inplace=True)
        self.price_df.rename(columns={
            'Date': 'date',
            'Region': 'region',
            'Commodity': 'commodity',
            'Price per Unit (Silver Drachma/kg)': 'price',
            'Type': 'type'
        }, inplace=True)

        # convert date strings to datetime
        self.weather_df['date'] = pd.to_datetime(self.weather_df['date'])
        self.price_df['date'] = pd.to_datetime(self.price_df['date'])

        # sort data by commodity/region/date for consistent order
        self.weather_df.sort_values(['region', 'date'], inplace=True)
        self.price_df.sort_values(['commodity', 'region', 'date'], inplace=True)

        # Reset indices
        self.weather_df.reset_index(drop=True, inplace=True)
        self.price_df.reset_index(drop=True, inplace=True)

    def add_weather_data(self, new_data):
        """
        Ingest new weather data. 
        `new_data`: list of dictionaries/ DataFrame with fields: date, region, temperature, rainfall, humidity, yield_impact.
        """
        df_new = pd.DataFrame(new_data) if isinstance(new_data, list) else new_data
        df_new.rename(columns={
            'Date': 'date',
            'Region': 'region',
            'Temperature (K)': 'temperature',
            'Rainfall (mm)': 'rainfall',
            'Humidity (%)': 'humidity',
            'Crop Yield Impact Score': 'yield_impact'
        }, inplace=True)
        df_new['date'] = pd.to_datetime(df_new['date'])
        self.weather_df = pd.concat([self.weather_df, df_new], ignore_index=True)
        self.weather_df.sort_values(['region', 'date'], inplace=True, ignore_index=True)

    def add_price_data(self, new_data):
        """
        Ingest new price data. 
        `new_data`: list of dictionaries/ DataFrame with fields: date, region, commodity, price, type.
        """
        df_new = pd.DataFrame(new_data) if isinstance(new_data, list) else new_data
        df_new.rename(columns={
            'Date': 'date',
            'Region': 'region',
            'Commodity': 'commodity',
            'Price per Unit (Silver Drachma/kg)': 'price',
            'Type': 'type'
        }, inplace=True)
        df_new['date'] = pd.to_datetime(df_new['date'])
        self.price_df = pd.concat([self.price_df, df_new], ignore_index=True)
        self.price_df.sort_values(['commodity', 'region', 'date'], inplace=True, ignore_index=True)

    def get_weather_on_date(self, region: str, date: pd.Timestamp):
        """
        Get weather features for a given region and date, or None if not available.
        """
        recs = self.weather_df[(self.weather_df['region'] == region) & (self.weather_df['date'] == date)]
        if recs.empty:
            return None
        rec = recs.iloc[0]
        return {
            'temperature': rec['temperature'],
            'rainfall': rec['rainfall'],
            'humidity': rec['humidity'],
            'yield_impact': rec['yield_impact']
        }
    
    def get_latest_weather(self, region: str, n: int = 1):
        """
        Return the last n weather records for a given region.
        """
        subset = self.weather_df[self.weather_df['region'] == region]
        if subset.empty:
            return None
        return subset.tail(n)

    def get_latest_price(self, commodity: str, region: str, n: int = 1):
        """
        Return the last n price records for a given commodity and region.
        """
        subset = self.price_df[(self.price_df['commodity'] == commodity) & (self.price_df['region'] == region)]
        if subset.empty:
            return None
        return subset.tail(n)