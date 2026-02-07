# src/data_prep.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_preprocess(file):
    """
    Input: file (path or file-like from Streamlit uploader)
    Output: df_monthly (city-month aggregated), encoders (dict)
    """
    df = pd.read_csv(file)
    # Parse correct date column
    if 'Date of Occurrence' in df.columns:
        df['Date'] = pd.to_datetime(df['Date of Occurrence'], errors='coerce')
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        raise KeyError("No date column found. Expected 'Date of Occurrence' or 'Date'.")

    df = df.dropna(subset=['Date']).copy()

    # Basic extraction
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # Label encode City (we keep encoder to decode names later)
    encoders = {}
    le_city = LabelEncoder()
    df['City_enc'] = le_city.fit_transform(df['City'])
    encoders['City'] = le_city

    # Group by city-month-year
    df_monthly = df.groupby(['City_enc', 'Year', 'Month']).size().reset_index(name='Crime_Count')

    # Sort and add a synthetic date for index
    df_monthly['Date'] = pd.to_datetime(df_monthly.assign(DAY=1)[['Year', 'Month', 'DAY']])
    df_monthly = df_monthly.sort_values(['City_enc', 'Date']).reset_index(drop=True)

    # Generate per-city features: lag1, lag2, lag3, trend index, month cyclical
    df_monthly['Lag1'] = df_monthly.groupby('City_enc')['Crime_Count'].shift(1)
    df_monthly['Lag2'] = df_monthly.groupby('City_enc')['Crime_Count'].shift(2)
    df_monthly['Lag3'] = df_monthly.groupby('City_enc')['Crime_Count'].shift(3)

    # Fill NaNs in lags with 0 (or could use forward/backfill)
    df_monthly[['Lag1','Lag2','Lag3']] = df_monthly[['Lag1','Lag2','Lag3']].fillna(0)

    # Trend index: simple month number (monotonic per city)
    df_monthly['TrendIndex'] = df_monthly.groupby('City_enc').cumcount() + 1

    # Month cyclical features (sine & cosine)
    df_monthly['Month_sin'] = np.sin(2 * np.pi * df_monthly['Month']/12)
    df_monthly['Month_cos'] = np.cos(2 * np.pi * df_monthly['Month']/12)

    # Keep useful columns
    df_monthly = df_monthly[['City_enc','Date','Year','Month','Crime_Count','Lag1','Lag2','Lag3','TrendIndex','Month_sin','Month_cos']]

    # Save encoders if needed
    encoders['City_name_map'] = {int(k): v for k, v in enumerate(le_city.inverse_transform(range(len(le_city.classes_))))} if hasattr(le_city, 'classes_') else {}

    return df_monthly, encoders
