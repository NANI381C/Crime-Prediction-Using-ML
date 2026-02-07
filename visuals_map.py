# visuals_map.py
import pandas as pd
import numpy as np
import folium
import os
from folium.plugins import TimestampedGeoJson

# ------------------------------------------------------------
# FIX: Always save cache inside project-level /data folder
# ------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

CACHE_FILE = os.path.join(DATA_DIR, "city_geo_cache.csv")


# ------------------------------------------------------------
# Detect city column safely (City, CityName, City_enc etc.)
# ------------------------------------------------------------
def detect_city_column(df):
    possible_cols = ["City", "CityName", "city", "CITY", "City_enc"]

    for col in possible_cols:
        if col in df.columns:
            return col

    raise KeyError(f"No city column found. Available columns: {list(df.columns)}")


# ------------------------------------------------------------
# GEO DATAFRAME BUILDER
# ------------------------------------------------------------
def build_city_geo_df(df_monthly, encoders):
    """
    Build geo dataframe with:
    city → lat, lon
    Always returns: City, lat, lon
    """

    city_col = detect_city_column(df_monthly)

    # If encoded → convert using LabelEncoder
    if city_col == "City_enc":
        if "City" in encoders:
            le = encoders["City"]
            df_monthly["City"] = le.inverse_transform(df_monthly["City_enc"].astype(int))
            city_col = "City"
        else:
            raise KeyError("City_enc found but no encoder available to decode city names.")

    # If CityName → rename to City
    if city_col == "CityName":
        df_monthly["City"] = df_monthly["CityName"]
        city_col = "City"

    # Static coordinates for Indian cities
    city_coords = {
        "Agra": (27.1767, 78.0081),
        "Ahmedabad": (23.0225, 72.5714),
        "Bangalore": (12.9716, 77.5946),
        "Bhopal": (23.2599, 77.4126),
        "Chennai": (13.0827, 80.2707),
        "Delhi": (28.7041, 77.1025),
        "Faridabad": (28.4089, 77.3178),
        "Ghaziabad": (28.6692, 77.4538),
        "Hyderabad": (17.3850, 78.4867),
        "Indore": (22.7196, 75.8577),
        "Jaipur": (26.9124, 75.7873),
        "Kalyan": (19.2403, 73.1305),
        "Kanpur": (26.4499, 80.3319),
        "Kolkata": (22.5726, 88.3639),
        "Lucknow": (26.8467, 80.9462),
        "Ludhiana": (30.9000, 75.8573),
        "Meerut": (28.9845, 77.7064),
        "Mumbai": (19.0760, 72.8777),
        "Nagpur": (21.1458, 79.0882),
        "Nashik": (19.9975, 73.7898),
        "Patna": (25.5941, 85.1376),
        "Pune": (18.5204, 73.8567),
        "Rajkot": (22.3039, 70.8022),
        "Srinagar": (34.0837, 74.7973),
        "Surat": (21.1702, 72.8311),
        "Thane": (19.2183, 72.9781),
        "Varanasi": (25.3176, 82.9739),
        "Vasai": (19.3919, 72.8397),
        "Visakhapatnam": (17.6868, 83.2185)
    }

    geo_data = []

    for _, row in df_monthly.iterrows():
        city = row[city_col]

        if city in city_coords:
            lat, lon = city_coords[city]
            geo_data.append([city, lat, lon])

    geo_df = pd.DataFrame(geo_data, columns=["City", "lat", "lon"])

    # Save cache
    geo_df.to_csv(CACHE_FILE, index=False)

    return geo_df


# ------------------------------------------------------------
# ANIMATED HEATMAP GEOJSON BUILDER
# ------------------------------------------------------------
def create_timestamped_geojson(df_monthly, geo_df):
    """
    Generate a valid GeoJSON for TimestampedGeoJson.
    Ensures:
    - flat feature list (NOT nested lists)
    - ISO formatted timestamps
    """

    # Detect city column
    city_col = detect_city_column(df_monthly)

    # Normalize column name
    if city_col in ["CityName"]:
        df_monthly["City"] = df_monthly["CityName"]
    elif city_col == "City_enc":
        df_monthly["City"] = df_monthly["City_enc"].astype(int)
    else:
        df_monthly["City"] = df_monthly[city_col]

    # Merge with coordinates
    merged = df_monthly.merge(geo_df, on="City", how="left")

    # Fix date column
    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged = merged.dropna(subset=["Date"])

    # Convert to ISO format
    merged["time_str"] = merged["Date"].dt.strftime("%Y-%m-%d")
    # ----------------------------------------------------
    # PERFORMANCE FIX: Reduce dataset for animation
    # ----------------------------------------------------
    if len(merged) > 500:   # only reduce large datasets
        merged = merged.sample(frac=0.25, random_state=42)


    features = []

    # Build flat list of features
    for _, row in merged.iterrows():

        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["lon"], row["lat"]],
            },
            "properties": {
                "time": row["time_str"],          # REQUIRED
                "popup": f"{row['City']} — Crimes: {row['Crime_Count']}",
                "crime_count": int(row["Crime_Count"])
            },
        }

        features.append(feature)

    # FINAL flat GeoJSON
    return {
        "type": "FeatureCollection",
        "features": features
    }
