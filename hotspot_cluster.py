# src/hotspot_cluster.py
from sklearn.cluster import KMeans
import folium
from geopy.geocoders import Nominatim
import time

def generate_hotspot_map(df_monthly, encoders, n_clusters=5):
    """
    df_monthly: aggregated monthly df (City_enc, Date, Crime_Count, ...)
    encoders: dict returned from data_prep (to decode city names)
    """
    df_mean = df_monthly.groupby('City_enc')['Crime_Count'].mean().reset_index()
    # decode names
    le = encoders.get('City', None)
    if le is not None:
        df_mean['City'] = le.inverse_transform(df_mean['City_enc'].astype(int))
    else:
        df_mean['City'] = df_mean['City_enc'].astype(str)

    # geocode each city (cache if you plan to re-run)
    geolocator = Nominatim(user_agent="crime_hotspot_app", timeout=10)
    lats, lons = [], []
    for city in df_mean['City']:
        try:
            loc = geolocator.geocode(f"{city}, India")
            if loc:
                lats.append(loc.latitude)
                lons.append(loc.longitude)
            else:
                lats.append(None)
                lons.append(None)
        except Exception:
            lats.append(None)
            lons.append(None)
        time.sleep(0.5)

    df_mean['lat'] = lats
    df_mean['lon'] = lons
    df_mean = df_mean.dropna(subset=['lat','lon'])

    if df_mean.empty:
        return None

    kmeans = KMeans(n_clusters=min(n_clusters, len(df_mean)), random_state=42)
    df_mean['cluster'] = kmeans.fit_predict(df_mean[['Crime_Count']])

    m = folium.Map(location=[20.5937,78.9629], zoom_start=5, tiles='CartoDB dark_matter')
    colors = ['red','orange','yellow','green','blue','purple','cadetblue']

    for _, r in df_mean.iterrows():
        folium.CircleMarker(
            location=[r['lat'], r['lon']],
            radius=6 + (r['Crime_Count'] / (df_mean['Crime_Count'].max()+1))*6,
            color=colors[int(r['cluster'])%len(colors)],
            fill=True,
            popup=f"{r['City']}: Avg {r['Crime_Count']:.1f}"
        ).add_to(m)
    return m
