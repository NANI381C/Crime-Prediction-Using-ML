import sys
import os

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_prep import load_and_preprocess
from src.model_lstm import train_and_forecast
from src.hotspot_cluster import generate_hotspot_map
from src.shap_explainer import explain_lstm_with_shap
from src.classifier_model import train_classifier
from src.visuals_map import build_city_geo_df, create_timestamped_geojson

import joblib
import os
from streamlit.components.v1 import html
st.write("APP ROOT DIR:", os.getcwd())
st.write("EXPECTED MAPPING PATH:", os.path.abspath("models/city_mapping.csv"))

# ---------------------------------------------
# FIX: ENABLE IMPORTING MODULES FROM src/
# ---------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import folium
from folium.plugins import TimestampedGeoJson
from streamlit.components.v1 import html

# ---------------------------------------------
# IMPORT PROJECT MODULES
# ---------------------------------------------
try:
    from src.data_prep import load_and_preprocess
    from src.model_lstm import train_and_forecast
    from src.hotspot_cluster import generate_hotspot_map
    from src.shap_explainer import explain_lstm_with_shap
    from src.classifier_model import train_classifier
    from src.visuals_map import build_city_geo_df, create_timestamped_geojson
except Exception as e:
    st.error("Failed to import modules from src/. Ensure files exist.")
    st.error(str(e))


# ---------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------
st.set_page_config(page_title="Crime Forecasting Dashboard",
                   layout="wide", page_icon="🔍")

# ---------------------------------------------
# DARK UI CSS
# ---------------------------------------------
st.markdown("""
<style>
:root {
    --bg: #0b0f14;
    --card: rgba(255,255,255,0.03);
    --accent: #34eaff;
    --muted: #9aa5b1;
}
.stApp { background: var(--bg); color: white; }
.card {
    background: var(--card);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 6px 18px rgba(0,0,0,0.6);
}
.metric {
    color: var(--accent);
    font-size: 34px;
    font-weight:800;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------
# HEADER
# ---------------------------------------------
st.markdown("<h1 style='color:white;'>🔍 Crime Forecasting Dashboard</h1>",
            unsafe_allow_html=True)

st.write("LSTM Forecasting • Hotspot Mapping • Google Maps • Classifier • SHAP Explainability")


# ---------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------
st.sidebar.title("⚙️ Controls")
uploaded = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

forecast_horizon = st.sidebar.slider("Forecast Steps (Months)", 1, 12, 3)
epochs = st.sidebar.slider("Training Epochs", 10, 200, 60)
window = st.sidebar.slider("LSTM Window", 1, 12, 3)

use_google_maps = st.sidebar.checkbox("Use Google Maps Tiles", True)

GOOGLE_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", None)
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]
except:
    pass


# ---------------------------------------------
# LOAD DATASET
# ---------------------------------------------
df_monthly = None
encoders = {}

if uploaded:
    try:
        df_monthly, encoders = load_and_preprocess(uploaded)
        st.success("Dataset Loaded Successfully 🎉")
    except Exception as e:
        st.error("Failed to load dataset:")
        st.error(str(e))

else:
    # fallback if no upload
    if os.path.exists("data/time_series_city.csv"):
        df_monthly = pd.read_csv("data/time_series_city.csv")
        st.info("Loaded default prepared dataset.")
    else:
        st.warning("Upload a dataset to start.")
        df_monthly = None


# ---------------------------------------------
# FIX: CLEAN DATE COLUMN IF NEEDED
# ---------------------------------------------
if df_monthly is not None:
    if "Year" in df_monthly.columns and "Month" in df_monthly.columns:
        try:
            df_monthly["Year"] = df_monthly["Year"].astype(int)
            df_monthly["Month"] = df_monthly["Month"].astype(int)
            df_monthly["Date"] = pd.to_datetime(
                df_monthly["Year"].astype(str)
                + "-" +
                df_monthly["Month"].astype(str).str.zfill(2)
                + "-01",
                errors="coerce"
            )
        except:
            pass


# ---------------------------------------------
# TABS
# ---------------------------------------------
tabs = st.tabs([
    "📊 Forecasting",
    "🗺 Hotspots",
    "📄 Project Info",
    "🧠 Classification",
    "📈 Multi-City",
    "🎞 Animated Heatmap"
])


# ===========================================================
# TAB 1 — FORECASTING
# ===========================================================
with tabs[0]:

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📊 LSTM Crime Forecasting")

    if df_monthly is None:
        st.warning("Upload a dataset first.")
    else:

        # ---------------------------------------------
        # FIX: Load City Mapping (Encoded → Name)
        # ---------------------------------------------
        city_map_path = os.path.join(ROOT_DIR, "models", "city_mapping.csv")


        city_col = None
        for col in ["City", "city", "CITY", "City_enc"]:
            if col in df_monthly.columns:
                city_col = col
                break

        if city_col is None:
            st.error("City column not found in dataset.")
        else:

            # ---------------------------------------------
            # If mapping exists, convert numbers to names
            # ---------------------------------------------
            if os.path.exists(city_map_path):
                st.success("City mapping loaded successfully ✔")
                mapping = pd.read_csv(city_map_path)
                mapping = mapping.rename(columns={
                    mapping.columns[0]: "CityName",
                    mapping.columns[1]: "CityCode"
                })

                # If df contains encoded numbers like 0,1,2...
                if df_monthly[city_col].dtype != object:

                    # merge encoded → city name
                    df_monthly = df_monthly.merge(
                        mapping,
                        left_on=city_col,
                        right_on="CityCode",
                        how="left"
                    )

                    city_display_col = "CityName"
                else:
                    # already names
                    city_display_col = city_col

            else:
                st.error("City mapping file missing: models/city_mapping.csv")
                city_display_col = city_col

            # ---------------------------------------------
            # Build city dropdown with names instead of numbers
            # ---------------------------------------------
            cities = sorted(df_monthly[city_display_col].dropna().unique())
            city_choice = st.selectbox("Select City", cities)

            # ---------------------------------------------
            # Convert chosen name → encoded for processing
            # ---------------------------------------------
            if os.path.exists(city_map_path):
                selected_code = int(mapping[mapping["CityName"] == city_choice]["CityCode"].values[0])
                df_city = df_monthly[df_monthly["CityCode"] == selected_code].sort_values("Date")
            else:
                df_city = df_monthly[df_monthly[city_display_col] == city_choice].sort_values("Date")

            st.write("Recent rows:")
            st.dataframe(df_city.tail(10))

            # ---------------------------------------------
            # Train & Forecast
            # ---------------------------------------------
            if st.button("🚀 Train & Forecast"):
                with st.spinner("Training LSTM..."):
                    model, scaler_X, scaler_y, metrics, future_preds = train_and_forecast(
                        df_city,
                        window=window,
                        epochs=epochs,
                        forecast_steps=forecast_horizon
                    )

                st.success("Training complete!")

                accuracy = 100 - metrics["mape"]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{accuracy:.2f}%")
                c2.metric("MAPE", f"{metrics['mape']:.2f}%")
                c3.metric("RMSE", f"{metrics['rmse']:.2f}")
                c4.metric("MAE", f"{metrics['mae']:.2f}")

                # Plot
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(metrics["test_true"], label="Actual", linewidth=2)
                ax.plot(metrics["test_pred"], label="Predicted", linewidth=2)
                ax.legend()
                st.pyplot(fig)

                # Forecast
                st.subheader("🔮 Future Forecast")
                for i, p in enumerate(future_preds):
                    st.write(f"Month +{i+1}: **{int(round(p))}** crimes")

                # Save model
                os.makedirs("models", exist_ok=True)
                try:
                    model.save("models/crime_lstm_model.keras")
                except:
                    model.save("models/crime_lstm_model.h5")

                joblib.dump(scaler_X, "models/scaler_X.joblib")
                joblib.dump(scaler_y, "models/scaler_y.joblib")

    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================
# TAB 2 — HOTSPOTS
# ===========================================================
with tabs[1]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🗺 Crime Hotspot Map")

    if df_monthly is not None:
        if st.button("Generate Hotspot Map"):

            with st.spinner("Generating map..."):

                m = generate_hotspot_map(df_monthly, encoders)

                # Optional: Replace tiles with Google Maps
                if use_google_maps and GOOGLE_API_KEY:
                    m = folium.Map(
                        location=[20.5937, 78.9629],
                        zoom_start=5,
                        tiles=f"https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}&key={GOOGLE_API_KEY}",
                        attr="Google"
                    )

                m.save("map.html")
                html(open("map.html").read(), height=600)

    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================
# TAB 3 — PROJECT INFO
# ===========================================================
with tabs[2]:

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📄 Project Overview")

    st.write("""
    **Key Features**
    - LSTM Time-Series Crime Forecasting  
    - SHAP Explainability  
    - Hotspot Detection  
    - Crime Classification  
    - Multi-City Trend Comparison  
    - Animated Crime Maps  
    """)

    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================
# TAB 4 — CLASSIFICATION
# ===========================================================
with tabs[3]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🧠 Crime Classification")

    if uploaded:
        if st.button("Train Classifier"):
            uploaded.seek(0)
            raw = pd.read_csv(uploaded)

            with st.spinner("Training..."):
                metrics = train_classifier(raw)

            st.success("Classifier Trained!")
            st.write(metrics["accuracy"])
            st.json(metrics["report"])

    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================
# TAB 5 — MULTI CITY COMPARISON
# ===========================================================
with tabs[4]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📈 Multi-City Comparison")

    if df_monthly is not None:

        city_col = "City" if "City" in df_monthly.columns else df_monthly.columns[0]
        cities = sorted(df_monthly[city_col].unique())

        chosen = st.multiselect("Select Cities", cities, default=cities[:3])

        if chosen:
            df_sel = df_monthly[df_monthly[city_col].isin(chosen)]
            pivot = df_sel.pivot_table(index="Date",
                                       columns=city_col,
                                       values="Crime_Count")
            st.line_chart(pivot)

    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================
# TAB 6 — ANIMATED HEATMAP
# ===========================================================
with tabs[5]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🎞 Animated Crime Heatmap")

    if df_monthly is not None:
        if st.button("Generate Animated Heatmap"):
            with st.spinner("Generating Map..."):
                geo = build_city_geo_df(df_monthly, encoders)
                geojson = create_timestamped_geojson(df_monthly, geo)

                m = folium.Map(location=[20.5937, 78.9629],
                               zoom_start=5,
                               tiles="CartoDB dark_matter")

                TimestampedGeoJson(geojson,
                                   transition_time=700,
                                   auto_play=True).add_to(m)

                m.save("animated_map.html")
                html(open("animated_map.html").read(), height=650)

    st.markdown("</div>", unsafe_allow_html=True)


# END OF FILE
