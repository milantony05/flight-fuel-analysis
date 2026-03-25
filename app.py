"""
✈️ Flight Fuel Analysis — Real-Time Streamlit Dashboard
Predicts fuel flow and CO₂ emissions for live global flights.
"""

import json
import time
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import joblib


# ──────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Fuel Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────────────────────────────
# Custom CSS — Dark Premium Theme
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #121929 50%, #0d1321 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 23, 0.95);
        border-right: 1px solid rgba(0, 212, 170, 0.1);
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(20, 30, 50, 0.8), rgba(15, 22, 40, 0.9));
        border: 1px solid rgba(0, 212, 170, 0.15);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .metric-card:hover {
        border-color: rgba(0, 212, 170, 0.4);
        box-shadow: 0 8px 30px rgba(0, 212, 170, 0.1);
        transform: translateY(-2px);
    }
    .metric-icon {
        font-size: 32px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4aa, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 4px 0;
    }
    .metric-value.warn {
        background: linear-gradient(135deg, #ff6b6b, #ffd93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #8892a4;
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Section headers */
    .section-header {
        color: #e0e6f0;
        font-size: 22px;
        font-weight: 600;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(0, 212, 170, 0.3);
    }

    /* Title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4aa 0%, #4ecdc4 50%, #00b4d8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 4px;
    }
    .sub-title {
        color: #8892a4;
        font-size: 16px;
        text-align: center;
        margin-bottom: 24px;
        font-weight: 400;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        background: rgba(0, 212, 170, 0.15);
        color: #00d4aa;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        border: 1px solid rgba(0, 212, 170, 0.3);
    }

    /* Data table styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Plotly chart backgrounds */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Remove white backgrounds from st elements */
    [data-testid="stMetricValue"] {
        color: #00d4aa !important;
    }

    div[data-testid="stExpander"] {
        background: rgba(20, 30, 50, 0.5);
        border: 1px solid rgba(0, 212, 170, 0.1);
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Data Functions
# ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_live_flights(n_sample=50):
    """Fetch and filter live flight data from OpenSky Network."""
    try:
        resp = requests.get("https://opensky-network.org/api/states/all", timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("states") is None:
            return pd.DataFrame()

        columns = [
            "icao24", "callsign", "origin_country", "time_position",
            "last_contact", "longitude", "latitude", "baro_altitude",
            "on_ground", "velocity", "true_track", "vertical_rate",
            "sensors", "geo_altitude", "squawk", "spi", "position_source"
        ]
        df = pd.DataFrame(data["states"], columns=columns)

        # Filter for cruise flights
        required = ["latitude", "longitude", "baro_altitude", "velocity", "true_track"]
        df = df.dropna(subset=required)
        for col in ["latitude", "longitude", "baro_altitude", "velocity", "true_track", "vertical_rate"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=required)

        df = df[
            (df["on_ground"] == False) &
            (df["baro_altitude"] > 5000) &
            (df["velocity"] > 100) &
            (df["latitude"].between(-85, 85)) &
            (df["longitude"].between(-180, 180))
        ]
        df["callsign"] = df["callsign"].str.strip()

        if len(df) > n_sample:
            df = df.sample(n=n_sample, random_state=42)

        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Failed to fetch flights: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_weather_batch(lats, lons):
    """Fetch weather data for multiple positions."""
    weather_data = []
    for lat, lon in zip(lats, lons):
        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={round(lat, 2)}&longitude={round(lon, 2)}"
                f"&hourly=wind_speed_250hPa,wind_direction_250hPa,temperature_250hPa"
                f"&forecast_days=1&timezone=auto"
            )
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            hourly = resp.json().get("hourly", {})
            weather_data.append({
                "wind_speed_250hPa": hourly.get("wind_speed_250hPa", [None])[0],
                "wind_direction_250hPa": hourly.get("wind_direction_250hPa", [None])[0],
                "temperature_250hPa": hourly.get("temperature_250hPa", [None])[0],
            })
        except Exception:
            weather_data.append({
                "wind_speed_250hPa": None,
                "wind_direction_250hPa": None,
                "temperature_250hPa": None,
            })
        time.sleep(0.3)  # Rate limiting

    return pd.DataFrame(weather_data)


def engineer_features(df):
    """Create physics-based features from raw flight and weather data."""
    df = df.copy()

    heading_rad = np.radians(df["true_track"])
    wind_dir_rad = np.radians(df["wind_direction_250hPa"])

    # Wind components
    df["headwind"] = df["wind_speed_250hPa"] * np.cos(heading_rad - wind_dir_rad)
    df["crosswind"] = df["wind_speed_250hPa"] * np.abs(np.sin(heading_rad - wind_dir_rad))

    # True Airspeed
    wind_ms = df["wind_speed_250hPa"] / 3.6
    headwind_ms = wind_ms * np.cos(heading_rad - wind_dir_rad)
    df["true_airspeed"] = df["velocity"] + headwind_ms

    # Air density (ISA)
    df["air_density"] = 1.225 * np.exp(-df["baro_altitude"] / 8500)

    # Mach number
    temp_kelvin = (df["temperature_250hPa"] + 273.15).clip(lower=180)
    speed_of_sound = 20.05 * np.sqrt(temp_kelvin)
    df["mach_number"] = df["true_airspeed"] / speed_of_sound

    # Flight level
    df["flight_level"] = (df["baro_altitude"] / 0.3048 / 100).round(0)

    # Vertical rate
    df["vertical_rate"] = df["vertical_rate"].fillna(0)
    df["abs_vertical_rate"] = np.abs(df["vertical_rate"])

    return df


# ──────────────────────────────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────────────────────────────

def load_model():
    """Load trained model and feature list."""
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    features_path = os.path.join(os.path.dirname(__file__), "features.json")

    if not os.path.exists(model_path) or not os.path.exists(features_path):
        return None, None

    model = joblib.load(model_path)
    with open(features_path, "r") as f:
        features = json.load(f)

    return model, features


# ──────────────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────────────

def main():
    # Title
    st.markdown('<div class="main-title">✈️ Flight Fuel Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-time ML-powered aviation fuel efficiency & sustainability tracking</div>', unsafe_allow_html=True)

    # Load model
    model, features = load_model()

    if model is None:
        st.error("⚠️ Model not found! Please run `analysis.ipynb` first to train and save the model.")
        st.info("The notebook will generate `model.joblib` and `features.json` in this directory.")
        st.stop()

    # Status badge
    st.markdown(
        '<div style="text-align: center; margin-bottom: 24px;">'
        '<span class="status-badge">● LIVE — Fetching Real-Time Data</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Fetch and process data
    with st.spinner("🛰️ Fetching live flight data from OpenSky Network..."):
        flights = fetch_live_flights(n_sample=50)

    if flights.empty:
        st.warning("No flight data available. OpenSky API may be temporarily unavailable. Please try again in a minute.")
        st.stop()

    with st.spinner("🌤️ Fetching weather data from Open-Meteo..."):
        weather = fetch_weather_batch(flights["latitude"].tolist(), flights["longitude"].tolist())
        df = pd.concat([flights.reset_index(drop=True), weather], axis=1)
        df = df.dropna(subset=["wind_speed_250hPa", "wind_direction_250hPa"]).reset_index(drop=True)

    if df.empty:
        st.warning("Weather data unavailable. Please try again.")
        st.stop()

    # Engineer features & predict
    df = engineer_features(df)

    # Fill NaNs in feature columns before prediction
    for f in features:
        if f in df.columns:
            df[f] = df[f].fillna(df[f].median())

    X = df[features]
    df["predicted_fuel_flow"] = model.predict(X)
    df["co2_per_hour"] = df["predicted_fuel_flow"] * 3.16

    # Efficiency score (lower fuel flow = more efficient)
    ff_min, ff_max = df["predicted_fuel_flow"].min(), df["predicted_fuel_flow"].max()
    df["efficiency_score"] = 100 * (1 - (df["predicted_fuel_flow"] - ff_min) / (ff_max - ff_min + 1e-10))

    # Color mapping for map (green = efficient, red = inefficient)
    df["color_r"] = (255 * (1 - df["efficiency_score"] / 100)).astype(int).clip(0, 255)
    df["color_g"] = (255 * (df["efficiency_score"] / 100)).astype(int).clip(0, 255)
    df["color_b"] = 80

    # ── KPI Cards ──
    avg_fuel = df["predicted_fuel_flow"].mean()
    total_co2 = df["co2_per_hour"].sum()
    avg_efficiency = df["efficiency_score"].mean()
    avg_headwind_pct = (df["headwind"].clip(lower=0).mean() / (df["wind_speed_250hPa"].mean() + 1e-10)) * 100

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">✈️</div>
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Flights Tracked</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">⛽</div>
            <div class="metric-value">{avg_fuel:,.0f}</div>
            <div class="metric-label">Avg Fuel Flow (kg/hr)</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🌍</div>
            <div class="metric-value warn">{total_co2:,.0f}</div>
            <div class="metric-label">Fleet CO₂ (kg/hr)</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">💨</div>
            <div class="metric-value">{avg_headwind_pct:.1f}%</div>
            <div class="metric-label">Avg Headwind Impact</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Global Flight Map ──
    st.markdown('<div class="section-header">🗺️ Global Flight Map</div>', unsafe_allow_html=True)
    st.markdown(
        '<span style="color: #8892a4; font-size: 13px;">'
        'Flights colored by fuel efficiency — '
        '<span style="color: #00ff88;">● Green = Efficient</span> · '
        '<span style="color: #ff4444;">● Red = Inefficient</span>'
        '</span>',
        unsafe_allow_html=True,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_color=["color_r", "color_g", "color_b", 200],
        get_radius=60000,
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=30.0,
        longitude=0.0,
        zoom=1.3,
        pitch=0,
    )

    tooltip = {
        "html": (
            "<div style='padding:8px; font-family:Inter,sans-serif;'>"
            "<b style='font-size:14px;'>{callsign}</b><br/>"
            "<span style='color:#8892a4;'>Country: {origin_country}</span><br/>"
            "Altitude: {baro_altitude} m<br/>"
            "Speed: {velocity} m/s<br/>"
            "Predicted Fuel: <b>{predicted_fuel_flow}</b> kg/hr<br/>"
            "Efficiency Score: {efficiency_score}%"
            "</div>"
        ),
        "style": {
            "backgroundColor": "#1a2332",
            "color": "#e0e6f0",
            "border": "1px solid rgba(0,212,170,0.3)",
            "borderRadius": "8px",
        },
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/dark-v10",
    )
    st.pydeck_chart(deck, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    st.markdown('<div class="section-header">📊 Analytics</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Fuel flow distribution
        fig1 = px.histogram(
            df, x="predicted_fuel_flow", nbins=20,
            title="Fuel Flow Distribution",
            labels={"predicted_fuel_flow": "Predicted Fuel Flow (kg/hr)"},
            color_discrete_sequence=["#00d4aa"],
            template="plotly_dark",
        )
        fig1.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#e0e6f0"),
            height=380,
            showlegend=False,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with chart_col2:
        # Altitude vs Fuel Flow
        fig2 = px.scatter(
            df, x="baro_altitude", y="predicted_fuel_flow",
            color="efficiency_score",
            color_continuous_scale="RdYlGn",
            title="Altitude vs Fuel Flow",
            labels={
                "baro_altitude": "Altitude (m)",
                "predicted_fuel_flow": "Fuel Flow (kg/hr)",
                "efficiency_score": "Efficiency %",
            },
            template="plotly_dark",
            hover_data=["callsign"],
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#e0e6f0"),
            height=380,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        # CO₂ by country (top 10)
        co2_by_country = (
            df.groupby("origin_country")["co2_per_hour"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig3 = px.bar(
            co2_by_country, x="origin_country", y="co2_per_hour",
            title="CO₂ Emissions by Country (Top 10)",
            labels={"origin_country": "Country", "co2_per_hour": "CO₂ (kg/hr)"},
            color="co2_per_hour",
            color_continuous_scale="Reds",
            template="plotly_dark",
        )
        fig3.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#e0e6f0"),
            height=380,
            showlegend=False,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with chart_col4:
        # Feature contributions (use model feature importances if available)
        trained_model = model.named_steps.get("model") if hasattr(model, "named_steps") else model
        if hasattr(trained_model, "feature_importances_"):
            feat_imp = pd.DataFrame({
                "Feature": features,
                "Importance": trained_model.feature_importances_,
            }).sort_values("Importance", ascending=True)

            fig4 = px.bar(
                feat_imp, x="Importance", y="Feature",
                orientation="h",
                title="Feature Importance",
                color="Importance",
                color_continuous_scale="Viridis",
                template="plotly_dark",
            )
            fig4.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#e0e6f0"),
                height=380,
                showlegend=False,
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

    # ── Sustainability KPI ──
    st.markdown('<div class="section-header">🌱 Sustainability — Carbon Intensity</div>', unsafe_allow_html=True)

    # Carbon intensity: kg CO₂ per km flown (estimated)
    # Rough estimate: each flight covers velocity * 1hr in meters
    df["est_distance_km"] = df["velocity"] * 3.6  # km/hr
    carbon_intensity = df["co2_per_hour"].sum() / (df["est_distance_km"].sum() + 1e-10)

    ci_col1, ci_col2, ci_col3 = st.columns(3)
    with ci_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📏</div>
            <div class="metric-value">{carbon_intensity:.2f}</div>
            <div class="metric-label">Carbon Intensity (kg CO₂/km)</div>
        </div>
        """, unsafe_allow_html=True)
    with ci_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🏆</div>
            <div class="metric-value">{avg_efficiency:.1f}%</div>
            <div class="metric-label">Avg Efficiency Score</div>
        </div>
        """, unsafe_allow_html=True)
    with ci_col3:
        most_efficient = df.loc[df["efficiency_score"].idxmax()]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">⭐</div>
            <div class="metric-value" style="font-size: 24px;">{most_efficient['callsign']}</div>
            <div class="metric-label">Most Efficient Flight</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Flight Details Table ──
    st.markdown('<div class="section-header">📋 Flight Details</div>', unsafe_allow_html=True)

    display_cols = [
        "callsign", "origin_country", "baro_altitude", "velocity",
        "headwind", "mach_number", "predicted_fuel_flow", "co2_per_hour",
        "efficiency_score"
    ]
    display_df = df[display_cols].copy()
    display_df.columns = [
        "Callsign", "Country", "Altitude (m)", "Speed (m/s)",
        "Headwind (km/h)", "Mach", "Fuel Flow (kg/hr)", "CO₂ (kg/hr)",
        "Efficiency %"
    ]
    # Round numeric columns
    for col in display_df.select_dtypes(include=[np.number]).columns:
        display_df[col] = display_df[col].round(1)

    display_df = display_df.sort_values("Fuel Flow (kg/hr)", ascending=True)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        hide_index=True,
    )

    # Footer
    st.markdown(
        '<div style="text-align: center; color: #555; margin-top: 40px; font-size: 12px;">'
        "Data: OpenSky Network · Open-Meteo | "
        "Model: scikit-learn / XGBoost | "
        "⚠️ Fuel flow values are physics-based estimates, not measured data"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
