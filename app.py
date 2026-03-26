"""
Flight Fuel Analysis — Real-Time Streamlit Dashboard
"""

import json
import time
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import pydeck as pdk
import joblib
import shap
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Fuel Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────────────────────────────
# CSS — Dark Premium Theme
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #121929 50%, #0d1321 100%);
        font-family: 'Inter', sans-serif;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    [data-testid="stSidebar"] {
        background: rgba(10, 14, 23, 0.95);
        border-right: 1px solid rgba(0, 212, 170, 0.1);
    }

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
    .metric-icon { font-size: 32px; margin-bottom: 8px; }
    .metric-value {
        font-size: 36px; font-weight: 700;
        background: linear-gradient(135deg, #00d4aa, #4ecdc4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 4px 0;
    }
    .metric-value.warn {
        background: linear-gradient(135deg, #ff6b6b, #ffd93d);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #8892a4; font-size: 13px; font-weight: 500;
        text-transform: uppercase; letter-spacing: 1px;
    }

    .section-header {
        color: #e0e6f0; font-size: 20px; font-weight: 600;
        margin: 28px 0 14px 0; padding-bottom: 8px;
        border-bottom: 2px solid rgba(0, 212, 170, 0.3);
    }

    .main-title {
        font-size: 40px; font-weight: 700;
        background: linear-gradient(135deg, #00d4aa 0%, #4ecdc4 50%, #00b4d8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 4px;
    }
    .sub-title {
        color: #8892a4; font-size: 15px; text-align: center;
        margin-bottom: 20px; font-weight: 400;
    }
    .status-badge {
        display: inline-block;
        background: rgba(0, 212, 170, 0.15); color: #00d4aa;
        padding: 4px 12px; border-radius: 20px; font-size: 12px;
        font-weight: 600; letter-spacing: 0.5px;
        border: 1px solid rgba(0, 212, 170, 0.3);
    }
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    .stPlotlyChart { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Data Functions
# ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_live_flights(n_sample=60):
    """Fetch live flight data from OpenSky Network."""
    try:
        resp = requests.get(
            "https://opensky-network.org/api/states/all",
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("states"):
            return pd.DataFrame()

        columns = [
            "icao24", "callsign", "origin_country", "time_position",
            "last_contact", "longitude", "latitude", "baro_altitude",
            "on_ground", "velocity", "true_track", "vertical_rate",
            "sensors", "geo_altitude", "squawk", "spi", "position_source"
        ]
        df = pd.DataFrame(data["states"], columns=columns)

        required = ["latitude", "longitude", "baro_altitude", "velocity", "true_track"]
        df = df.dropna(subset=required)
        for col in required + ["vertical_rate"]:
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
    """Fetch weather for multiple positions using Open-Meteo."""
    weather_data = []
    batch_size = 50
    pairs = list(zip(lats, lons))

    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        lat_str = ",".join(str(round(lat, 2)) for lat, _ in chunk)
        lon_str = ",".join(str(round(lon, 2)) for _, lon in chunk)
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat_str}&longitude={lon_str}"
            f"&hourly=wind_speed_250hPa,wind_direction_250hPa,temperature_250hPa"
            f"&forecast_days=1"
        )
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            result = resp.json()
            # API returns a list when multiple locations are requested
            if isinstance(result, dict):
                result = [result]
            for loc in result:
                hourly = loc.get("hourly", {})
                weather_data.append({
                    "wind_speed_250hPa": hourly.get("wind_speed_250hPa", [None])[0],
                    "wind_direction_250hPa": hourly.get("wind_direction_250hPa", [None])[0],
                    "temperature_250hPa": hourly.get("temperature_250hPa", [None])[0],
                })
        except Exception:
            for _ in chunk:
                weather_data.append({
                    "wind_speed_250hPa": None,
                    "wind_direction_250hPa": None,
                    "temperature_250hPa": None,
                })
        time.sleep(0.5)

    return pd.DataFrame(weather_data)


def engineer_features(df):
    """Physics-based feature engineering."""
    df = df.copy()
    heading_rad = np.radians(df["true_track"])
    wind_dir_rad = np.radians(df["wind_direction_250hPa"])

    df["headwind"] = df["wind_speed_250hPa"] * np.cos(heading_rad - wind_dir_rad)
    df["crosswind"] = df["wind_speed_250hPa"] * np.abs(np.sin(heading_rad - wind_dir_rad))

    wind_ms = df["wind_speed_250hPa"] / 3.6
    headwind_ms = wind_ms * np.cos(heading_rad - wind_dir_rad)
    df["true_airspeed"] = df["velocity"] + headwind_ms

    temp_kelvin = (df["temperature_250hPa"] + 273.15).clip(lower=180)
    speed_of_sound = 20.05 * np.sqrt(temp_kelvin)
    df["mach_number"] = df["true_airspeed"] / speed_of_sound

    df["vertical_rate"] = df["vertical_rate"].fillna(0)
    df["abs_vertical_rate"] = np.abs(df["vertical_rate"])

    return df


def load_model():
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "model.joblib")
    features_path = os.path.join(base, "features.json")
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        return None, None
    model = joblib.load(model_path)
    with open(features_path) as f:
        features = json.load(f)
    return model, features


# ──────────────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown('<div class="main-title">✈️ Flight Fuel Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Real-time ML-powered aviation fuel efficiency & sustainability tracking</div>',
        unsafe_allow_html=True
    )

    # Load model
    model, features = load_model()
    if model is None:
        st.error("⚠️ Model not found. Please run `analysis.ipynb` first to train and save the model.")
        st.stop()

    st.markdown(
        '<div style="text-align:center;margin-bottom:20px;">'
        '<span class="status-badge">● LIVE — Fetching Real-Time Data</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Fetch Data ──
    with st.spinner("🛰️ Fetching live flights from OpenSky Network..."):
        flights = fetch_live_flights(n_sample=60)

    if flights.empty:
        st.warning("No flight data available. OpenSky API may be temporarily unavailable.")
        st.stop()

    with st.spinner("🌤️ Fetching weather data from Open-Meteo..."):
        weather = fetch_weather_batch(
            tuple(flights["latitude"].tolist()),
            tuple(flights["longitude"].tolist())
        )
        df = pd.concat([flights.reset_index(drop=True), weather], axis=1)
        df = df.dropna(subset=["wind_speed_250hPa", "wind_direction_250hPa"]).reset_index(drop=True)

    if df.empty:
        st.warning("Weather data unavailable. Please try again.")
        st.stop()

    # ── Engineer & Predict ──
    df = engineer_features(df)
    for f in features:
        if f in df.columns:
            df[f] = df[f].fillna(df[f].median())

    X = df[features]
    df["predicted_fuel_flow"] = model.predict(X)
    df["co2_per_hour"] = df["predicted_fuel_flow"] * 3.16

    ff_min, ff_max = df["predicted_fuel_flow"].min(), df["predicted_fuel_flow"].max()
    df["efficiency_score"] = 100 * (
        1 - (df["predicted_fuel_flow"] - ff_min) / (ff_max - ff_min + 1e-10)
    )

    # ── KPI Cards ──
    avg_fuel = df["predicted_fuel_flow"].mean()
    total_co2 = df["co2_per_hour"].sum()
    avg_eff = df["efficiency_score"].mean()
    n_flights = len(df)

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (k1, "✈️", str(n_flights), "Flights Tracked", False),
        (k2, "⛽", f"{avg_fuel:,.0f}", "Avg Fuel Flow (kg/hr)", False),
        (k3, "🌍", f"{total_co2:,.0f}", "Fleet CO₂ (kg/hr)", True),
        (k4, "🏆", f"{avg_eff:.1f}%", "Avg Efficiency Score", False),
    ]
    for col, icon, val, label, warn in kpis:
        warn_cls = " warn" if warn else ""
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value{warn_cls}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Global Flight Map ──
    st.markdown('<div class="section-header">🗺️ Global Flight Map</div>', unsafe_allow_html=True)
    st.markdown(
        '<span style="color:#8892a4;font-size:13px;">'
        'Color by fuel efficiency — '
        '<span style="color:#00ff88;">● Green = Efficient</span> · '
        '<span style="color:#ff4444;">● Red = Inefficient</span>'
        '</span>',
        unsafe_allow_html=True,
    )

    # Color mapping
    df["color_r"] = (255 * (1 - df["efficiency_score"] / 100)).clip(0, 255).astype(int)
    df["color_g"] = (255 * (df["efficiency_score"] / 100)).clip(0, 255).astype(int)
    df["color_b"] = 80

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_color=["color_r", "color_g", "color_b", 200],
        get_radius=80000,
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": (
            "<div style='padding:8px;font-family:Inter,sans-serif;'>"
            "<b style='font-size:14px;'>{callsign}</b><br/>"
            "<span style='color:#8892a4;'>Country: {origin_country}</span><br/>"
            "Altitude: {baro_altitude} m<br/>"
            "Speed: {velocity} m/s<br/>"
            "Predicted Fuel: <b>{predicted_fuel_flow}</b> kg/hr<br/>"
            "Efficiency: {efficiency_score}%"
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
        initial_view_state=pdk.ViewState(latitude=30.0, longitude=0.0, zoom=1.2, pitch=0),
        tooltip=tooltip,
        # Use Carto dark style — no Mapbox token required
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    )
    st.pydeck_chart(deck, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Analytics: 2 focused charts ──
    st.markdown('<div class="section-header">📊 Analytics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    CHART_LAYOUT = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e0e6f0"),
        height=360,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    with col1:
        fig1 = px.scatter(
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
        fig1.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # SHAP feature importance
        trained_model = model.named_steps.get("model") if hasattr(model, "named_steps") else model
        scaler = model.named_steps.get("scaler") if hasattr(model, "named_steps") else None
        try:
            X_scaled = pd.DataFrame(scaler.transform(X), columns=features) if scaler else X
            explainer = shap.TreeExplainer(trained_model)
            shap_values = explainer.shap_values(X_scaled)
            fig2 = plt.figure(figsize=(7, 5))
            shap.summary_plot(shap_values, X_scaled, plot_type="bar", show=False, color="#008bfb")
            plt.title("SHAP Feature Importance", fontsize=14, fontweight="bold", pad=12)
            fig2.patch.set_facecolor("white")
            plt.gca().set_facecolor("#eaeaf2")
            st.pyplot(fig2, clear_figure=True, use_container_width=True)
        except Exception as e:
            st.info(f"SHAP unavailable: {e}")

    # ── CO₂ by Country ──
    co2_by_country = (
        df.groupby("origin_country")["co2_per_hour"]
        .sum().sort_values(ascending=False).head(10).reset_index()
    )
    fig3 = px.bar(
        co2_by_country, x="origin_country", y="co2_per_hour",
        title="CO₂ Emissions by Country (Top 10)",
        labels={"origin_country": "Country", "co2_per_hour": "CO₂ (kg/hr)"},
        color="co2_per_hour", color_continuous_scale="Reds",
        template="plotly_dark",
    )
    fig3.update_layout(**CHART_LAYOUT, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Carbon Intensity ──
    st.markdown('<div class="section-header">🌱 Sustainability</div>', unsafe_allow_html=True)
    df["est_distance_km"] = df["velocity"] * 3.6
    carbon_intensity = df["co2_per_hour"].sum() / (df["est_distance_km"].sum() + 1e-10)
    most_efficient = df.loc[df["efficiency_score"].idxmax()]

    ci1, ci2, ci3 = st.columns(3)
    for col, icon, val, label, warn in [
        (ci1, "📏", f"{carbon_intensity:.2f}", "Carbon Intensity (kg CO₂/km)", False),
        (ci2, "🏆", f"{avg_eff:.1f}%", "Avg Efficiency Score", False),
        (ci3, "⭐", str(most_efficient["callsign"]), "Most Efficient Flight", False),
    ]:
        warn_cls = " warn" if warn else ""
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value{warn_cls}" style="font-size:28px;">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Flight Table ──
    st.markdown('<div class="section-header">📋 Flight Details</div>', unsafe_allow_html=True)
    display_cols = [
        "callsign", "origin_country", "baro_altitude", "velocity",
        "headwind", "mach_number", "predicted_fuel_flow", "co2_per_hour", "efficiency_score"
    ]
    display_df = df[display_cols].copy()
    display_df.columns = [
        "Callsign", "Country", "Altitude (m)", "Speed (m/s)",
        "Headwind (km/h)", "Mach", "Fuel Flow (kg/hr)", "CO₂ (kg/hr)", "Efficiency %"
    ]
    for col in display_df.select_dtypes(include=[np.number]).columns:
        display_df[col] = display_df[col].round(1)
    display_df = display_df.sort_values("Fuel Flow (kg/hr)", ascending=True)

    st.dataframe(display_df, use_container_width=True, height=380, hide_index=True)

    st.markdown(
        '<div style="text-align:center;color:#555;margin-top:30px;font-size:12px;">'
        "Data: OpenSky Network · Open-Meteo | "
        "Model: XGBoost | "
        "⚠️ Fuel values are physics-based estimates, not measured data"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
