# ✈️ Flight Fuel Analysis — ML-Powered Aviation Sustainability

Predict fuel flow and CO₂ emissions for live global flights using real-time data from **OpenSky Network** and **Open-Meteo** APIs, powered by machine learning.

## 🎯 Project Overview

This project combines **real-time aviation data** with **machine learning** to analyze fuel efficiency across global flights. It consists of two parts:

1. **Jupyter Notebook** (`analysis.ipynb`) — Data collection, feature engineering, EDA, and ML model training
2. **Streamlit Dashboard** (`app.py`) — Real-time global flight map with live ML inference and sustainability KPIs

## 🔧 Setup

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter Notebook first (trains and saves the model)
jupyter notebook analysis.ipynb

# Then launch the Streamlit Dashboard
streamlit run app.py
```

## 📊 Data Sources

| Source | Endpoint | Data | Auth |
|--------|----------|------|------|
| [OpenSky Network](https://opensky-network.org) | `/api/states/all` | Live flight positions, altitude, speed, heading | Anonymous (free) |
| [Open-Meteo](https://open-meteo.com) | `/v1/forecast` | Wind speed/direction at 250hPa (cruise altitude) | No key needed |

## 🧠 Features

- **Physics-based fuel estimation** using aerodynamic drag principles
- **Random Forest & XGBoost** models with cross-validation
- **SHAP explainability** for model interpretability
- **Real-time Streamlit dashboard** with global flight map
- **Sustainability KPIs** including carbon intensity scoring

## 📁 Project Structure

```
flight-fuel-analysis/
├── analysis.ipynb      # ML notebook (run first)
├── app.py              # Streamlit dashboard
├── model.joblib         # Trained model (generated)
├── features.json        # Feature list (generated)
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## ⚠️ Notes

- This is a **simulation-based study** — fuel flow targets are derived from physics models, not measured data
- Flight samples are limited to 50–80 flights to respect API rate limits
- The OpenSky API may occasionally return empty responses; the app handles this gracefully
