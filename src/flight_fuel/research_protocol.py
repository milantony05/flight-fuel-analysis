"""Validation helpers for a future untouched confirmatory batch.

This module intentionally does not download or open a private provider. It
validates an explicitly supplied table against the frozen protocol before any
confirmatory analysis is run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    required = {
        "protocol_version", "frozen_on", "status", "primary_hypotheses",
        "required_confirmation_columns", "locked_primary_setting", "claim_rules",
    }
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"Protocol is missing fields: {sorted(missing)}")
    if protocol["status"] != "prospective_protocol_no_confirmatory_labels_loaded":
        raise ValueError("Confirmatory protocol must retain its prospective status")
    return protocol


def validate_confirmation_frame(frame: pd.DataFrame, protocol: dict) -> dict:
    """Validate schema, identifiers, time bounds and numerical fields."""
    required = list(protocol["required_confirmation_columns"])
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Confirmation table is missing columns: {missing}")
    if not len(frame):
        raise ValueError("Confirmation table is empty")
    if frame.interval_id.isna().any() or frame.interval_id.duplicated().any():
        raise ValueError("interval_id must be complete and unique")
    if frame.flight_id.isna().any() or frame.aircraft_type.isna().any():
        raise ValueError("flight_id and aircraft_type must be complete")
    start = pd.to_datetime(frame.interval_start_utc, utc=True, errors="coerce")
    end = pd.to_datetime(frame.interval_end_utc, utc=True, errors="coerce")
    if start.isna().any() or end.isna().any() or (end <= start).any():
        raise ValueError("Intervals require valid UTC start and end times")
    freeze_day = pd.Timestamp(protocol["frozen_on"], tz="UTC")
    if (start < freeze_day + pd.Timedelta(days=1)).any():
        raise ValueError("Confirmation intervals must begin after the protocol freeze day")
    numeric = frame[[
        "predicted_fuel_kg", "reference_fuel_kg",
        "predicted_error_risk_kg", "adsb_position_count",
    ]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Prediction, label, risk and position count must be finite")
    if (numeric[["predicted_fuel_kg", "reference_fuel_kg", "predicted_error_risk_kg"]] < 0).any().any():
        raise ValueError("Fuel and risk values must be nonnegative")
    if (numeric.adsb_position_count < 0).any():
        raise ValueError("adsb_position_count must be nonnegative")
    return {
        "rows": int(len(frame)),
        "flights": int(frame.flight_id.nunique()),
        "aircraft_types": int(frame.aircraft_type.nunique()),
        "first_interval_start_utc": start.min().isoformat(),
        "last_interval_end_utc": end.max().isoformat(),
        "model_versions": sorted(frame.model_version.astype(str).unique().tolist()),
    }


def confirmation_fingerprint(frame: pd.DataFrame, protocol: dict) -> str:
    """Create a stable fingerprint for exact confirmatory-table identity."""
    columns = list(protocol["required_confirmation_columns"])
    canonical = frame[columns].sort_values("interval_id").to_json(
        orient="records", date_format="iso", date_unit="us", double_precision=15
    )
    return hashlib.sha256(canonical.encode()).hexdigest()

