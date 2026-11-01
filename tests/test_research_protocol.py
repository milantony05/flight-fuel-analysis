"""Tests for the frozen confirmatory protocol and streaming contract."""

import unittest
from pathlib import Path

import pandas as pd

from src.flight_fuel.research_protocol import (
    confirmation_fingerprint,
    load_protocol,
    validate_confirmation_frame,
)
from src.flight_fuel.streaming import ApprovedLiveTelemetry, HistoricalReplay

ROOT = Path(__file__).resolve().parents[1]


class ResearchProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(ROOT / "configs/confirmatory-protocol.json")
        self.frame = pd.DataFrame({
            "interval_id": ["i-1", "i-2"],
            "flight_id": ["f-1", "f-2"],
            "interval_start_utc": ["2026-11-01T00:00:00Z", "2026-11-01T01:00:00Z"],
            "interval_end_utc": ["2026-11-01T00:10:00Z", "2026-11-01T01:20:00Z"],
            "aircraft_type": ["A320", "B738"],
            "predicted_fuel_kg": [100.0, 250.0],
            "reference_fuel_kg": [110.0, 240.0],
            "predicted_error_risk_kg": [12.0, 18.0],
            "adsb_position_count": [50, 0],
            "origin_icao": ["AAAA", "BBBB"],
            "destination_icao": ["CCCC", "DDDD"],
            "model_version": ["frozen-v1", "frozen-v1"],
        })

    def test_confirmation_schema_and_fingerprint_are_stable(self):
        summary = validate_confirmation_frame(self.frame, self.protocol)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["flights"], 2)
        first = confirmation_fingerprint(self.frame, self.protocol)
        second = confirmation_fingerprint(self.frame.iloc[::-1], self.protocol)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_confirmation_rejects_duplicate_intervals(self):
        bad = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            validate_confirmation_frame(bad, self.protocol)

    def test_confirmation_rejects_pre_freeze_intervals(self):
        bad = self.frame.copy()
        bad.loc[0, "interval_start_utc"] = "2025-09-20T00:00:00Z"
        bad.loc[0, "interval_end_utc"] = "2025-09-20T00:10:00Z"
        with self.assertRaises(ValueError):
            validate_confirmation_frame(bad, self.protocol)

    def test_historical_stream_can_withhold_labels(self):
        source = HistoricalReplay(ROOT / "demo/index.html", reveal_delayed_labels=False)
        event = next(iter(source.events()))
        self.assertEqual(event.source, "historical_replay")
        self.assertIsNone(event.reference_fuel_kg)
        self.assertEqual(event.reference_status, "withheld")

    def test_live_source_requires_an_explicit_approved_provider(self):
        with self.assertRaises(ValueError):
            ApprovedLiveTelemetry()


if __name__ == "__main__":
    unittest.main()
