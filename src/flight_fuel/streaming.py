"""Provider-safe streaming contract for historical replay and future live data.

Historical events may reveal their committed reference after the simulated
interval closes. Live events never contain reference fuel unless a separate,
authorized delayed-ground-truth process supplies it.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

from .advanced import load_demo_payload


@dataclass(frozen=True)
class PredictionEvent:
    event_id: str
    source: str
    aircraft_type: str
    duration_min: float
    adsb_position_count: int
    predicted_fuel_kg: float
    predicted_error_risk_kg: float
    reference_fuel_kg: float | None
    reference_status: str


class EventSource(Protocol):
    def events(self) -> Iterable[PredictionEvent]: ...


class HistoricalReplay:
    """Replay the committed benchmark as sequential, closed intervals."""

    def __init__(self, dashboard_path: Path, *, reveal_delayed_labels: bool = True):
        self.payload = load_demo_payload(dashboard_path)
        self.reveal_delayed_labels = reveal_delayed_labels

    def events(self) -> Iterable[PredictionEvent]:
        for index, row in enumerate(self.payload["rows"]):
            yield PredictionEvent(
                event_id=f"historical-{index:05d}",
                source="historical_replay",
                aircraft_type=str(row[0]),
                duration_min=float(row[1]),
                predicted_error_risk_kg=float(row[2]),
                predicted_fuel_kg=float(row[3]),
                reference_fuel_kg=float(row[4]) if self.reveal_delayed_labels else None,
                adsb_position_count=int(row[5]),
                reference_status=(
                    "revealed_after_interval_close"
                    if self.reveal_delayed_labels else "withheld"
                ),
            )


class ApprovedLiveTelemetry:
    """Explicit integration point; no provider is silently assumed."""

    def __init__(self, provider_name: str | None = None):
        if not provider_name:
            raise ValueError(
                "Live mode requires an approved provider name, credentials, "
                "rate-limit handling and a documented license."
            )
        self.provider_name = provider_name

    def events(self) -> Iterable[PredictionEvent]:
        raise NotImplementedError(
            "Implement the approved provider adapter without exposing a live "
            "reference-fuel field."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-demo", type=Path, default=Path("data/benchmark-source.html"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--pace-seconds", type=float, default=0.0)
    parser.add_argument("--withhold-labels", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or args.pace_seconds < 0:
        raise ValueError("limit must be positive and pace must be nonnegative")
    source = HistoricalReplay(
        args.source_demo, reveal_delayed_labels=not args.withhold_labels
    )
    for index, event in enumerate(source.events()):
        if index >= args.limit:
            break
        print(json.dumps(asdict(event), separators=(",", ":")), flush=True)
        if args.pace_seconds:
            time.sleep(args.pace_seconds)


if __name__ == "__main__":
    main()
