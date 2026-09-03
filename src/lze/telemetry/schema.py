"""Telemetry packet schema.

Mirrors the Kronos downlink described in the CDR (Telemetry board + Ground
Station): GPS position at ~1 Hz from the NEO-M10S, a Kalman-fused velocity
vector from the IMU, and a redundant barometric altitude. This is the only
information the live predictor is allowed to use -- it never sees the true
state.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class TelemetryPacket:
    """One downlinked telemetry frame."""

    t: float            # seconds since launch detect (rocket clock)
    lat: float          # GPS latitude [deg]
    lon: float          # GPS longitude [deg]
    alt_gps: float      # GPS altitude ASL [m]
    alt_baro_agl: float  # barometric altitude AGL [m] (pad-zeroed)
    # Kalman-fused velocity in the launch-pad ENU frame [m/s].
    ve: float           # east
    vn: float           # north
    vu: float           # up
    # Convenience / link-health fields.
    packet_id: int = 0
    rssi: float = 0.0   # dBm (not used by the model; shown on dashboard)

    @property
    def horizontal_speed(self) -> float:
        return (self.ve ** 2 + self.vn ** 2) ** 0.5

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "TelemetryPacket":
        return cls(**json.loads(s))
