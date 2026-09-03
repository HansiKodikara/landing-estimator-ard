"""Replay a simulated trajectory as a live telemetry stream.

Turns a :class:`~lze.sim.trajectory.Trajectory` (ground truth) into a sequence
of noisy :class:`TelemetryPacket`s at the downlink rate, exactly as if the
rocket were flying and the ground station were receiving it. This drives the
demo and lets us validate the whole live pipeline against a known landing point.
"""
from __future__ import annotations

import time
from typing import Iterator, Optional

import numpy as np

from ..geo import Origin
from ..sim.trajectory import Trajectory
from .schema import TelemetryPacket


def trajectory_to_packets(
    tr: Trajectory,
    origin: Origin,
    rate_hz: float = 1.0,
    gps_h_noise: float = 3.0,
    gps_v_noise: float = 5.0,
    baro_noise: float = 2.0,
    vel_noise: float = 1.5,
    seed: int = 0,
) -> list[TelemetryPacket]:
    """Resample the trajectory at ``rate_hz`` and add sensor noise."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / rate_hz
    t_grid = np.arange(0.0, tr.t_landing + 1e-9, dt)

    e = np.interp(t_grid, tr.t, tr.e)
    n = np.interp(t_grid, tr.t, tr.n)
    u = np.interp(t_grid, tr.t, tr.u)
    ve = np.interp(t_grid, tr.t, tr.ve)
    vn = np.interp(t_grid, tr.t, tr.vn)
    vu = np.interp(t_grid, tr.t, tr.vu)

    packets: list[TelemetryPacket] = []
    for i, t in enumerate(t_grid):
        # Add GPS noise in ENU, then convert to lat/lon.
        e_noisy = e[i] + rng.normal(0, gps_h_noise)
        n_noisy = n[i] + rng.normal(0, gps_h_noise)
        u_gps = max(0.0, u[i] + rng.normal(0, gps_v_noise))
        lat, lon, alt_gps = origin.enu_to_geo(e_noisy, n_noisy, u_gps)
        packets.append(
            TelemetryPacket(
                t=float(t),
                lat=lat,
                lon=lon,
                alt_gps=alt_gps,
                alt_baro_agl=float(max(0.0, u[i] + rng.normal(0, baro_noise))),
                ve=float(ve[i] + rng.normal(0, vel_noise)),
                vn=float(vn[i] + rng.normal(0, vel_noise)),
                vu=float(vu[i] + rng.normal(0, vel_noise)),
                packet_id=i,
                rssi=float(-60 - 0.01 * np.hypot(e[i], n[i]) + rng.normal(0, 2)),
            )
        )
    return packets


def stream_packets(
    packets: list[TelemetryPacket],
    realtime: bool = False,
    speed: float = 1.0,
    on_gap: Optional[float] = None,
) -> Iterator[TelemetryPacket]:
    """Yield packets, optionally pacing them in (accelerated) real time.

    ``speed`` > 1 replays faster than real time. ``realtime=False`` yields as
    fast as possible (for tests/batch runs).
    """
    prev_t: Optional[float] = None
    for pkt in packets:
        if realtime and prev_t is not None:
            delay = (pkt.t - prev_t) / max(speed, 1e-6)
            if delay > 0:
                time.sleep(delay)
        prev_t = pkt.t
        yield pkt
