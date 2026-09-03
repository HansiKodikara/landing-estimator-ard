#!/usr/bin/env python3
"""End-to-end demo: simulate a Kronos flight, predict its landing live, and bake
a self-contained replay dashboard.

    python scripts/demo.py --out flight_replay.html

Prints the prediction sharpening from boost to touchdown and writes an HTML file
you can open in any browser (no server needed).
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np

from lze.config import load_config
from lze.geo import Origin
from lze.live.predictor import LandingPredictor
from lze.live.replay_page import build_replay_page, write_replay_page
from lze.model.surrogate import Surrogate
from lze.sim import engine_name, simulate
from lze.telemetry.replay import trajectory_to_packets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default="data/surrogate.joblib")
    ap.add_argument("--engine", choices=["auto", "rocketpy", "fallback"], default="auto")
    ap.add_argument("--out", default="flight_replay.html")
    ap.add_argument("--wind-speed", type=float, default=8.0)
    ap.add_argument("--wind-from", type=float, default=225.0, help="deg wind blows FROM")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sur = Surrogate.load(args.model)
    lat, lon, elev = cfg.site_origin
    origin = Origin(lat, lon, elev)

    wd = np.radians(args.wind_from)
    we, wn = -args.wind_speed * np.sin(wd), -args.wind_speed * np.cos(wd)

    print(f"Engine: {engine_name(args.engine)}   Site: {cfg.launch_site['name']}")
    tr = simulate(
        cfg, prefer=args.engine,
        wind_east=float(we), wind_north=float(wn),
        dry_mass=15.0, inclination=86.0, heading=90.0, dt=0.5,
    )
    true_lat, true_lon, _ = origin.enu_to_geo(tr.landing_e, tr.landing_n, 0.0)
    print(f"Simulated flight: apogee {tr.apogee_alt/0.3048:.0f} ft, "
          f"flight {tr.t_landing:.0f} s, true landing "
          f"{true_lat:.5f}, {true_lon:.5f} ({tr.landing_e:.0f} E, {tr.landing_n:.0f} N m)")

    packets = trajectory_to_packets(
        tr, origin, rate_hz=cfg.telemetry["rate_hz"], seed=args.seed
    )

    # First pass: print the prediction sharpening over the flight.
    pred = LandingPredictor(cfg, sur, origin)
    print("\n  T+    alt   phase    pred error   zone r   ETA")
    print("  ---------------------------------------------------")
    last_phase = None
    for i, pkt in enumerate(packets):
        p = pred.process(pkt)
        err = pred.error_against_truth(p, true_lat, true_lon)
        show = (p.phase != last_phase) or (i % 15 == 0) or (i == len(packets) - 1)
        if show:
            print(f"  {p.t:4.0f}s {p.alt_agl:5.0f}m  {p.phase:6s}  "
                  f"{err:7.0f} m   {p.uncertainty_m:4.0f} m  {p.remaining_time_s:4.0f}s")
        last_phase = p.phase
    print(f"\nFinal landing prediction error: {err:.0f} m")

    # Second pass (fresh predictor) to bake the replay page.
    pred2 = LandingPredictor(cfg, sur, origin)
    html = build_replay_page(
        pred2, packets, origin,
        truth_land_e=tr.landing_e, truth_land_n=tr.landing_n, speed=8.0,
    )
    write_replay_page(args.out, html)
    print(f"Wrote self-contained replay dashboard to {args.out}")


if __name__ == "__main__":
    main()
