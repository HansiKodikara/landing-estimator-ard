#!/usr/bin/env python3
"""Run the live landing-zone dashboard server.

Two telemetry sources:

* ``--source replay`` (default): simulate a flight and stream it in accelerated
  real time -- great for demos on any machine.
* ``--source serial``: read LoRa telemetry from a serial modem (the real ground
  station on the Raspberry Pi). Requires ``pyserial``.
* ``--source ard``: subscribe to a running ARD dashboard backend (Socket.IO) and
  predict the landing zone from *its* telemetry -- runs the recovery map
  alongside the ARD dashboard without touching that repo.
* ``--source ard-rest``: same, but over ARD's documented REST API
  (``/telemetry/history`` then polling ``/telemetry/latest``) -- no websocket
  and no extra dependencies.
* ``--source ard-file``: replay a captured ARD ``.jsonl`` telemetry log offline.

    python scripts/run_live.py --port 8000                        # replay demo
    python scripts/run_live.py --source serial --serial-port /dev/ttyUSB0
    python scripts/run_live.py --source ard --ard-url http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np

from lze.config import load_config
from lze.geo import Origin, wind_vector
from lze.live.predictor import LandingPredictor
from lze.live.server import LiveServer
from lze.model.surrogate import Surrogate
from lze.sim import simulate
from lze.telemetry.ard_adapter import (
    ArdReplaySource,
    ArdRestSource,
    ArdSocketIOSource,
    ard_envelopes_from_jsonl,
)
from lze.telemetry.replay import trajectory_to_packets
from lze.telemetry.source import ReplaySource, SerialLoRaSource


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default="data/surrogate.joblib")
    ap.add_argument(
        "--source",
        choices=["replay", "serial", "ard", "ard-rest", "ard-file"],
        default="replay",
    )
    ap.add_argument("--serial-port", default="/dev/ttyUSB0")
    ap.add_argument("--ard-url", default="http://127.0.0.1:5000",
                    help="ARD dashboard backend URL (--source ard / ard-rest)")
    ap.add_argument("--ard-poll-hz", type=float, default=4.0,
                    help="Polling rate for --source ard-rest")
    ap.add_argument("--ard-file", default=None,
                    help="Captured ARD telemetry .jsonl to replay (--source ard-file)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--speed", type=float, default=8.0, help="replay speed multiplier")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--wind-seed-speed", type=float, default=None,
                    help="Launch-day forecast wind speed [m/s] to seed ascent "
                         "predictions (needs a forecast-seeded model).")
    ap.add_argument("--wind-seed-from", type=float, default=None,
                    help="Forecast wind FROM heading [deg] (pairs with --wind-seed-speed).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sur = Surrogate.load(args.model)
    lat, lon, elev = cfg.site_origin
    origin = Origin(lat, lon, elev)

    wind_seed = None
    if args.wind_seed_speed is not None:
        heading = args.wind_seed_from if args.wind_seed_from is not None else \
            float(cfg.environment["wind_heading"])
        wind_seed = wind_vector(args.wind_seed_speed, heading)
        print(f"Seeding forecast wind: {args.wind_seed_speed:.1f} m/s FROM {heading:.0f} deg "
              f"-> ENU ({wind_seed[0]:.1f}, {wind_seed[1]:.1f})")

    predictor = LandingPredictor(cfg, sur, origin, wind_seed=wind_seed)

    truth = None
    if args.source == "replay":
        rng = np.random.default_rng(args.seed)
        ws = rng.uniform(4, 10)
        wd = np.radians(rng.uniform(0, 360))
        tr = simulate(
            cfg, wind_east=float(-ws * np.sin(wd)), wind_north=float(-ws * np.cos(wd)),
            dry_mass=15.0, inclination=86.0, heading=float(rng.uniform(0, 360)), dt=0.5,
        )
        t_lat, t_lon, _ = origin.enu_to_geo(tr.landing_e, tr.landing_n, 0.0)
        truth = {"e": tr.landing_e, "n": tr.landing_n, "lat": t_lat, "lon": t_lon}
        packets = trajectory_to_packets(tr, origin, rate_hz=cfg.telemetry["rate_hz"], seed=args.seed)
        source = ReplaySource(packets, realtime=True, speed=args.speed)
        print(f"Replaying a simulated flight (apogee {tr.apogee_alt/0.3048:.0f} ft) "
              f"at {args.speed}x")
    elif args.source == "serial":
        source = SerialLoRaSource(port=args.serial_port)
        print(f"Reading LoRa telemetry from {args.serial_port}")
    elif args.source == "ard":
        source = ArdSocketIOSource(url=args.ard_url, origin=origin)
        print(f"Subscribing to ARD dashboard telemetry at {args.ard_url}")
    elif args.source == "ard-rest":
        source = ArdRestSource(url=args.ard_url, origin=origin, poll_hz=args.ard_poll_hz)
        if not source.health():
            print(f"WARNING: no response from {args.ard_url}/health -- is the "
                  f"ARD backend running? Will keep retrying.")
        print(f"Polling ARD REST API at {args.ard_url} ({args.ard_poll_hz:g} Hz)")
    else:  # ard-file
        if not args.ard_file:
            ap.error("--source ard-file requires --ard-file <capture.jsonl>")
        envelopes = ard_envelopes_from_jsonl(args.ard_file)
        source = ArdReplaySource(envelopes, origin)
        print(f"Replaying {len(envelopes)} ARD telemetry frames from {args.ard_file}")

    server = LiveServer(predictor, source, host=args.host, port=args.port, truth=truth)
    server.serve_forever()


if __name__ == "__main__":
    main()
