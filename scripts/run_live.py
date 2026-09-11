#!/usr/bin/env python3
"""Run the live landing-zone dashboard server.

Telemetry sources:

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
* ``--source featherweight``: read a Featherweight Ground Station v2 directly
  over USB serial. This link carries a real GPS fix, which the ARD downlink
  does not, so it is the only source that can anchor the prediction to a
  measured horizontal position on a real flight.
* ``--source featherweight-file``: replay a captured Ground Station text log.

    python scripts/run_live.py --port 8000                        # replay demo
    python scripts/run_live.py --source serial --serial-port /dev/ttyUSB0
    python scripts/run_live.py --source ard --ard-url http://127.0.0.1:5000
    python scripts/run_live.py --source featherweight --serial-port /dev/ttyUSB0

Pass ``--record NAME`` on a real launch. Nothing is stored without it.
"""
from __future__ import annotations

import argparse
import os
import sys

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
from lze.telemetry.featherweight import (
    FeatherweightReplaySource,
    FeatherweightSource,
)
from lze.telemetry.replay import trajectory_to_packets
from lze.telemetry.source import ReplaySource, SerialLoRaSource, anchor_origin


#: Fixes medianed into the map origin. Ten seconds at 1 Hz -- long enough for
#: GPS noise to average down, short enough not to be a wait on the pad.
ORIGIN_FIXES = 10


def check_writable(path: str) -> None:
    """Refuse to start if a recording path cannot be written.

    Checked before the model loads and before the port is bound, because the
    alternative is discovering it from a dead worker thread once the rocket is
    already on the rail.

    The directory is deliberately NOT created. "--record /mnt/usb/flight" with
    the stick unmounted should stop, not quietly fill the root filesystem and
    leave someone believing the flight is on the USB.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        sys.exit(f"--record: directory does not exist: {parent}\n"
                 f"  Create it first, or pick a path that exists. "
                 f"Refusing to start rather than record nothing.")
    if not os.access(parent, os.W_OK):
        sys.exit(f"--record: directory is not writable: {parent}")
    # Probe by actually opening it -- os.access can be wrong about read-only
    # mounts. Tidy up afterwards: leaving an empty file behind would look like a
    # recording that captured nothing, which is the opposite of reassuring.
    existed = os.path.exists(path)
    try:
        with open(path, "a"):
            pass
    except OSError as exc:
        sys.exit(f"--record: cannot write {path}: {exc}")
    if not existed:
        try:
            os.unlink(path)
        except OSError:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default="data/surrogate.joblib")
    ap.add_argument(
        "--source",
        choices=["replay", "serial", "ard", "ard-rest", "ard-file",
                 "featherweight", "featherweight-file"],
        default="replay",
    )
    ap.add_argument("--serial-port", default="/dev/ttyUSB0")
    ap.add_argument("--serial-baud", type=int, default=115200,
                    help="Serial baud rate (Featherweight Ground Station v2 "
                         "is 115200 8N1 per the tracker manual)")
    ap.add_argument("--fw-file", default=None,
                    help="Captured Ground Station serial text log "
                         "(--source featherweight-file)")
    ap.add_argument("--origin-here", nargs="+", default=None, metavar="LAT LON ELEV",
                    help="Pin the map origin to this point instead of taking it "
                         "from the first GPS fixes, e.g. --origin-here -33.8688 "
                         "151.2093 40. Rarely needed: the tracker's own first "
                         "fix is the pad, and is measured by the receiver that "
                         "will measure the flight.")
    ap.add_argument("--fw-launch-vu", type=float, default=15.0,
                    help="Upward speed [m/s] that starts the flight clock "
                         "(--source featherweight)")
    ap.add_argument("--ard-url", default="http://127.0.0.1:5000",
                    help="ARD dashboard backend URL (--source ard / ard-rest)")
    ap.add_argument("--ard-poll-hz", type=float, default=4.0,
                    help="Polling rate for --source ard-rest")
    ap.add_argument("--ard-file", default=None,
                    help="Captured ARD telemetry .jsonl to replay (--source ard-file)")
    ap.add_argument("--record", default=None, metavar="NAME",
                    help="Record the flight to NAME.jsonl (one record per "
                         "packet: receive time, the packet, the prediction for "
                         "it) and, for sources with raw lines, NAME.txt (the "
                         "raw stream with receive times). A launch happens "
                         "once; nothing is recorded unless you pass this.")
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
    sur = _bootstrap.load_model(args.model)
    lat, lon, elev = cfg.site_origin
    if args.origin_here:
        # Same argparse trap as a recovery coordinate: a bare "-33.86" is read
        # as a flag unless it arrives as its own token, so take them separately.
        try:
            parts = " ".join(args.origin_here).replace(",", " ").split()
            lat, lon, elev = (float(x) for x in parts)
        except ValueError:
            ap.error("--origin-here needs LAT LON ELEV "
                     f"(got {' '.join(args.origin_here)!r})")
        print(f"Pad overridden to {lat:.5f}, {lon:.5f} at {elev:.0f} m "
              f"(config says {cfg.launch_site['name']})")
    origin = Origin(lat, lon, elev)

    wind_seed = None
    if args.wind_seed_speed is not None:
        heading = args.wind_seed_from if args.wind_seed_from is not None else \
            float(cfg.environment["wind_heading"])
        wind_seed = wind_vector(args.wind_seed_speed, heading)
        print(f"Seeding forecast wind: {args.wind_seed_speed:.1f} m/s FROM {heading:.0f} deg "
              f"-> ENU ({wind_seed[0]:.1f}, {wind_seed[1]:.1f})")

    predictor = LandingPredictor(cfg, sur, origin, wind_seed=wind_seed)

    # One argument, two artifacts. The raw text is the redundancy worth having:
    # if the parser turns out to be wrong about the real hardware, that file can
    # be re-parsed and the decoded .jsonl cannot.
    log_path = f"{args.record}.jsonl" if args.record else None
    raw_path = f"{args.record}.txt" if args.record else None
    for path in (log_path, raw_path):
        if path:
            check_writable(path)

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
    elif args.source == "ard-file":
        if not args.ard_file:
            ap.error("--source ard-file requires --ard-file <capture.jsonl>")
        envelopes = ard_envelopes_from_jsonl(args.ard_file)
        source = ArdReplaySource(envelopes, origin)
        print(f"Replaying {len(envelopes)} ARD telemetry frames from {args.ard_file}")
    elif args.source == "featherweight":
        source = FeatherweightSource(
            port=args.serial_port, baud=args.serial_baud, origin=origin,
            capture_path=raw_path,
            launch_detect_vu_ms=args.fw_launch_vu,
        )
        print(f"Reading Featherweight Ground Station v2 on {args.serial_port} "
              f"at {args.serial_baud} baud")
    else:  # featherweight-file
        if not args.fw_file:
            ap.error("--source featherweight-file requires --fw-file <capture.txt>")
        source = FeatherweightReplaySource.from_file(
            args.fw_file, origin, launch_detect_vu_ms=args.fw_launch_vu,
        )
        print(f"Replaying Featherweight serial log {args.fw_file}")

    if log_path:
        print(f"Recording every packet and prediction to {log_path}")
    if args.source != "replay" and not args.origin_here:
        print(f"Taking the map origin from up to {ORIGIN_FIXES} stationary "
              f"fixes (nothing to configure) ...")
        source, measured = anchor_origin(source, n=ORIGIN_FIXES)
        if measured is None:
            # Either no fix ever arrived, or the feed began already in motion.
            # Neither is worth guessing about, and the configured pad is a real
            # answer -- just not a measured one, so say which you are getting.
            print(f"\n  NOTE: no stationary fix to anchor on -- either nothing "
                  f"arrived, or\n  the tracker was already moving when this "
                  f"started. Falling back to the\n  configured pad "
                  f"({cfg.launch_site['name']}: {lat:.4f}, {lon:.4f}, "
                  f"{elev:.0f} m).\n  If that is not where you are, stop and "
                  f"pass --origin-here LAT LON ELEV.\n")
        else:
            origin = measured
            print(f"Origin set to {origin.lat:.6f}, {origin.lon:.6f} at "
                  f"{origin.elevation:.0f} m  <- measured, not configured")
            predictor = LandingPredictor(cfg, sur, origin, wind_seed=wind_seed)

    server = LiveServer(predictor, source, host=args.host, port=args.port,
                        truth=truth, log_path=log_path)
    server.serve_forever()


if __name__ == "__main__":
    main()