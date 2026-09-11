#!/usr/bin/env python3
"""Measure how well the landing estimator works, end to end.

    python scripts/evaluate.py

One command, four outputs:

  1. what model is loaded, and which simulator taught it
  2. live accuracy on fresh flights the model has never seen
  3. whether the drawn recovery circle actually contains the landing
  4. a self-contained flight_replay.html to open in front of people

Deliberately prints the caveats alongside the numbers -- the figures measure
how faithfully the surrogate imitates the simulator, which is not the same
claim as predicting reality.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import _bootstrap  # noqa: F401
import numpy as np

from lze.config import load_config
from lze.geo import Origin
from lze.live.predictor import LandingPredictor
from lze.live.replay_page import build_replay_page, write_replay_page
from lze.model.surrogate import Surrogate
from lze.sim import engine_name, simulate
from lze.telemetry.replay import trajectory_to_packets

PHASES = ["boost", "coast", "drogue", "main"]
RULE = "=" * 68


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="data/surrogate.joblib")
    ap.add_argument("--config", default=None)
    ap.add_argument("--flights", type=int, default=25,
                    help="fresh flights to evaluate on")
    ap.add_argument("--out", default="flight_replay.html")
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    cfg = load_config(args.config)
    lat, lon, elev = cfg.site_origin
    origin = Origin(lat, lon, elev)
    sur = _bootstrap.load_model(args.model)

    # ---------- 1. provenance ----------
    print(f"\n{RULE}\n  KRONOS LANDING-ZONE ESTIMATOR -- evaluation\n{RULE}")
    env = sur.metadata.get("env") or {}
    meta_engine = sur.metadata.get("engine", "unknown")
    print(f"\n1. WHAT IS LOADED")
    print(f"   model            {args.model}")
    print(f"   trained          {env.get('saved_utc', '?')}")
    print(f"   model type       {env.get('model_class', '?')}")
    print(f"   taught by        {meta_engine}"
          f"{'   <-- 3-DOF approximation, NOT RocketPy' if 'rocketpy' not in str(meta_engine).lower() else ''}")
    print(f"   trained on       {sur.metadata.get('n_samples', '?'):,} samples "
          f"from {sur.metadata.get('n_train_flights', '?')} flights")

    # ---------- 2. live accuracy on unseen flights ----------
    print(f"\n2. ACCURACY ON {args.flights} FRESH FLIGHTS THE MODEL HAS NEVER SEEN")
    print(f"   (full live path: noisy 1 Hz telemetry -> estimator -> surrogate)")
    rng = np.random.default_rng(args.seed)
    errs = defaultdict(list)
    finals = []
    covered = defaultdict(list)
    t0 = time.time()

    for _ in range(args.flights):
        ws = rng.uniform(2.0, 11.0)
        wd = np.radians(rng.uniform(0.0, 360.0))
        tr = simulate(
            cfg, prefer="fallback",
            wind_east=float(-ws * np.sin(wd)), wind_north=float(-ws * np.cos(wd)),
            dry_mass=float(rng.uniform(14.2, 16.3)),
        )
        tlat, tlon, _ = origin.enu_to_geo(tr.landing_e, tr.landing_n, 0.0)
        pred = LandingPredictor(cfg, sur, origin)
        last = None
        for pkt in trajectory_to_packets(tr, origin, rate_hz=1.0,
                                         seed=int(rng.integers(1e6))):
            p = pred.process(pkt)
            e = pred.error_against_truth(p, tlat, tlon)
            errs[p.phase].append(e)
            covered[p.phase].append(e <= p.uncertainty_m)
            last = e
        finals.append(last)

    print(f"\n   {'phase':>8} {'samples':>9} {'mean err':>10} {'median':>9} {'in circle':>11}")
    print("   " + "-" * 52)
    for ph in PHASES:
        if not errs[ph]:
            continue
        c = 100.0 * sum(covered[ph]) / len(covered[ph])
        print(f"   {ph:>8} {len(errs[ph]):>9,} {np.mean(errs[ph]):>9.0f}m "
              f"{np.median(errs[ph]):>8.0f}m {c:>10.0f}%")

    first, last_ph = PHASES[0], PHASES[-1]
    if errs[first] and errs[last_ph]:
        ratio = np.mean(errs[first]) / np.mean(errs[last_ph])
        print(f"\n   >> the recovery zone tightens {ratio:.0f}x from boost to main <<")
    print(f"   final fix at touchdown:  mean {np.mean(finals):.0f} m  "
          f"median {np.median(finals):.0f} m   ({len(finals)} flights)")
    print(f"   evaluated in {time.time() - t0:.1f} s")

    # ---------- 3. the shareable demo ----------
    tr = simulate(cfg, prefer="fallback", wind_east=6.0, wind_north=-3.0,
                  dry_mass=15.0, dt=0.5)
    packets = trajectory_to_packets(tr, origin, rate_hz=cfg.telemetry["rate_hz"],
                                    seed=args.seed)
    html = build_replay_page(
        LandingPredictor(cfg, sur, origin), packets, origin,
        truth_land_e=tr.landing_e, truth_land_n=tr.landing_n, speed=8.0,
    )
    write_replay_page(args.out, html)
    print(f"\n3. SHAREABLE DEMO")
    print(f"   wrote {args.out} -- open it in any browser, no server, no internet.")
    print(f"   Watch the recovery circle collapse as the rocket descends.")

    # ---------- 4. caveats, stated up front ----------
    print(f"\n4. READ THIS BEFORE QUOTING THE NUMBERS")
    print(f"   * Flights come from {engine_name('auto')}. These figures measure how")
    print(f"     faithfully the surrogate imitates that simulator -- not how well it")
    print(f"     predicts reality. Install RocketPy (6-DOF) for a stronger claim.")
    print(f"   * The ARD telemetry downlink carries no GPS yet, so on a real flight")
    print(f"     there is no measured horizontal position to anchor the prediction.")
    print(f"     Workaround that exists today: --source featherweight reads the")
    print(f"     Featherweight Ground Station v2 over USB and does carry real GPS.")
    print(f"   * Run scripts/check_model.py on the Raspberry Pi itself to confirm")
    print(f"     speed and memory on the actual hardware.")
    print(f"\n{RULE}\n")


if __name__ == "__main__":
    main()