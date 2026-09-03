#!/usr/bin/env python3
"""Evaluate the *live* pipeline (estimator + surrogate + smoothing) end to end.

Unlike ``train_model.py`` (which scores the raw regressor on individual samples),
this simulates fresh flights, replays them as noisy telemetry, runs the online
predictor, and reports landing error as a function of altitude/phase -- the real
"does the recovery area tighten during flight?" question.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np

from lze.config import load_config
from lze.geo import Origin
from lze.live.predictor import LandingPredictor
from lze.model.surrogate import Surrogate
from lze.sim import engine_name, simulate
from lze.telemetry.replay import trajectory_to_packets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default="data/surrogate.joblib")
    ap.add_argument("--engine", choices=["auto", "rocketpy", "fallback"], default="auto")
    ap.add_argument("--n-flights", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sur = Surrogate.load(args.model)
    lat, lon, elev = cfg.site_origin
    origin = Origin(lat, lon, elev)
    rng = np.random.default_rng(args.seed)

    print(f"Evaluating {args.n_flights} fresh flights (engine={engine_name(args.engine)})\n")

    # Bin errors by fraction of flight remaining and by phase.
    phase_err: dict[str, list[float]] = {"boost": [], "coast": [], "drogue": [], "main": []}
    final_errs = []

    for _ in range(args.n_flights):
        ws = rng.uniform(2, 11)
        wd = np.radians(rng.uniform(0, 360))
        we, wn = -ws * np.sin(wd), -ws * np.cos(wd)
        tr = simulate(
            cfg,
            prefer=args.engine,
            wind_east=float(we),
            wind_north=float(wn),
            dry_mass=float(rng.uniform(14.0, 16.5)),
            drag_multiplier=float(rng.uniform(0.9, 1.1)),
            inclination=float(rng.uniform(83, 89)),
            heading=float(rng.uniform(0, 360)),
            dt=0.5,
        )
        true_lat, true_lon, _ = origin.enu_to_geo(tr.landing_e, tr.landing_n, 0.0)
        pkts = trajectory_to_packets(tr, origin, rate_hz=cfg.telemetry["rate_hz"],
                                     seed=int(rng.integers(1e6)))
        pred = LandingPredictor(cfg, sur, origin)
        last = None
        for pkt in pkts:
            p = pred.process(pkt)
            err = pred.error_against_truth(p, true_lat, true_lon)
            phase_err[p.phase].append(err)
            last = err
        if last is not None:
            final_errs.append(last)

    print("Mean landing error by phase (should decrease):")
    for ph in ["boost", "coast", "drogue", "main"]:
        errs = phase_err[ph]
        if errs:
            print(f"    {ph:7s}: mean={np.mean(errs):6.1f} m   median={np.median(errs):6.1f} m   n={len(errs)}")
    print(f"\nFinal-fix landing error: mean={np.mean(final_errs):.1f} m  "
          f"median={np.median(final_errs):.1f} m  (n={len(final_errs)} flights)")


if __name__ == "__main__":
    main()
