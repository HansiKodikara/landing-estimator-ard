#!/usr/bin/env python3
"""Generate the surrogate training dataset with RocketPy (or the fallback sim).

Example::

    python scripts/generate_dataset.py --n-flights 60 --out data/dataset.npz

Output is a compressed ``.npz`` with the structured sample array plus per-flight
landing points for evaluation.
"""
from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
import numpy as np

from lze.config import load_config
from lze.sim import engine_name
from lze.sim.dataset import generate_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="Path to kronos.yaml")
    ap.add_argument("--n-flights", type=int, default=None, help="Override MC flight count")
    ap.add_argument("--engine", choices=["auto", "rocketpy", "fallback"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt", type=float, default=1.0, help="Sampling interval [s]")
    ap.add_argument(
        "--forecast-wind-noise",
        type=float,
        default=None,
        help="Train a forecast-seeded model: seed each flight's estimator with "
        "its true wind + this Gaussian noise (m/s). Omit for the default "
        "zero-wind-until-chute model. Pair with run_live --wind-seed-* at inference.",
    )
    ap.add_argument("--out", default="data/dataset.npz")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Simulation engine: {engine_name(args.engine)}")
    t0 = time.time()

    def progress(i, n, tr):
        dist = (tr.landing_e ** 2 + tr.landing_n ** 2) ** 0.5
        print(
            f"  flight {i:3d}/{n}  apogee={tr.apogee_alt/0.3048:6.0f} ft  "
            f"land={dist:6.0f} m  samples={len(tr)}"
        )

    res = generate_dataset(
        cfg,
        n_flights=args.n_flights,
        prefer=args.engine,
        seed=args.seed,
        dt=args.dt,
        progress=progress,
        forecast_wind_noise=args.forecast_wind_noise,
    )

    land_e = np.array([tr.landing_e for tr in res.trajectories], dtype=np.float32)
    land_n = np.array([tr.landing_n for tr in res.trajectories], dtype=np.float32)

    np.savez_compressed(
        args.out,
        samples=res.samples,
        flight_id=res.flight_id,   # per-sample; training splits by whole flight
        landing_e=land_e,
        landing_n=land_n,
        engine=res.engine,
    )
    dt = time.time() - t0
    print(
        f"\nWrote {len(res.samples):,} samples from {len(res.trajectories)} flights "
        f"to {args.out} ({dt:.1f}s, engine={res.engine})"
    )


if __name__ == "__main__":
    main()
