#!/usr/bin/env python3
"""Train the landing-drift surrogate from a generated dataset.

Example::

    python scripts/train_model.py --dataset data/dataset.npz --out data/surrogate.joblib
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np

from lze.model.train import train_surrogate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="data/dataset.npz")
    ap.add_argument("--out", default="data/surrogate.joblib")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    samples = data["samples"]
    flight_id = data["flight_id"]
    print(f"Loaded {len(samples):,} samples from {len(np.unique(flight_id))} flights "
          f"(engine={data['engine']})")

    surrogate, report = train_surrogate(
        samples, flight_id, test_frac=args.test_frac, seed=args.seed
    )
    print("\n=== Evaluation (held-out flights) ===")
    print(report.format())

    # Record which simulator taught this model. The fallback engine is a much
    # cruder teacher than RocketPy, and a model must never hide which it was.
    surrogate.metadata["engine"] = str(data["engine"])
    surrogate.metadata["eval"] = {
        "landing_mae_m": report.overall_mae_m,
        "landing_rmse_m": report.overall_rmse_m,
        "baseline_mae_m": report.baseline_mae_m,
        "per_phase_mae_m": report.per_phase_mae_m,
    }
    surrogate.save(args.out)
    print(f"\nSaved surrogate to {args.out}")


if __name__ == "__main__":
    main()
